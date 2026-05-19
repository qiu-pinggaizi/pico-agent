"""Core agent conversation loop.

The AIAgent drives the LLM ↔ tool-call cycle:
    while turn < max_turns:
        maybe compress → call LLM → if tool_calls → dispatch & continue
                                    → else → return text
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

from pico.compression import ContextCompressor
from pico.config import Config
from pico.llm import LLMProvider, create_provider, estimate_cost
from pico.memory import Memory, NoMemory
from pico.session import MemorylessSession, SessionStore
from pico.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default timeout for individual tool calls (seconds)
DEFAULT_TOOL_TIMEOUT = 300


def _validate_message_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure message role alternation (never two assistant or two user in a row).

    Consecutive messages with the same role (excluding 'tool') are merged
    by concatenating their content. This prevents API errors from role violations.
    """
    if not messages:
        return messages
    cleaned: list[dict[str, Any]] = [messages[0]]
    for msg in messages[1:]:
        prev = cleaned[-1]
        # Tool messages can repeat (multiple tool results after one assistant call)
        if msg["role"] == "tool" or prev["role"] == "tool":
            cleaned.append(msg)
        elif msg["role"] == prev["role"]:
            # Merge consecutive same-role messages
            prev["content"] = (prev.get("content", "") or "") + "\n" + (msg.get("content", "") or "")
        else:
            cleaned.append(msg)
    return cleaned


class AIAgent:
    """Central agent that orchestrates LLM calls and tool dispatch.

    Args:
        config: Pico Config instance.
        tool_registry: Populated ToolRegistry with available tools.
        session: SessionStore (or MemorylessSession) for message persistence.
        memory: Memory (or NoMemory) for cross-session context.
        session_id: Active session ID. If empty, a new one will be created on first run.
    """

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry,
        session: SessionStore | MemorylessSession | None = None,
        memory: Memory | NoMemory | None = None,
        session_id: str = "",
    ) -> None:
        self.config = config
        self.tools = tool_registry
        self.session = session or SessionStore()
        self.memory = memory or Memory()
        self.session_id = session_id

        self.llm: LLMProvider = create_provider(config)
        self.compressor = ContextCompressor(
            llm=self.llm,
            max_tokens=config.max_tokens,
            threshold=config.compression_threshold,
        )
        # Interrupt flag for graceful shutdown (set by CLI on Ctrl+C)
        self._interrupt_requested = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Execute one user interaction, which may involve multiple tool-call turns.

        Args:
            user_message: The user's input text.

        Returns:
            The assistant's final text response.
        """
        # Ensure we have a session
        if not self.session_id:
            title = user_message[:50].replace("\n", " ")
            if isinstance(self.session, SessionStore):
                self.session_id = self.session.create_session(title=title)

        # Persist user message
        self.session.add_message(self.session_id, "user", user_message)

        # Build conversation
        messages: list[dict[str, Any]] = []
        if isinstance(self.session, SessionStore):
            history = self.session.get_messages_as_dicts(self.session_id)
            # Drop system messages already in history; we inject them fresh
            messages = [m for m in history if m["role"] != "system"]
        else:
            messages = self.session.get_messages_as_dicts(self.session_id)

        system_prompt = self._build_system_prompt()

        # Agentic loop: LLM ↔ tools
        for turn in range(self.config.max_turns):
            # Check for interrupt
            if self._interrupt_requested:
                break
            # Compress if context is getting too large
            messages = self.compressor.maybe_compress(messages)

            # Validate message role alternation before sending to LLM
            messages = _validate_message_roles(messages)

            logger.debug("LLM turn %d/%d, %d messages", turn + 1, self.config.max_turns, len(messages))

            response = self.llm.chat(
                messages=messages,
                tools=self.tools.get_schemas(),
                system=system_prompt,
            )

            # Log token usage and record cost
            if response.usage:
                logger.info("Turn %d usage: %s", turn + 1, response.usage)
                if isinstance(self.session, SessionStore) and self.session_id:
                    cost = estimate_cost(self.config.model, response.usage)
                    self.session.record_usage(
                        self.session_id,
                        input_tokens=response.usage.get("prompt_tokens", 0),
                        output_tokens=response.usage.get("completion_tokens", 0),
                        cost=cost,
                    )

            # If there are tool calls, execute them and loop
            if response.tool_calls:
                # Append the assistant message with tool_calls
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # Persist the assistant message with tool_calls
                self.session.add_message(
                    self.session_id, "assistant", response.content or "",
                    tool_calls=[{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
                )

                for tc in response.tool_calls:
                    logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
                    result = self._dispatch_tool_with_timeout(tc.name, tc.arguments)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                    messages.append(tool_msg)
                    # Persist tool result
                    self.session.add_message(self.session_id, "tool", result, tool_call_id=tc.id)
                continue

            # Pure text response — save and return
            final_text = response.content or "(empty response)"
            self.session.add_message(self.session_id, "assistant", final_text)

            # Auto-title session from first user message if using StoreSession
            if isinstance(self.session, SessionStore) and self.session_id:
                sess = self.session.get_session(self.session_id)
                if sess and sess.title == "New Session":
                    title = user_message[:60].replace("\n", " ").strip()
                    self.session.update_session_title(self.session_id, title)

            return final_text

        # Exhausted max turns
        exhausted_msg = "Reached maximum tool-call iterations. Please try again."
        self.session.add_message(self.session_id, "assistant", exhausted_msg)
        return exhausted_msg

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _dispatch_tool_with_timeout(
        self, name: str, args: dict[str, Any], timeout: int = DEFAULT_TOOL_TIMEOUT
    ) -> str:
        """Dispatch a tool call with timeout protection.

        Runs the tool in a thread pool with the given timeout. If the tool
        doesn't finish in time, returns a timeout error JSON.
        """
        import json
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.tools.dispatch, name, args)
                return future.result(timeout=timeout)
        except FutureTimeout:
            logger.warning("Tool %s timed out after %ds", name, timeout)
            return json.dumps({
                "success": False,
                "error": f"Tool '{name}' timed out after {timeout}s. "
                         "The operation may still be running on the server.",
            }, ensure_ascii=False)

    def _build_system_prompt(self) -> str:
        """Build the system prompt including memory context and dynamic tool list."""
        parts: list[str] = [
            "You are Pico Agent — an AI assistant that GETS THINGS DONE with minimal user effort.\n\n"
            "## Core Principle\n"
            "You do the work, not the user. When a user gives a vague request, "
            "you figure out the details and execute step by step. Never ask the user "
            "to do things manually that you can do with tools.\n\n"
        ]

        # Dynamic tool list from registry — always in sync
        tool_list_parts: list[str] = ["## Available Tools\n"]
        schemas = self.tools.get_schemas()
        # Group by prefix for readability
        groups: dict[str, list[str]] = {}
        for schema in schemas:
            name = schema.get("name", "unknown")
            # Extract prefix (first word before underscore or whole name)
            prefix = name.split("_")[0] if "_" in name else name
            groups.setdefault(prefix, []).append(name)

        for prefix, names in sorted(groups.items()):
            display_prefix = prefix.capitalize()
            tool_list_parts.append(f"- **{display_prefix}**: {', '.join(f'`{n}`' for n in sorted(names))}")
        tool_list_parts.append("")

        parts.append("\n".join(tool_list_parts))

        parts.append(
            "## Workflow Patterns\n"
            "When the user says something like 'train a model on my data':\n"
            "1. Call `auto_train(data_dir=<path>)` — it handles everything: detect format, generate YAML, recommend model, start training.\n"
            "2. If dry_run, show the plan, then ask if they want to proceed.\n"
            "3. After training, offer to run `quick_eval` and `deploy_model`.\n\n"
            "When the user says 'find and download a dataset for X':\n"
            "1. Call `dataset_search(query='X', sources='all')`\n"
            "2. Show top results, recommend the best one.\n"
            "3. Call `dataset_download(source=..., name=...)`\n"
            "4. Call `dataset_info` to verify.\n\n"
            "When the user says 'clone and run repo X':\n"
            "1. Call `code_search(query='X')` or use the URL directly.\n"
            "2. Call `code_clone(url=...)`\n"
            "3. Call `code_install(repo_path=...)` — auto-detects deps.\n"
            "4. Call `code_browse` to show structure.\n"
            "5. Read README with `code_browse(target='README.md')`.\n"
            "6. Offer to run examples or tests.\n\n"
            "## Style\n"
            "- Be proactive: chain multiple tool calls without waiting.\n"
            "- Show progress: 'Downloading dataset...' → 'Setting up...' → 'Training started!'\n"
            "- After any major task, summarize what was done and suggest next steps.\n"
            "- If something fails, suggest fixes rather than just reporting the error.\n"
            "- Use dry_run=true for destructive operations first if unsure.\n\n"
            "## Data Integrity (CRITICAL)\n"
            "- NEVER fabricate, hallucinate, or invent data that was not returned by a tool.\n"
            "- If a tool returns an error, 'success: false', or empty output, you MUST:\n"
            "  1. Report the exact error message to the user.\n"
            "  2. Do NOT make up plausible-looking results (metrics, GPU info, file lists, etc.).\n"
            "  3. Do NOT assume the operation succeeded just because you expected it to.\n"
            "- If a tool call times out or the connection fails, say so explicitly.\n"
            "- It is ALWAYS better to say 'I could not get this data' than to fabricate it.",
        )

        memory_content = self.memory.load()
        if memory_content:
            parts.append(f"\n## Cross-session Memory\n{memory_content}")

        return "\n\n".join(parts)
