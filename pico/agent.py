"""Core agent conversation loop.

The AIAgent drives the LLM ↔ tool-call cycle:
    while turn < max_turns:
        maybe compress → call LLM → if tool_calls → dispatch & continue
                                    → else → return text
"""

from __future__ import annotations

import logging
from typing import Any

from pico.compression import ContextCompressor
from pico.config import Config
from pico.llm import LLMProvider, create_provider
from pico.memory import Memory, NoMemory
from pico.session import MemorylessSession, SessionStore
from pico.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
            # Compress if context is getting too large
            messages = self.compressor.maybe_compress(messages)

            logger.debug("LLM turn %d/%d, %d messages", turn + 1, self.config.max_turns, len(messages))

            response = self.llm.chat(
                messages=messages,
                tools=self.tools.get_schemas(),
                system=system_prompt,
            )

            # Log token usage
            if response.usage:
                logger.info("Turn %d usage: %s", turn + 1, response.usage)

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

                for tc in response.tool_calls:
                    logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
                    result = self.tools.dispatch(tc.name, tc.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
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
        exhausted_msg = "达到最大迭代次数，请重试。"
        self.session.add_message(self.session_id, "assistant", exhausted_msg)
        return exhausted_msg

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt including memory context."""
        parts: list[str] = [
            "You are Pico Agent — an AI assistant that GETS THINGS DONE with minimal user effort.\n\n"
            "## Core Principle\n"
            "You do the work, not the user. When a user gives a vague request, "
            "you figure out the details and execute step by step. Never ask the user "
            "to do things manually that you can do with tools.\n\n"
            "## Available Tools\n"
            "File: `read_file`, `write_file`, `search_files`\n"
            "Terminal: `terminal`\n"
            "Web: `web_search`, `web_extract`\n"
            "Vision: `vision_analyze`\n"
            "Dataset: `dataset_search`, `dataset_download`, `dataset_info`\n"
            "Code: `code_search`, `code_clone`, `code_install`, `code_browse`, `code_exec`, `code_list`\n"
            "Training (easy): `auto_train`, `quick_eval`, `deploy_model`\n"
            "Training (advanced): `yolo_config`, `train_start`, `train_monitor`, `evaluate`, `bad_cases`\n"
            "Annotation: `sam_annotate`, `convert_annotation`\n"
            "Export: `export_onnx`, `export_trt`, `benchmark`\n"
            "Remote: `remote_terminal`, `remote_file_upload`, `remote_file_download`, `remote_file_sync`, `remote_server_info`\n\n"
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
            "- Use dry_run=true for destructive operations first if unsure.",
        ]

        memory_content = self.memory.load()
        if memory_content:
            parts.append(f"\n## Cross-session Memory\n{memory_content}")

        return "\n\n".join(parts)
