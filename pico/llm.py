"""LLM Provider abstraction — OpenAI-compatible and Anthropic backends.

All providers implement the LLMProvider ABC and return LLMResponse objects
with content, tool_calls, and token usage information.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents an LLM tool/function call.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to invoke.
        arguments: Parsed arguments dictionary.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: Text content of the response (may be empty if tool_calls present).
        tool_calls: List of tool calls requested by the model.
        usage: Token usage dict with keys like prompt_tokens, completion_tokens, total_tokens.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: Conversation message list.
            tools: Tool schemas in OpenAI function-calling format.
            system: System prompt (handled differently per provider).

        Returns:
            LLMResponse with content, tool_calls, and usage.
        """


def _is_retryable_error(exc: Exception) -> bool:
    """Return True if the error is transient and worth retrying."""
    exc_str = str(exc).lower()
    # Rate limit (429)
    if "429" in exc_str or "rate limit" in exc_str or "too many requests" in exc_str:
        return True
    # Server errors (5xx)
    if "500" in exc_str or "502" in exc_str or "503" in exc_str or "504" in exc_str:
        return True
    # Timeout / connection errors
    if "timeout" in exc_str or "timed out" in exc_str or "connection" in exc_str:
        return True
    # Overloaded
    if "overloaded" in exc_str or "capacity" in exc_str:
        return True
    return False


def _call_with_retry(fn, *, max_retries: int = 3, base_delay: float = 1.0):
    """Call *fn()* with exponential backoff on transient errors.

    Non-retryable errors (auth, bad request) are raised immediately.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable_error(e):
                raise
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, max_retries, e, delay,
                )
                time.sleep(delay)
    # All retries exhausted
    raise last_exc  # type: ignore[misc]


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_prefix: (input_per_mtok, output_per_mtok) in USD
    "claude-sonnet": (3.0, 15.0),
    "claude-opus": (15.0, 75.0),
    "claude-haiku": (0.8, 4.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
    "deepseek": (0.27, 1.1),
    "qwen": (0.3, 0.6),
}


def estimate_cost(model: str, usage: dict[str, int]) -> float:
    """Estimate cost in USD from model name and token usage dict.

    Args:
        model: Model name/identifier.
        usage: Dict with ``prompt_tokens`` and ``completion_tokens`` keys.

    Returns:
        Estimated cost in USD.
    """
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    model_lower = model.lower()
    input_rate = 1.0
    output_rate = 3.0
    for prefix, (in_price, out_price) in MODEL_PRICING.items():
        if prefix in model_lower:
            input_rate = in_price
            output_rate = out_price
            break
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider.

    Works with OpenAI, DeepSeek, vLLM, and any API following the
    OpenAI chat completions format.
    """

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client: Any = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Build message list with system prompt
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        logger.debug("OpenAI chat request: model=%s, messages=%d, tools=%d",
                      self.model, len(api_messages), len(tools or []))

        response = _call_with_retry(lambda: self._client.chat.completions.create(**kwargs))
        choice = response.choices[0]

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args if isinstance(args, dict) else {},
                ))

        # Extract usage
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        content = choice.message.content or ""

        logger.debug("OpenAI response: content_len=%d, tool_calls=%d, usage=%s",
                      len(content), len(tool_calls), usage)

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str, api_key: str, base_url: str = "", max_tokens: int = 4096) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._client: Any = None

    def _convert_tools_for_anthropic(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool schemas to Anthropic format."""
        anthropic_tools = []
        for t in tools:
            # t is {"name": ..., "description": ..., "parameters": ...}
            anthropic_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def _convert_messages_for_anthropic(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separate system from user/assistant messages and convert tool result format."""
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "system":
                system_parts.append(msg.get("content", ""))
                i += 1
                continue

            if role == "tool":
                # Merge consecutive tool results into previous assistant's tool_use
                # Actually, Anthropic expects tool_result as a user message
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                    ],
                })
                i += 1
                continue

            if role == "assistant" and msg.get("tool_calls"):
                # Convert assistant message with tool_calls to Anthropic format
                content_blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                converted.append({"role": "assistant", "content": content_blocks})
                i += 1
                continue

            # Standard user or assistant message
            converted.append({"role": role, "content": msg.get("content", "")})
            i += 1

        system_text = "\n".join(system_parts) if system_parts else None
        return converted, [{"type": "text", "text": system_text}] if system_text else []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
            )

        # Build system prompt from separate param and messages
        all_system_parts: list[str] = []
        if system:
            all_system_parts.append(system)

        converted_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                all_system_parts.append(msg.get("content", ""))
                continue

            if role == "tool":
                converted_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })
                continue

            if role == "assistant" and msg.get("tool_calls"):
                content_blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                converted_messages.append({"role": "assistant", "content": content_blocks})
                continue

            converted_messages.append({"role": role, "content": msg.get("content", "")})

        system_text = "\n".join(all_system_parts) if all_system_parts else anthropic.NOT_GIVEN

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
            "system": system_text,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = self._convert_tools_for_anthropic(tools)

        logger.debug("Anthropic chat request: model=%s, messages=%d, tools=%d",
                      self.model, len(converted_messages), len(tools or []))

        response = _call_with_retry(lambda: self._client.messages.create(**kwargs))

        content_text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
            # Other block types (thinking, etc.) are silently skipped

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        logger.debug("Anthropic response: content_len=%d, tool_calls=%d, usage=%s",
                      len(content_text), len(tool_calls), usage)

        return LLMResponse(content=content_text, tool_calls=tool_calls, usage=usage)


def create_provider(config: Any) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: Config instance with provider, model, api_key, base_url attributes.

    Returns:
        Concrete LLMProvider instance.

    Raises:
        ValueError: If provider is unknown.
    """
    provider_name = config.provider.lower()
    if provider_name == "openai":
        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    elif provider_name == "anthropic":
        return AnthropicProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=4096,  # API output limit, not context budget
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
