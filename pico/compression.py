"""Context window compression.

Simple token estimation (len(text) / 4) and LLM-based summarization
of older messages when the context window exceeds the configured threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _LLMCallable(Protocol):
    """Minimal protocol satisfied by LLMProvider.chat."""

    def chat(self, messages: list[dict[str, Any]], tools: Any = None, system: str | None = None) -> Any: ...


def _estimate_tokens(text: str) -> int:
    """Estimate token count — ~4 chars per token (conservative)."""
    return len(text) // 4


def _messages_token_count(messages: list[dict[str, Any]]) -> int:
    """Sum estimated tokens over all messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += _estimate_tokens(str(block.get("text", "")))
    return total


class ContextCompressor:
    """Compress message history when it exceeds the token budget.

    Args:
        llm: LLM provider used for summarization.
        max_tokens: Total token budget.
        threshold: Fraction (0..1) of max_tokens that triggers compression.
    """

    def __init__(self, llm: _LLMCallable, max_tokens: int = 128000, threshold: float = 0.80) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.threshold = threshold

    def maybe_compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compress the message list if it exceeds the threshold.

        Strategy:
          1. Keep the most recent ~40 % of messages unchanged.
          2. Ask the LLM to summarize the older messages.
          3. Replace the old block with a single summary message.

        Args:
            messages: Current conversation messages.

        Returns:
            (Possibly compressed) message list.
        """
        current_tokens = _messages_token_count(messages)
        limit = int(self.max_tokens * self.threshold)

        if current_tokens <= limit:
            return messages

        logger.info(
            "Context compression triggered: %d tokens > %d limit (%.0f%% of %d)",
            current_tokens, limit, self.threshold * 100, self.max_tokens,
        )

        # Split: older half to summarise, newer half to keep
        mid = len(messages) // 2
        # Ensure we keep at least a few messages
        if mid < 2:
            return messages

        old_messages = messages[:mid]
        kept_messages = messages[mid:]

        summary_text = self._summarize(old_messages)
        logger.debug("Compression summary (%d chars from %d messages)",
                      len(summary_text), len(old_messages))

        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": f"[对话摘要]\n{summary_text}",
        }

        return [summary_msg] + kept_messages

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        """Ask the LLM to produce a concise summary of the given messages."""
        conversation_lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if content:
                conversation_lines.append(f"{role}: {content}")

        transcript = "\n".join(conversation_lines)
        prompt = (
            "请用简洁的中文总结以下对话的关键信息，保留重要的事实、决定和待办事项。\n"
            "不要添加额外的解释，只输出摘要内容。\n\n"
            f"对话内容：\n{transcript}"
        )

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                system="你是一个对话摘要助手。请生成简洁准确的摘要。",
            )
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.error("Failed to generate compression summary: %s", e)
            return f"[压缩失败: {e}] 对话共 {len(messages)} 条消息。"
