"""Tests for pico.compression — context window compression."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pico.compression import (
    ContextCompressor,
    _estimate_tokens,
    _messages_token_count,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Test _estimate_tokens function."""

    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 0

    def test_short_string(self) -> None:
        # 4 chars → 1 token
        assert _estimate_tokens("abcd") == 1

    def test_rounding(self) -> None:
        # Integer division: 5 chars → 1 token
        assert _estimate_tokens("abcde") == 1
        # 8 chars → 2 tokens
        assert _estimate_tokens("abcdefgh") == 2

    def test_long_string(self) -> None:
        text = "a" * 1000
        assert _estimate_tokens(text) == 250


class TestMessagesTokenCount:
    """Test _messages_token_count function."""

    def test_empty_messages(self) -> None:
        assert _messages_token_count([]) == 0

    def test_single_message(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        # "hello" = 5 chars → 1 token
        assert _messages_token_count(msgs) == 1

    def test_multiple_messages(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},      # 5 chars → 1
            {"role": "assistant", "content": "world!"}, # 6 chars → 1
        ]
        assert _messages_token_count(msgs) == 2

    def test_content_as_list_of_blocks(self) -> None:
        """Messages with content as list of dicts (Anthropic format)."""
        msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": "hello world"}],
        }]
        # "hello world" = 11 chars → 2 tokens
        assert _messages_token_count(msgs) == 2

    def test_message_with_tool_calls(self) -> None:
        """Tool calls should be counted."""
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "tc_1",
                "name": "terminal",
                "arguments": '{"command": "ls"}',
            }],
        }]
        # "terminal" = 8 → 2, '{"command": "ls"}' = 17 → 4, total = 6
        total = _messages_token_count(msgs)
        assert total > 0

    def test_message_with_no_content(self) -> None:
        """Message without content key should count as 0."""
        msgs = [{"role": "assistant"}]
        assert _messages_token_count(msgs) == 0


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------


class TestCompressSmallContext:
    """Compression should NOT trigger when context is small."""

    def test_small_context_unchanged(self) -> None:
        mock_llm = MagicMock()
        compressor = ContextCompressor(llm=mock_llm, max_tokens=1000, threshold=0.80)

        messages = [
            {"role": "user", "content": "hi"},       # ~0 tokens
            {"role": "assistant", "content": "hello"}, # ~1 token
        ]

        result = compressor.maybe_compress(messages)
        assert result is messages  # Same object returned (no copy)
        mock_llm.chat.assert_not_called()

    def test_exactly_at_threshold_unchanged(self) -> None:
        """Messages at exactly the threshold should NOT be compressed."""
        mock_llm = MagicMock()
        # max_tokens=40, threshold=0.80 → limit=32
        # 32 tokens = 128 chars
        compressor = ContextCompressor(llm=mock_llm, max_tokens=40, threshold=0.80)

        messages = [{"role": "user", "content": "a" * 128}]  # 128 chars → 32 tokens
        result = compressor.maybe_compress(messages)
        assert result is messages


class TestCompressLargeContext:
    """Compression SHOULD trigger when context exceeds threshold."""

    def test_large_context_triggers_compression(self) -> None:
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Summary of old conversation"
        mock_llm.chat.return_value = mock_resp

        # max_tokens=100, threshold=0.80 → limit=80
        compressor = ContextCompressor(llm=mock_llm, max_tokens=100, threshold=0.80)

        # 400 chars each → 100 tokens each. 4 messages → 400 tokens > 80 limit
        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
            {"role": "user", "content": "c" * 400},
            {"role": "assistant", "content": "d" * 400},
        ]

        result = compressor.maybe_compress(messages)
        # Should have: [summary_msg] + kept_messages
        assert len(result) < len(messages)
        assert result[0]["role"] == "system"
        assert "Conversation Summary" in result[0]["content"]
        mock_llm.chat.assert_called_once()

    def test_compression_preserves_recent_messages(self) -> None:
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Summary text"
        mock_llm.chat.return_value = mock_resp

        compressor = ContextCompressor(llm=mock_llm, max_tokens=100, threshold=0.80)

        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
            {"role": "user", "content": "c" * 400},
            {"role": "assistant", "content": "d" * 400},
        ]

        result = compressor.maybe_compress(messages)
        # The last messages should be preserved
        # mid = 4//2 = 2, so messages[2:] is kept
        kept = result[1:]  # Everything except summary
        assert kept == messages[2:]

    def test_llm_failure_handled_gracefully(self) -> None:
        """If LLM raises, compression should still work (fallback message)."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("API error")

        compressor = ContextCompressor(llm=mock_llm, max_tokens=100, threshold=0.80)

        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
            {"role": "user", "content": "c" * 400},
            {"role": "assistant", "content": "d" * 400},
        ]

        result = compressor.maybe_compress(messages)
        assert len(result) < len(messages)
        assert "Compression failed" in result[0]["content"] or "Conversation Summary" in result[0]["content"]

    def test_few_messages_no_compression_even_if_large(self) -> None:
        """If mid < 2, don't compress (not enough messages to split)."""
        mock_llm = MagicMock()
        compressor = ContextCompressor(llm=mock_llm, max_tokens=10, threshold=0.80)

        # Only 2 messages, mid = 1, so mid < 2 → no compression
        messages = [
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 1000},
        ]

        result = compressor.maybe_compress(messages)
        assert result is messages
