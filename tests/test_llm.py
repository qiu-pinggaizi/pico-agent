"""Tests for LLM providers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pico.llm import (
    AnthropicProvider,
    LLMResponse,
    OpenAIProvider,
    ToolCall,
    create_provider,
)


# ---------------------------------------------------------------------------
# Mock response helpers
# ---------------------------------------------------------------------------


def _make_openai_response(
    content: str = "hello",
    tool_calls: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock OpenAI API response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_anthropic_response(
    content_blocks: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a mock Anthropic API response object."""
    response = MagicMock()
    response.content = content_blocks or []
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseOpenAIResponse:
    """test_parse_openai_response — mock OpenAI API response, verify parsing."""

    def test_text_only(self) -> None:
        mock_resp = _make_openai_response(content="Hi there")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        # OpenAI is imported locally inside OpenAIProvider.chat()
        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(model="gpt-4", api_key="fake")
            result = provider.chat(messages=[{"role": "user", "content": "test"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there"
        assert result.tool_calls == []
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5

    def test_with_tool_calls(self) -> None:
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "read_file"
        tc.function.arguments = '{"path": "/tmp/x.txt"}'

        mock_resp = _make_openai_response(content="", tool_calls=[tc])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(model="gpt-4", api_key="fake")
            result = provider.chat(messages=[{"role": "user", "content": "test"}])

        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/x.txt"}


class TestParseAnthropicResponse:
    """test_parse_anthropic_response — mock Anthropic API response, verify parsing."""

    def test_text_only(self) -> None:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Anthropic says hi"

        mock_resp = _make_anthropic_response(content_blocks=[text_block])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        # anthropic is imported locally inside AnthropicProvider.chat()
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_anthropic_module.NOT_GIVEN = None

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            provider = AnthropicProvider(model="claude-3", api_key="fake")
            result = provider.chat(messages=[{"role": "user", "content": "test"}])

        assert result.content == "Anthropic says hi"
        assert result.tool_calls == []
        assert result.usage["prompt_tokens"] == 10

    def test_with_tool_use(self) -> None:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = ""

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_1"
        tool_block.name = "write_file"
        tool_block.input = {"path": "/tmp/a.txt", "content": "data"}

        mock_resp = _make_anthropic_response(content_blocks=[text_block, tool_block])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_anthropic_module.NOT_GIVEN = None

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            provider = AnthropicProvider(model="claude-3", api_key="fake")
            result = provider.chat(messages=[{"role": "user", "content": "test"}])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "write_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/a.txt", "content": "data"}


class TestCreateProvider:
    """test_create_provider — verify provider factory."""

    @pytest.mark.parametrize(
        "provider_name, expected_cls",
        [
            ("openai", OpenAIProvider),
            ("anthropic", AnthropicProvider),
        ],
    )
    def test_create_known_provider(self, provider_name: str, expected_cls: type) -> None:
        cfg = MagicMock()
        cfg.provider = provider_name
        cfg.model = "test-model"
        cfg.api_key = "fake-key"
        cfg.base_url = "https://fake.test/v1"

        provider = create_provider(cfg)
        assert isinstance(provider, expected_cls)

    def test_create_unknown_provider_raises(self) -> None:
        cfg = MagicMock()
        cfg.provider = "unknown_llm"
        cfg.model = "test-model"
        cfg.api_key = "fake-key"
        cfg.base_url = "https://fake.test/v1"

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider(cfg)
