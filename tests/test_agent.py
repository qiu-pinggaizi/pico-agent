"""Tests for AIAgent conversation loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pico.agent import AIAgent
from pico.config import Config
from pico.llm import LLMProvider, LLMResponse, ToolCall
from pico.memory import NoMemory
from pico.session import MemorylessSession, SessionStore
from pico.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _echo_tool(msg: str = "") -> str:
    return json.dumps({"echo": msg})


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        name="echo",
        toolset="test",
        description="Echo a message",
        parameters={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
        },
        handler=_echo_tool,
    )
    return reg


def _make_config(max_turns: int = 5) -> Config:
    return Config(data={
        "model": {
            "provider": "openai",
            "model": "test-model",
            "api_key": "fake",
            "base_url": "https://fake.test/v1",
        },
        "agent": {
            "max_turns": max_turns,
            "max_tokens": 128000,
            "compression_threshold": 0.80,
        },
    })


class SequentialLLM(LLMProvider):
    """Returns responses in order; repeats last one if exhausted."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self._idx = 0
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]


class AlwaysToolCallLLM(LLMProvider):
    """Always returns a tool call on every chat() invocation."""

    def __init__(self) -> None:
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc_x", name="echo", arguments={"msg": "loop"})],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimpleResponse:
    """test_simple_response — mock LLM returns text → agent returns it."""

    def test_simple_response(self, tmp_path: Path) -> None:
        config = _make_config()
        registry = _build_registry()
        session = MemorylessSession()
        memory = NoMemory()

        llm = SequentialLLM([LLMResponse(content="Hello from LLM")])

        with patch("pico.agent.create_provider", return_value=llm):
            agent = AIAgent(
                config=config,
                tool_registry=registry,
                session=session,
                memory=memory,
            )

        result = agent.run("Hi there")
        assert result == "Hello from LLM"
        assert llm.call_count == 1


class TestToolCallLoop:
    """test_tool_call_loop — LLM returns tool_call → agent dispatches → gets tool result → LLM returns text."""

    def test_tool_call_loop(self, tmp_path: Path) -> None:
        config = _make_config()
        registry = _build_registry()
        session = MemorylessSession()
        memory = NoMemory()

        # First call: return a tool call; second call: return text
        tool_call = ToolCall(id="tc_1", name="echo", arguments={"msg": "ping"})
        llm = SequentialLLM([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="Tool was called, done!"),
        ])

        with patch("pico.agent.create_provider", return_value=llm):
            agent = AIAgent(
                config=config,
                tool_registry=registry,
                session=session,
                memory=memory,
            )

        result = agent.run("Use the echo tool")
        assert result == "Tool was called, done!"
        assert llm.call_count == 2


class TestMaxTurns:
    """test_max_turns — mock LLM always returns tool_calls, verify agent stops."""

    def test_max_turns(self, tmp_path: Path) -> None:
        max_turns = 3
        config = _make_config(max_turns=max_turns)
        registry = _build_registry()
        session = MemorylessSession()
        memory = NoMemory()

        llm = AlwaysToolCallLLM()

        with patch("pico.agent.create_provider", return_value=llm):
            agent = AIAgent(
                config=config,
                tool_registry=registry,
                session=session,
                memory=memory,
            )

        result = agent.run("Loop forever")
        # The exhausted message is in English now
        assert "maximum" in result.lower() or "iterations" in result.lower() or "retry" in result.lower()
        assert llm.call_count == max_turns
