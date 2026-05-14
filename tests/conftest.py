"""Shared fixtures for Pico Agent tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pico.config import Config
from pico.llm import LLMProvider, LLMResponse, ToolCall
from pico.session import SessionStore


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Config:
    """A Config object with safe test settings (no real API keys)."""
    data = {
        "model": {
            "provider": "openai",
            "model": "test-model",
            "api_key": "test-key-not-real",
            "base_url": "https://fake.api.test/v1",
        },
        "agent": {
            "max_turns": 3,
            "max_tokens": 10000,
            "compression_threshold": 0.80,
        },
        "servers": {},
        "default_server": "",
    }
    return Config(data=data)


class MockLLMProvider(LLMProvider):
    """A mock LLM that returns canned responses.

    Set *responses* to a list of LLMResponse objects that will be returned
    in order on successive calls to ``chat()``.  Set *side_effect* to an
    exception to raise on the next call.
    """

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses = list(responses or [LLMResponse(content="mock reply")])
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        idx = min(self._call_count, len(self.responses) - 1)
        self._call_count += 1
        return self.responses[idx]


@pytest.fixture()
def mock_llm() -> MockLLMProvider:
    """A mock LLMProvider that returns a simple text response."""
    return MockLLMProvider(responses=[LLMResponse(content="mock reply")])


@pytest.fixture()
def tmp_session_db(tmp_path: Path) -> SessionStore:
    """A SessionStore backed by a temporary SQLite database."""
    db_path = tmp_path / "test_sessions.db"
    store = SessionStore(db_path=db_path)
    yield store
    store.close()
