"""Tests for ToolRegistry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pico.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _echo_handler(**kwargs: Any) -> str:
    """Trivial handler that echoes its arguments as JSON."""
    return json.dumps(kwargs)


def _greet_handler(name: str = "world") -> str:
    return json.dumps({"greeting": f"hello {name}"})


def _always_fail_check() -> bool:
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterAndDispatch:
    """test_register_and_dispatch — register a simple tool, dispatch, verify result."""

    def test_register_and_dispatch(self) -> None:
        registry = ToolRegistry()
        registry.register(
            name="echo",
            toolset="test",
            description="Echo args back",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            handler=_echo_handler,
        )
        result = json.loads(registry.dispatch("echo", {"msg": "hi"}))
        assert result == {"msg": "hi"}

    def test_register_with_schema(self) -> None:
        """Test the schema= calling convention."""
        registry = ToolRegistry()
        schema = {
            "name": "greet",
            "description": "Say hello",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }
        registry.register(name="greet", toolset="test", schema=schema, handler=_greet_handler)
        result = json.loads(registry.dispatch("greet", {"name": "alice"}))
        assert result == {"greeting": "hello alice"}


class TestGetSchemas:
    """test_get_schemas — verify OpenAI-compatible schema output."""

    def test_get_schemas(self) -> None:
        registry = ToolRegistry()
        registry.register(
            name="alpha",
            toolset="t1",
            description="Alpha tool",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=_echo_handler,
        )
        registry.register(
            name="beta",
            toolset="t2",
            description="Beta tool",
            parameters={"type": "object", "properties": {}},
            handler=_echo_handler,
        )
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"alpha", "beta"}
        # Each schema has the expected keys
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "parameters" in s

    def test_get_schemas_filtered_by_toolset(self) -> None:
        registry = ToolRegistry()
        registry.register(name="a", toolset="x", description="A", handler=_echo_handler)
        registry.register(name="b", toolset="y", description="B", handler=_echo_handler)
        schemas = registry.get_schemas(toolsets=["x"])
        assert len(schemas) == 1
        assert schemas[0]["name"] == "a"


class TestDispatchErrors:
    """test_dispatch_unknown_tool and test_dispatch_bad_args."""

    def test_dispatch_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = json.loads(registry.dispatch("nonexistent", {}))
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_dispatch_bad_args(self) -> None:
        """Handler that requires a keyword arg — pass unexpected arg to trigger TypeError."""
        registry = ToolRegistry()

        def strict_handler(required_arg: str) -> str:
            return required_arg

        registry.register(
            name="strict",
            toolset="test",
            description="Needs required_arg",
            handler=strict_handler,
        )
        # Pass wrong keyword — should return error JSON
        result = json.loads(registry.dispatch("strict", {"wrong": "value"}))
        assert result["success"] is False
        assert "Invalid arguments" in result["error"]


class TestCheckFn:
    """test_check_fn — disabled tool returns error on dispatch."""

    def test_check_fn_blocks_dispatch(self) -> None:
        registry = ToolRegistry()
        registry.register(
            name="guarded",
            toolset="test",
            description="Blocked tool",
            handler=_echo_handler,
            check_fn=_always_fail_check,
        )
        # Dispatch should fail
        result = json.loads(registry.dispatch("guarded", {}))
        assert result["success"] is False
        assert "not available" in result["error"]

    def test_check_fn_excluded_from_schemas(self) -> None:
        registry = ToolRegistry()
        registry.register(
            name="guarded",
            toolset="test",
            description="Blocked tool",
            handler=_echo_handler,
            check_fn=_always_fail_check,
        )
        registry.register(
            name="open",
            toolset="test",
            description="Open tool",
            handler=_echo_handler,
        )
        schemas = registry.get_schemas()
        names = [s["name"] for s in schemas]
        assert "guarded" not in names
        assert "open" in names
