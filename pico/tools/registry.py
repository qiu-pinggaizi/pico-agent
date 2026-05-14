"""Tool registry — registration, schema generation, and dispatch.

The ToolRegistry stores tool definitions (name, toolset, schema, handler, check_fn)
and provides methods to:
  - register tools programmatically
  - auto-discover tools from Python modules in the tools/ directory
  - export OpenAI-compatible function schemas
  - dispatch tool calls by name with error handling
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Internal tool definition.

    Attributes:
        name: Unique tool name used in LLM tool calls.
        toolset: Logical grouping (e.g. "file", "terminal", "web", "vision").
        description: Human-readable description shown to the LLM.
        parameters: JSON Schema for the tool's input parameters.
        handler: Callable that implements the tool. Must return a JSON string.
        check_fn: Optional callable returning True if the tool is available.
    """

    name: str
    toolset: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    check_fn: Callable[[], bool] | None = None


def _error_json(message: str) -> str:
    """Build a standard error JSON string."""
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


class ToolRegistry:
    """Central registry for all agent tools.

    Usage::

        registry = ToolRegistry()
        registry.register(
            name="read_file",
            toolset="file",
            description="Read a file's contents",
            parameters={...},
            handler=read_file_handler,
        )
        schemas = registry.get_schemas()
        result = registry.dispatch("read_file", {"path": "/tmp/x.txt"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(
        self,
        name: str = "",
        toolset: str = "",
        description: str = "",
        parameters: dict[str, Any] | None = None,
        handler: Callable[..., str] | None = None,
        check_fn: Callable[[], bool] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool.

        Supports two calling conventions:

        1. Explicit: ``register(name=..., toolset=..., description=..., parameters=..., handler=...)``
        2. Schema dict: ``register(name=..., toolset=..., schema={"name": ..., "description": ..., "parameters": ...}, handler=...)``

        Args:
            name: Unique tool name.
            toolset: Logical grouping name.
            description: Description for the LLM (ignored if *schema* provided).
            parameters: JSON Schema for input parameters (ignored if *schema* provided).
            handler: Implementation callable; must return a JSON string.
            check_fn: Optional availability check. If it returns False the
                      tool is registered but disabled at dispatch time.
            schema: Alternative to description+parameters; a dict with keys
                    ``name``, ``description``, ``parameters``.
        """
        if schema:
            description = schema.get("description", description)
            parameters = schema.get("parameters", parameters or {"type": "object", "properties": {}})
        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        if handler is None:
            raise ValueError(f"handler is required for tool '{name}'")

        self._tools[name] = ToolDef(
            name=name,
            toolset=toolset,
            description=description,
            parameters=parameters,
            handler=handler,
            check_fn=check_fn,
        )
        logger.debug("Registered tool: %s (toolset=%s)", name, toolset)

    # ------------------------------------------------------------------
    # schemas
    # ------------------------------------------------------------------
    def get_schemas(self, toolsets: list[str] | None = None) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas.

        Args:
            toolsets: If provided, only return tools belonging to these toolsets.

        Returns:
            List of tool schema dicts suitable for the ``tools`` parameter of
            an OpenAI chat completion request.
        """
        schemas: list[dict[str, Any]] = []
        for td in self._tools.values():
            if toolsets and td.toolset not in toolsets:
                continue
            if td.check_fn and not td.check_fn():
                continue
            schemas.append({
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            })
        return schemas

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name and return the JSON result string.

        Args:
            name: Registered tool name.
            args: Arguments dictionary.

        Returns:
            JSON string produced by the tool handler.
            On any error, returns ``{"success": false, "error": "..."}``.
        """
        td = self._tools.get(name)
        if td is None:
            logger.warning("Unknown tool requested: %s", name)
            return _error_json(f"Unknown tool: {name}")

        if td.check_fn and not td.check_fn():
            logger.warning("Tool %s is not available (check failed)", name)
            return _error_json(f"Tool '{name}' is not available on this system")

        try:
            logger.debug("Dispatching tool %s with args %s", name, args)
            # Support two handler signatures:
            #   1. handler(**kwargs) — individual parameters
            #   2. handler(args_dict) — single dict parameter
            sig = inspect.signature(td.handler)
            params = list(sig.parameters.values())
            if len(params) == 1 and params[0].kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ) and params[0].name in ("args", "kwargs"):
                result = td.handler(args)
            else:
                result = td.handler(**args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            return result
        except TypeError as e:
            logger.error("Tool %s bad arguments: %s", name, e)
            return _error_json(f"Invalid arguments for '{name}': {e}")
        except Exception as e:
            logger.exception("Tool %s execution failed", name)
            return _error_json(f"Tool '{name}' failed: {e}")

    # ------------------------------------------------------------------
    # auto-discovery
    # ------------------------------------------------------------------
    def discover(self, tools_dir: str | Path | None = None) -> None:
        """Auto-import and register tools from Python modules.

        Scans ``*.py`` files *and* sub-packages inside *tools_dir*.
        Each module/package is expected to expose either ``register(registry)``
        or ``register_tools(registry)``.

        Sub-packages (directories with ``__init__.py``) are imported as
        ``pico.tools.<dirname>`` and their ``register_tools()`` or
        ``register()`` function is called.

        Args:
            tools_dir: Directory to scan. Defaults to the ``pico/tools`` package
                       directory.
        """
        if tools_dir is None:
            tools_dir = Path(__file__).parent
        tools_dir = Path(tools_dir)

        logger.debug("Discovering tools in %s", tools_dir)

        # 1. Register direct .py modules
        for mod_path in sorted(tools_dir.glob("*.py")):
            if mod_path.name.startswith("_"):
                continue

            module_name = f"pico.tools.{mod_path.stem}"
            try:
                mod = importlib.import_module(module_name)
                reg_fn = getattr(mod, "register", None) or getattr(mod, "register_tools", None)
                if reg_fn and callable(reg_fn):
                    reg_fn(self)
                    logger.debug("Called register() on %s", module_name)
            except Exception as e:
                logger.warning("Failed to import tool module %s: %s", module_name, e)

        # 2. Register sub-packages (detection/, remote/, etc.)
        for subdir in sorted(tools_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name.startswith("_"):
                continue
            init_file = subdir / "__init__.py"
            if not init_file.exists():
                continue

            pkg_name = f"pico.tools.{subdir.name}"
            try:
                mod = importlib.import_module(pkg_name)
                reg_fn = getattr(mod, "register", None) or getattr(mod, "register_tools", None)
                if reg_fn and callable(reg_fn):
                    reg_fn(self)
                    logger.debug("Called register_tools() on %s", pkg_name)
            except Exception as e:
                logger.warning("Failed to import tool package %s: %s", pkg_name, e)
