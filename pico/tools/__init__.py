"""Pico Agent tools package.

Auto-discovers and registers all tool modules in this directory
when the agent starts up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pico.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def discover_and_register(registry: "ToolRegistry") -> None:
    """Discover and register all tools from the tools/ directory.

    This is the main entry point called by the agent during startup.

    Args:
        registry: The ToolRegistry instance to populate.
    """
    tools_dir = Path(__file__).parent
    registry.discover(tools_dir)
    logger.info("Tool discovery complete: %d tools registered", len(registry._tools))
