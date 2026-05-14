"""Sub-agent delegation for isolated task execution.

Creates a temporary AIAgent with its own context (no memory, no persistent session)
to handle a specific goal, returning only the final summary.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pico.memory import NoMemory
from pico.session import MemorylessSession

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def delegate_task(
    goal: str,
    context: str = "",
    toolsets: list[str] | None = None,
    config: Any = None,
    tool_registry: Any = None,
) -> str:
    """Create an isolated sub-agent to execute a task.

    The sub-agent has no memory and does not persist its session.
    Only the final text response is returned.

    Args:
        goal: High-level description of what to accomplish.
        context: Additional context (e.g. data from the parent agent).
        toolsets: List of toolset names to restrict available tools.
                  None means inherit the parent's full registry.
        config: The pico Config instance (shared with parent).
        tool_registry: The parent ToolRegistry instance.

    Returns:
        The sub-agent's final text response.

    Raises:
        RuntimeError: If config or tool_registry is not provided.
    """
    # Import here to avoid circular imports
    from pico.agent import AIAgent
    from pico.tools.registry import ToolRegistry

    if config is None or tool_registry is None:
        raise RuntimeError("delegate_task requires config and tool_registry arguments")

    # Build a filtered registry if toolsets specified
    if toolsets is not None:
        child_registry = ToolRegistry()
        for tool in tool_registry._tools.values():
            if tool.toolset in toolsets:
                child_registry._tools[tool.name] = tool
    else:
        child_registry = tool_registry

    child_session = MemorylessSession()
    child_memory = NoMemory()

    logger.info("Delegating task: %s (toolsets=%s)", goal[:80], toolsets)

    child = AIAgent(
        config=config,
        tool_registry=child_registry,
        session=child_session,
        memory=child_memory,
        session_id="",
    )

    prompt = goal
    if context:
        prompt = f"{goal}\n\nContext:\n{context}"

    result = child.run(prompt)

    logger.info("Delegation complete: %d chars", len(result))
    return result
