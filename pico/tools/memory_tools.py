"""Memory tools — allow the LLM to read, write, search, and delete persistent memory.

All handlers return JSON strings.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pico.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _success(data: Any) -> str:
    return json.dumps({"success": True, **(data if isinstance(data, dict) else {"result": data})}, ensure_ascii=False)


def _error(msg: str) -> str:
    return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def memory_read(limit: int = 0, **kwargs: Any) -> str:
    """Read persistent memory content."""
    try:
        from pico.memory import Memory

        memory = Memory()
        content = memory.load()

        if not content.strip():
            return _success({"entries": [], "message": "Memory is empty."})

        entries = [line.strip() for line in content.strip().split("\n") if line.strip()]
        if limit > 0:
            entries = entries[:limit]

        return _success({"entries": entries, "total": len(entries)})
    except Exception as e:
        return _error(str(e))


def memory_write(content: str = "", **kwargs: Any) -> str:
    """Add a memory entry."""
    try:
        if not content or not content.strip():
            return _error("content is required and cannot be empty.")

        from pico.memory import Memory

        memory = Memory()
        memory.add(content.strip())
        return _success({"message": f"Memory entry added: {content.strip()}"})
    except Exception as e:
        return _error(str(e))


def memory_delete(keyword: str = "", **kwargs: Any) -> str:
    """Remove memory entries matching keyword."""
    try:
        if not keyword or not keyword.strip():
            return _error("keyword is required.")

        from pico.memory import Memory

        memory = Memory()
        removed = memory.remove(keyword.strip())
        if removed:
            return _success({"message": f"Removed entries matching '{keyword}'"})
        else:
            return _success({"message": f"No entries matched '{keyword}'"})
    except Exception as e:
        return _error(str(e))


def memory_search(query: str = "", **kwargs: Any) -> str:
    """Search memory entries by keyword."""
    try:
        if not query or not query.strip():
            return _error("query is required.")

        from pico.memory import Memory

        memory = Memory()
        content = memory.load()

        if not content.strip():
            return _success({"matches": [], "message": "Memory is empty."})

        entries = [line.strip() for line in content.strip().split("\n") if line.strip()]
        query_lower = query.lower()
        matches = [e for e in entries if query_lower in e.lower()]

        return _success({
            "matches": matches,
            "total_matches": len(matches),
            "query": query,
        })
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry: ToolRegistry) -> None:
    """Register memory tools with the given registry."""
    registry.register(
        name="memory_read",
        toolset="memory",
        description="Read the current persistent memory content. Returns all stored facts, preferences, and notes about the user and environment.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max number of memory entries to return (0 = all). Default: 0.",
                    "default": 0,
                },
            },
            "required": [],
        },
        handler=memory_read,
    )

    registry.register(
        name="memory_write",
        toolset="memory",
        description=(
            "Add a new entry to persistent memory. Use this to save important facts, "
            "user preferences, environment details, or lessons learned. "
            "Write as a declarative fact, e.g. 'User prefers Python 3.10'. "
            "Do NOT save temporary task state or session progress."
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory entry content to save.",
                },
            },
            "required": ["content"],
        },
        handler=memory_write,
    )

    registry.register(
        name="memory_delete",
        toolset="memory",
        description="Remove memory entries matching a keyword. Use to remove outdated or incorrect facts.",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Keyword to search for in memory entries. All matching entries will be removed.",
                },
            },
            "required": ["keyword"],
        },
        handler=memory_delete,
    )

    registry.register(
        name="memory_search",
        toolset="memory",
        description="Search persistent memory for entries containing specific keywords.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find matching memory entries.",
                },
            },
            "required": ["query"],
        },
        handler=memory_search,
    )
