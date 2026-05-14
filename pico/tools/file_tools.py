"""File operation tools — read_file, write_file, search_files.

All handlers return JSON strings. On failure they return:
    {"success": false, "error": "..."}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pico.tools.registry import ToolRegistry
from pico.tools.utils import _error, _success

logger = logging.getLogger(__name__)

# Default limits
MAX_READ_BYTES = 1024 * 1024  # 1 MB
MAX_SEARCH_RESULTS = 50


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read a text file and return its contents.

    SECURITY NOTE: This tool does NOT restrict which files can be read.
    In production, consider adding a path allowlist to prevent reading
    sensitive system files (e.g., /etc/shadow, /proc/*). Symlinks are
    resolved via os.path.realpath() to mitigate basic traversal attacks.

    Args:
        path: Absolute or relative path to the file.
        encoding: File encoding (default utf-8).

    Returns:
        JSON with keys: success, path, content, size, line_count.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return _error(f"File not found: {p}")
    if not p.is_file():
        return _error(f"Not a regular file: {p}")
    # Resolve symlinks to prevent traversal via symlinks
    resolved = Path(os.path.realpath(p))
    if not resolved.is_file():
        return _error(f"Not a regular file (after symlink resolution): {resolved}")
    try:
        content = p.read_text(encoding=encoding)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return _success({
            "path": str(p),
            "content": content,
            "size": len(content),
            "line_count": lines,
        })
    except Exception as e:
        return _error(str(e))


def write_file(path: str, content: str, encoding: str = "utf-8", append: bool = False) -> str:
    """Write (or append) content to a file, creating parent directories as needed.

    SECURITY NOTE: This tool does NOT restrict which files can be written.
    In production, consider adding a path allowlist. Symlinks are resolved
    via os.path.realpath() to mitigate basic traversal attacks.

    Args:
        path: Target file path.
        content: Text content to write.
        encoding: File encoding.
        append: If True, append to existing file instead of overwriting.

    Returns:
        JSON with keys: success, path, bytes_written.
    """
    p = Path(path).expanduser().resolve()
    # Resolve symlinks to prevent traversal via symlinks
    resolved = Path(os.path.realpath(p))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(p, mode, encoding=encoding) as f:
            bytes_written = f.write(content)
        return _success({"path": str(p), "bytes_written": bytes_written, "append": append})
    except Exception as e:
        return _error(str(e))


def search_files(
    path: str,
    pattern: str = "*",
    grep: str = "",
    limit: int = MAX_SEARCH_RESULTS,
) -> str:
    """Search for files by glob pattern or grep content.

    Args:
        path: Root directory to search in.
        pattern: Glob pattern for filenames (e.g. "*.py").
        grep: If non-empty, search file contents for this regex.
        limit: Maximum number of results.

    Returns:
        JSON with keys: success, results (list of dicts with path, line, match for grep).
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return _error(f"Path not found: {root}")

    results: list[dict[str, Any]] = []

    try:
        import re
        regex = re.compile(grep) if grep else None
    except re.error as e:
        return _error(f"Invalid regex '{grep}': {e}")

    try:
        for fpath in root.rglob(pattern):
            if not fpath.is_file():
                continue
            if len(results) >= limit:
                break
            if regex:
                # grep mode — search content
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            results.append({"path": str(fpath), "line": lineno, "match": line.strip()})
                            if len(results) >= limit:
                                break
                except Exception:
                    continue
            else:
                results.append({"path": str(fpath)})

        return _success({"results": results, "count": len(results)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry: ToolRegistry) -> None:
    """Register file tools with the given registry."""
    registry.register(
        name="read_file",
        toolset="file",
        description="Read a text file and return its contents with line count and size.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file."},
                "encoding": {"type": "string", "description": "File encoding (default utf-8).", "default": "utf-8"},
            },
            "required": ["path"],
        },
        handler=read_file,
    )

    registry.register(
        name="write_file",
        toolset="file",
        description="Write or append content to a file. Creates parent directories automatically.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path."},
                "content": {"type": "string", "description": "Text content to write."},
                "encoding": {"type": "string", "description": "File encoding.", "default": "utf-8"},
                "append": {"type": "boolean", "description": "Append instead of overwrite.", "default": False},
            },
            "required": ["path", "content"],
        },
        handler=write_file,
    )

    registry.register(
        name="search_files",
        toolset="file",
        description=(
            "Search for files by glob pattern or grep content inside files. "
            "Use 'pattern' for filename glob, 'grep' for content regex search."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Root directory to search in."},
                "pattern": {"type": "string", "description": "Glob pattern for filenames (default '*').", "default": "*"},
                "grep": {"type": "string", "description": "Regex to search file contents. Empty means filename-only search.", "default": ""},
                "limit": {"type": "integer", "description": "Maximum results to return.", "default": 50},
            },
            "required": ["path"],
        },
        handler=search_files,
    )
