"""Remote server tools for Pico Agent.

Provides SSH connection management, remote terminal execution,
file transfer, and server status monitoring via paramiko.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from pico.tools.remote.file_transfer import (
    file_download_handler,
    file_sync_handler,
    file_upload_handler,
)
from pico.tools.remote.remote_terminal import remote_terminal_handler
from pico.tools.remote.server_info import server_status_handler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _paramiko_available() -> bool:
    return importlib.util.find_spec("paramiko") is not None


def _has_servers() -> bool:
    """Check if the loaded config contains at least one remote server."""
    try:
        from pico.config import get_config
        cfg = get_config()
        servers = cfg.get("servers") if isinstance(cfg, dict) else getattr(cfg, "servers", {})
        return bool(servers)
    except Exception:
        return False


def _paramiko_available() -> bool:
    return importlib.util.find_spec("paramiko") is not None


def _check_remote() -> bool:
    """Return True if remote tools should be available."""
    if not _paramiko_available():
        return False
    try:
        from pico.config import get_config
        cfg = get_config()
        servers = cfg.servers if hasattr(cfg, "servers") else cfg.get("servers", {})
        return bool(servers)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def register_tools(registry: Any) -> None:  # noqa: C901
    """Register all remote tools into *registry*."""

    check_fn = _check_remote

    # remote_terminal --------------------------------------------------------
    registry.register(
        name="remote_terminal",
        toolset="remote",
        schema={
            "name": "remote_terminal",
            "description": "Execute a shell command on a remote server via SSH.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute remotely.",
                    },
                    "server": {
                        "type": "string",
                        "description": "Server name from config (default: 'default').",
                        "default": "default",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 300).",
                        "default": 300,
                    },
                    "work_dir": {
                        "type": "string",
                        "description": "Working directory for the command.",
                        "default": None,
                    },
                },
                "required": ["command"],
            },
        },
        handler=remote_terminal_handler,
        check_fn=check_fn,
    )

    # file_upload ------------------------------------------------------------
    registry.register(
        name="file_upload",
        toolset="remote",
        schema={
            "name": "file_upload",
            "description": "Upload a local file to a remote server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_path": {"type": "string", "description": "Local file path."},
                    "remote_path": {"type": "string", "description": "Remote file path."},
                    "server": {"type": "string", "description": "Server name.", "default": "default"},
                },
                "required": ["local_path", "remote_path"],
            },
        },
        handler=file_upload_handler,
        check_fn=check_fn,
    )

    # file_download ----------------------------------------------------------
    registry.register(
        name="file_download",
        toolset="remote",
        schema={
            "name": "file_download",
            "description": "Download a file from a remote server to local.",
            "parameters": {
                "type": "object",
                "properties": {
                    "remote_path": {"type": "string", "description": "Remote file path."},
                    "local_path": {"type": "string", "description": "Local file path."},
                    "server": {"type": "string", "description": "Server name.", "default": "default"},
                },
                "required": ["remote_path", "local_path"],
            },
        },
        handler=file_download_handler,
        check_fn=check_fn,
    )

    # file_sync --------------------------------------------------------------
    registry.register(
        name="file_sync",
        toolset="remote",
        schema={
            "name": "file_sync",
            "description": "Synchronize a directory between local and remote (upload or download).",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_dir": {"type": "string", "description": "Local directory path."},
                    "remote_dir": {"type": "string", "description": "Remote directory path."},
                    "direction": {
                        "type": "string",
                        "enum": ["upload", "download"],
                        "description": "Transfer direction.",
                        "default": "upload",
                    },
                    "server": {"type": "string", "description": "Server name.", "default": "default"},
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to exclude.",
                        "default": ["*.pyc", "__pycache__", ".git", ".venv"],
                    },
                },
                "required": ["local_dir", "remote_dir"],
            },
        },
        handler=file_sync_handler,
        check_fn=check_fn,
    )

    # server_status ----------------------------------------------------------
    registry.register(
        name="server_status",
        toolset="remote",
        schema={
            "name": "server_status",
            "description": "Get remote server status: GPU, CPU, memory, disk, and running processes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Server name from config.",
                        "default": "default",
                    },
                },
            },
        },
        handler=server_status_handler,
        check_fn=check_fn,
    )
