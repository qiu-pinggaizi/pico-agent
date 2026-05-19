"""Remote terminal execution tool.

Wraps :class:`SSHClient.execute` into a registered tool handler.
Enforces workspace sandbox: all commands run inside the configured workspace,
and destructive commands (rm) targeting paths outside workspace are blocked.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that could delete files outside workspace
_RM_PATTERNS = [
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*(/|\.\.)"),  # rm /... or rm .. or rm -rf /
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*~"),          # rm ~/...
    re.compile(r"\brmdir\s+(/|\.\.)"),                 # rmdir /...
    re.compile(r"\bmv\s+.*\s+(/|\.\.)"),               # mv ... /...
]


def _check_delete_safety(command: str, workspace: str | None) -> str | None:
    """Return error message if command tries to delete outside workspace, else None."""
    if not workspace:
        return None
    for pat in _RM_PATTERNS:
        if pat.search(command):
            return (
                f"Blocked: destructive command targets paths outside workspace '{workspace}'. "
                f"Only files inside the workspace can be deleted."
            )
    return None


def remote_terminal_handler(
    command: str = "",
    server: str = "default",
    timeout: int = 300,
    work_dir: str | None = None,
    **_kwargs: Any,
) -> str:
    """Execute a shell command on a remote server.

    Args:
        command: Shell command to execute.
        server: Server name from config.
        timeout: Timeout in seconds.
        work_dir: Working directory for the command.

    Returns:
        JSON string with output and exit_code, or error.
    """
    from pico.tools.remote.ssh_client import get_ssh_client

    if not command:
        return json.dumps({"success": False, "error": "command is required"})

    try:
        client = get_ssh_client(server)
        workspace = client.workspace

        # Check destructive command safety
        block_reason = _check_delete_safety(command, workspace)
        if block_reason:
            return json.dumps({"success": False, "error": block_reason})

        # If work_dir specified, validate it's within workspace
        if work_dir:
            client.check_workspace_path(work_dir)
            command = f"cd {work_dir} && {command}"
        elif workspace:
            # Default: cd into workspace
            command = f"cd {workspace} && {command}"

        result = client.execute(command, timeout=timeout)

        output = result["stdout"]
        if result["stderr"]:
            output += f"\n--- stderr ---\n{result['stderr']}"

        return json.dumps({
            "success": result["exit_code"] == 0,
            "output": output,
            "exit_code": result["exit_code"],
        })
    except PermissionError as e:
        return json.dumps({"success": False, "error": str(e)})
    except ConnectionError as e:
        logger.warning("remote_terminal SSH connection failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.exception("remote_terminal failed")
        return json.dumps({"success": False, "error": str(e)})
