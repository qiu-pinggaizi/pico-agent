"""Remote terminal execution tool.

Wraps :class:`SSHClient.execute` into a registered tool handler.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


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

        # wrap in cd if work_dir specified
        if work_dir:
            command = f"cd {work_dir} && {command}"

        result = client.execute(command, timeout=timeout)

        output = result["stdout"]
        if result["stderr"]:
            output += f"\n--- stderr ---\n{result['stderr']}"

        return json.dumps({
            "success": result["exit_code"] == 0,
            "output": output,
            "exit_code": result["exit_code"],
        })
    except Exception as e:
        logger.exception("remote_terminal failed")
        return json.dumps({"success": False, "error": str(e)})
