"""Terminal command execution tool.

Runs shell commands locally with configurable timeout and optional background mode.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any

from pico.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Track background processes
_bg_processes: dict[str, subprocess.Popen] = {}
_bg_counter = 0


def _success(data: Any) -> str:
    return json.dumps({"success": True, **(data if isinstance(data, dict) else {"result": data})}, ensure_ascii=False)


def _error(msg: str) -> str:
    return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


def terminal(command: str, timeout: int = 120, work_dir: str | None = None, background: bool = False) -> str:
    """Execute a shell command and return stdout/stderr.

    Args:
        command: The shell command to run.
        timeout: Maximum seconds to wait (default 120). Ignored for background commands.
        work_dir: Working directory for the command.
        background: If True, start the command in the background and return
                    a process ID for later polling.

    Returns:
        JSON string with:
          - Foreground: success, stdout, stderr, exit_code, duration_sec
          - Background: success, pid, process_id (for polling)
    """
    global _bg_counter

    try:
        if background:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=work_dir,
            )
            _bg_counter += 1
            proc_id = f"bg_{_bg_counter}"
            _bg_processes[proc_id] = proc

            logger.info("Background command started: pid=%d, id=%s", proc.pid, proc_id)
            return _success({
                "pid": proc.pid,
                "process_id": proc_id,
                "message": f"Command running in background with process_id={proc_id}",
            })

        # Foreground execution
        start = time.monotonic()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
        )
        duration = time.monotonic() - start

        stdout = result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout
        stderr = result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr

        logger.debug("Command completed: exit_code=%d, duration=%.2fs", result.returncode, duration)

        return _success({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "duration_sec": round(duration, 2),
        })

    except subprocess.TimeoutExpired:
        return _error(f"Command timed out after {timeout}s: {command}")
    except Exception as e:
        return _error(f"Failed to execute command: {e}")


def terminal_poll(process_id: str) -> str:
    """Poll a background process for its status.

    Args:
        process_id: The ID returned by a background terminal() call.

    Returns:
        JSON with process status, and if finished, stdout/stderr/exit_code.
    """
    proc = _bg_processes.get(process_id)
    if proc is None:
        return _error(f"Unknown process_id: {process_id}")

    retcode = proc.poll()
    if retcode is None:
        return _success({
            "status": "running",
            "pid": proc.pid,
            "process_id": process_id,
        })

    # Process finished — retrieve output
    stdout, stderr = proc.communicate()
    stdout = stdout.decode("utf-8", errors="replace") if stdout else ""
    stderr = stderr.decode("utf-8", errors="replace") if stderr else ""

    # Trim to reasonable size
    stdout = stdout[-10000:] if len(stdout) > 10000 else stdout
    stderr = stderr[-5000:] if len(stderr) > 5000 else stderr

    # Clean up
    del _bg_processes[process_id]

    return _success({
        "status": "finished",
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": retcode,
        "process_id": process_id,
    })


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry: ToolRegistry) -> None:
    """Register terminal tools with the given registry."""
    registry.register(
        name="terminal",
        toolset="terminal",
        description=(
            "Execute a shell command and return stdout/stderr. "
            "Use background=true for long-running commands; poll later with terminal_poll."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout": {"type": "integer", "description": "Max seconds to wait (default 120).", "default": 120},
                "work_dir": {"type": "string", "description": "Working directory. Optional."},
                "background": {"type": "boolean", "description": "Run in background.", "default": False},
            },
            "required": ["command"],
        },
        handler=terminal,
    )

    registry.register(
        name="terminal_poll",
        toolset="terminal",
        description="Poll a background process started with terminal(background=true) for its status and output.",
        parameters={
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "The process_id returned by the background terminal call."},
            },
            "required": ["process_id"],
        },
        handler=terminal_poll,
    )
