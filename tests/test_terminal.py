"""Tests for pico.tools.terminal — shell command execution."""

from __future__ import annotations

import json
import time

import pytest

from pico.tools.terminal import terminal, terminal_poll


class TestEchoCommand:
    """Test basic echo command execution."""

    def test_echo_stdout(self) -> None:
        result_json = terminal("echo hello")
        result = json.loads(result_json)
        assert result["success"] is True
        assert "hello" in result["stdout"]
        assert result["exit_code"] == 0

    def test_echo_stderr_empty(self) -> None:
        result_json = terminal("echo hello")
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["stderr"] == ""

    def test_duration_is_positive(self) -> None:
        result_json = terminal("echo test")
        result = json.loads(result_json)
        assert result["duration_sec"] >= 0


class TestNonZeroExitCode:
    """Test that non-zero exit codes are captured."""

    def test_exit_code_1(self) -> None:
        result_json = terminal("exit 1")
        result = json.loads(result_json)
        assert result["success"] is True  # The tool call itself succeeded
        assert result["exit_code"] == 1

    def test_exit_code_2(self) -> None:
        result_json = terminal("exit 2")
        result = json.loads(result_json)
        assert result["exit_code"] == 2

    def test_stderr_capture(self) -> None:
        result_json = terminal("echo error >&2; exit 1")
        result = json.loads(result_json)
        assert result["exit_code"] == 1
        assert "error" in result["stderr"]


class TestTimeoutBehavior:
    """Test command timeout handling."""

    def test_timeout_triggers_error(self) -> None:
        """A command that exceeds the timeout should return an error."""
        result_json = terminal("sleep 10", timeout=1)
        result = json.loads(result_json)
        assert result["success"] is False
        assert "timed out" in result.get("error", "").lower()

    def test_fast_command_within_timeout(self) -> None:
        """A fast command should complete normally within timeout."""
        result_json = terminal("echo fast", timeout=5)
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["exit_code"] == 0


class TestWorkingDirectory:
    """Test work_dir parameter."""

    def test_work_dir_changes_cwd(self) -> None:
        result_json = terminal("pwd", work_dir="/tmp")
        result = json.loads(result_json)
        assert result["success"] is True
        assert "/tmp" in result["stdout"].strip()

    def test_invalid_work_dir(self) -> None:
        result_json = terminal("pwd", work_dir="/nonexistent_directory_xyz")
        result = json.loads(result_json)
        # Should fail because directory doesn't exist
        assert result["success"] is False


class TestBackgroundExecution:
    """Test background mode and polling."""

    def test_background_starts_process(self) -> None:
        result_json = terminal("sleep 0.5 && echo done", background=True)
        result = json.loads(result_json)
        assert result["success"] is True
        assert "process_id" in result
        assert "pid" in result

    def test_background_poll_completion(self) -> None:
        result_json = terminal("echo bg_done", background=True)
        result = json.loads(result_json)
        proc_id = result["process_id"]

        # Wait a bit for the command to finish
        time.sleep(0.2)

        poll_json = terminal_poll(proc_id)
        poll = json.loads(poll_json)
        assert poll["success"] is True
        assert poll["status"] == "finished"
        assert "bg_done" in poll["stdout"]
        assert poll["exit_code"] == 0

    def test_poll_unknown_process_id(self) -> None:
        poll_json = terminal_poll("bg_99999")
        poll = json.loads(poll_json)
        assert poll["success"] is False
        assert "unknown" in poll.get("error", "").lower()


class TestOutputTruncation:
    """Test that large output is truncated."""

    def test_stdout_truncation(self) -> None:
        """stdout > 10000 chars should be truncated."""
        # Generate ~15000 chars of output
        result_json = terminal("python3 -c \"print('x' * 15000)\"")
        result = json.loads(result_json)
        assert result["success"] is True
        assert len(result["stdout"]) <= 10001  # 10000 + possible newline
