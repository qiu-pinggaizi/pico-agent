"""Tests for remote workspace sandbox."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock
from types import ModuleType

import pytest


# ---------------------------------------------------------------------------
# Mock paramiko if not installed
# ---------------------------------------------------------------------------

def _ensure_paramiko_mock():
    """Create a fake paramiko module if not installed."""
    if "paramiko" not in sys.modules:
        mock_paramiko = ModuleType("paramiko")
        mock_paramiko.SSHClient = MagicMock
        mock_paramiko.SFTPClient = MagicMock
        mock_paramiko.AutoAddPolicy = MagicMock
        sys.modules["paramiko"] = mock_paramiko


_ensure_paramiko_mock()

from pico.tools.remote.ssh_client import SSHClient


# ---------------------------------------------------------------------------
# SSHClient.check_workspace_path
# ---------------------------------------------------------------------------


class TestCheckWorkspacePath:
    """Test the path validation in SSHClient."""

    def _make_client(self, workspace=None):
        return SSHClient(host="dummy", workspace=workspace)

    def test_no_workspace_allows_anything(self):
        client = self._make_client(workspace=None)
        # Should not raise
        client.check_workspace_path("/etc/passwd")
        client.check_workspace_path("/ipcdata-tj/data/jinj/")

    def test_path_inside_workspace_allowed(self):
        client = self._make_client(workspace="/ipcdata-tj/data/jinj/agent")
        client.check_workspace_path("/ipcdata-tj/data/jinj/agent")
        client.check_workspace_path("/ipcdata-tj/data/jinj/agent/models")
        client.check_workspace_path("/ipcdata-tj/data/jinj/agent/subdir/file.txt")

    def test_path_outside_workspace_blocked(self):
        client = self._make_client(workspace="/ipcdata-tj/data/jinj/agent")
        with pytest.raises(PermissionError, match="outside workspace"):
            client.check_workspace_path("/ipcdata-tj/data/jinj/")

    def test_sibling_dir_blocked(self):
        client = self._make_client(workspace="/ipcdata-tj/data/jinj/agent")
        with pytest.raises(PermissionError, match="outside workspace"):
            client.check_workspace_path("/ipcdata-tj/data/jinj/face_rec")

    def test_root_blocked(self):
        client = self._make_client(workspace="/ipcdata-tj/data/jinj/agent")
        with pytest.raises(PermissionError, match="outside workspace"):
            client.check_workspace_path("/")

    def test_traversal_blocked(self):
        client = self._make_client(workspace="/ipcdata-tj/data/jinj/agent")
        with pytest.raises(PermissionError, match="outside workspace"):
            client.check_workspace_path("/ipcdata-tj/data/jinj/agent/../../etc/passwd")

    def test_workspace_trailing_slash_stripped(self):
        client = self._make_client(workspace="/ipcdata-tj/data/jinj/agent/")
        assert client.workspace == "/ipcdata-tj/data/jinj/agent"


# ---------------------------------------------------------------------------
# remote_terminal_handler delete safety
# ---------------------------------------------------------------------------


class TestDeleteSafety:
    """Test _check_delete_safety pattern matching."""

    def _check(self, command, workspace):
        from pico.tools.remote.remote_terminal import _check_delete_safety
        return _check_delete_safety(command, workspace)

    def test_rm_root_blocked(self):
        assert self._check("rm -rf /", "/ws") is not None

    def test_rm_slash_etc_blocked(self):
        assert self._check("rm /etc/passwd", "/ws") is not None

    def test_rm_parent_blocked(self):
        assert self._check("rm ../something", "/ws") is not None

    def test_rm_home_blocked(self):
        assert self._check("rm -rf ~/data", "/ws") is not None

    def test_rmdir_root_blocked(self):
        assert self._check("rmdir /tmp", "/ws") is not None

    def test_mv_to_root_blocked(self):
        assert self._check("mv file.txt /", "/ws") is not None

    def test_rm_in_workspace_allowed(self):
        assert self._check("rm models/old.pt", "/ws") is None

    def test_rm_relative_allowed(self):
        assert self._check("rm -rf __pycache__", "/ws") is None

    def test_no_workspace_skips_check(self):
        assert self._check("rm -rf /", None) is None


# ---------------------------------------------------------------------------
# remote_terminal_handler with workspace
# ---------------------------------------------------------------------------


class TestRemoteTerminalWithWorkspace:
    """Test that remote_terminal_handler applies workspace defaults."""

    @patch("pico.tools.remote.ssh_client.get_ssh_client")
    def test_default_cd_workspace(self, mock_get):
        """When no work_dir, commands should cd into workspace."""
        mock_client = MagicMock()
        mock_client.workspace = "/ipcdata-tj/data/jinj/agent"
        mock_client.execute.return_value = {"stdout": "OK", "stderr": "", "exit_code": 0}
        mock_get.return_value = mock_client

        from pico.tools.remote.remote_terminal import remote_terminal_handler
        result = json.loads(remote_terminal_handler(
            command="ls -la", server="default"
        ))

        assert result["success"] is True
        called_cmd = mock_client.execute.call_args[0][0]
        assert called_cmd.startswith("cd /ipcdata-tj/data/jinj/agent &&")

    @patch("pico.tools.remote.ssh_client.get_ssh_client")
    def test_work_dir_outside_workspace_blocked(self, mock_get):
        """work_dir outside workspace should be blocked."""
        mock_client = MagicMock()
        mock_client.workspace = "/ipcdata-tj/data/jinj/agent"
        mock_client.check_workspace_path.side_effect = PermissionError("Access denied")
        mock_get.return_value = mock_client

        from pico.tools.remote.remote_terminal import remote_terminal_handler
        result = json.loads(remote_terminal_handler(
            command="ls", server="default", work_dir="/etc"
        ))

        assert result["success"] is False
        assert "Access denied" in result["error"]

    @patch("pico.tools.remote.ssh_client.get_ssh_client")
    def test_destructive_command_blocked(self, mock_get):
        """rm / should be blocked."""
        mock_client = MagicMock()
        mock_client.workspace = "/ipcdata-tj/data/jinj/agent"
        mock_get.return_value = mock_client

        from pico.tools.remote.remote_terminal import remote_terminal_handler
        result = json.loads(remote_terminal_handler(
            command="rm -rf /", server="default"
        ))

        assert result["success"] is False
        assert "Blocked" in result["error"]
