"""paramiko SSH client wrapper with connection pooling.

Each named server (from ~/.pico-agent/config.yaml ``servers`` section) gets
its own :class:`SSHClient` instance cached in a global pool.  Connections are
created lazily on first use and reused for subsequent calls.
"""

from __future__ import annotations

import logging
import os
import socket
import stat
import time
from pathlib import Path
from typing import Any, Callable, Optional

import paramiko

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

_ssh_clients: dict[str, "SSHClient"] = {}


def get_ssh_client(server_name: str) -> "SSHClient":
    """Return (or create) a pooled SSHClient for *server_name*."""
    if server_name in _ssh_clients:
        client = _ssh_clients[server_name]
        if client.is_connected():
            return client
        # stale — drop and recreate
        _ssh_clients.pop(server_name, None)

    from pico.config import get_config
    cfg = get_config()
    servers = cfg.servers  # Config dataclass property
    default_name = cfg.default_server

    if server_name == "default":
        if default_name and default_name in servers:
            server_name = default_name

    if server_name not in servers:
        raise ValueError(
            f"Server '{server_name}' not found in config. "
            f"Available: {list(servers.keys())}"
        )

    srv_cfg = servers[server_name]
    client = SSHClient(
        host=srv_cfg["host"],
        port=srv_cfg.get("port", 22),
        user=srv_cfg.get("user", os.getenv("USER", "root")),
        key_path=srv_cfg.get("key_path"),
        password=srv_cfg.get("password"),
        conda_env=srv_cfg.get("conda_env"),
    )
    _ssh_clients[server_name] = client
    return client


# ---------------------------------------------------------------------------
# SSHClient
# ---------------------------------------------------------------------------

class SSHClient:
    """Thin wrapper around :class:`paramiko.SSHClient` with SCP helpers."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        user: str = "root",
        key_path: Optional[str] = None,
        password: Optional[str] = None,
        conda_env: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.key_path = key_path
        self.password = password
        self.conda_env = conda_env
        self._client: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    # -- connection management -----------------------------------------------

    def _connect(self) -> None:
        if self._client is not None:
            return
        logger.info("SSH connecting to %s@%s:%s", self.user, self.host, self.port)
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        kwargs: dict[str, Any] = {
            "hostname": self.host,
            "port": self.port,
            "username": self.user,
            "timeout": 15,
        }
        if self.key_path:
            kwargs["key_filename"] = os.path.expanduser(self.key_path)
        if self.password:
            kwargs["password"] = self.password

        self._client.connect(**kwargs)
        logger.info("SSH connected to %s", self.host)

    def is_connected(self) -> bool:
        if self._client is None:
            return False
        transport = self._client.get_transport()
        if transport is None or not transport.is_active():
            self._close()
            return False
        return True

    def _close(self) -> None:
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _get_sftp(self) -> paramiko.SFTPClient:
        self._connect()
        if self._sftp is None:
            self._sftp = self._client.open_sftp()  # type: ignore[union-attr]
        return self._sftp

    # -- command execution ---------------------------------------------------

    def execute(self, command: str, timeout: int = 300) -> dict[str, Any]:
        """Execute *command* and return ``{stdout, stderr, exit_code}``."""
        self._connect()
        # Optionally wrap in conda activate
        if self.conda_env:
            command = f"source activate {self.conda_env} 2>/dev/null || conda activate {self.conda_env}; {command}"

        logger.debug("Remote exec: %s", command)
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)  # type: ignore[union-attr]
        stdout.channel.settimeout(timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        return {"stdout": out, "stderr": err, "exit_code": exit_code}

    def stream_command(
        self,
        command: str,
        callback: Callable[[str], None],
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Execute *command* streaming stdout lines to *callback*."""
        self._connect()
        if self.conda_env:
            command = f"source activate {self.conda_env} 2>/dev/null || conda activate {self.conda_env}; {command}"

        transport = self._client.get_transport()  # type: ignore[union-attr]
        chan = transport.open_session()
        chan.settimeout(timeout)
        chan.exec_command(command)

        out_lines: list[str] = []
        while not chan.exit_status_ready():
            if chan.recv_ready():
                chunk = chan.recv(4096).decode("utf-8", errors="replace")
                for line in chunk.splitlines(keepends=True):
                    out_lines.append(line)
                    callback(line.rstrip("\n"))
            time.sleep(0.05)

        # drain remaining
        while chan.recv_ready():
            chunk = chan.recv(4096).decode("utf-8", errors="replace")
            for line in chunk.splitlines(keepends=True):
                out_lines.append(line)
                callback(line.rstrip("\n"))

        err = chan.recv_stderr(4096).decode("utf-8", errors="replace") if chan.recv_stderr_ready() else ""
        exit_code = chan.recv_exit_status()
        return {"stdout": "".join(out_lines), "stderr": err, "exit_code": exit_code}

    # -- file transfer -------------------------------------------------------

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a single file."""
        self._connect()
        sftp = self._get_sftp()
        # ensure remote directory exists
        remote_dir = str(Path(remote_path).parent)
        self._mkdir_p(sftp, remote_dir)
        logger.info("Upload %s -> %s:%s", local_path, self.host, remote_path)
        sftp.put(local_path, remote_path)

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a single file."""
        self._connect()
        sftp = self._get_sftp()
        local_dir = str(Path(local_path).parent)
        os.makedirs(local_dir, exist_ok=True)
        logger.info("Download %s:%s -> %s", self.host, remote_path, local_path)
        sftp.get(remote_path, local_path)

    def upload_dir(self, local_dir: str, remote_dir: str, exclude: list[str] | None = None) -> int:
        """Recursively upload *local_dir* to *remote_dir*. Returns file count."""
        import fnmatch

        exclude = exclude or []
        sftp = self._get_sftp()
        count = 0
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        for root, dirs, files in os.walk(local_dir):
            # filter dirs
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in exclude)]

            rel_root = os.path.relpath(root, local_dir)
            remote_root = os.path.join(remote_dir, rel_root) if rel_root != "." else remote_dir

            self._mkdir_p(sftp, remote_root)
            for fname in files:
                if any(fnmatch.fnmatch(fname, p) for p in exclude):
                    continue
                local_file = os.path.join(root, fname)
                remote_file = os.path.join(remote_root, fname)
                sftp.put(local_file, remote_file)
                count += 1
        return count

    def download_dir(self, remote_dir: str, local_dir: str, exclude: list[str] | None = None) -> int:
        """Recursively download *remote_dir* to *local_dir*. Returns file count."""
        import fnmatch

        exclude = exclude or []
        sftp = self._get_sftp()
        count = 0
        self._download_dir_recursive(sftp, remote_dir, local_dir, exclude, count_ref := [0])
        return count_ref[0]

    def _download_dir_recursive(
        self,
        sftp: paramiko.SFTPClient,
        remote_dir: str,
        local_dir: str,
        exclude: list[str],
        count_ref: list[int],
    ) -> None:
        os.makedirs(local_dir, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            name = entry.filename
            if any(fnmatch.fnmatch(name, p) for p in exclude):
                continue
            remote_path = f"{remote_dir}/{name}"
            local_path = os.path.join(local_dir, name)
            if stat.S_ISDIR(entry.st_mode):
                self._download_dir_recursive(sftp, remote_path, local_path, exclude, count_ref)
            else:
                sftp.get(remote_path, local_path)
                count_ref[0] += 1

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
        """Recursively create remote directories."""
        parts = remote_dir.strip("/").split("/")
        current = ""
        for part in parts:
            current = f"{current}/{part}"
            try:
                sftp.stat(current)
            except FileNotFoundError:
                try:
                    sftp.mkdir(current)
                except OSError:
                    pass  # may already exist due to race
