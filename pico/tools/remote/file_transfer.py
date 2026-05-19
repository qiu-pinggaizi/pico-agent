"""Remote file transfer tools: upload, download, and directory sync."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def file_upload_handler(args: dict[str, Any]) -> str:
    """Upload a local file to a remote server."""
    from pico.tools.remote.ssh_client import get_ssh_client

    local_path: str = args.get("local_path", "")
    remote_path: str = args.get("remote_path", "")
    server: str = args.get("server", "default")

    if not local_path or not remote_path:
        return json.dumps({"success": False, "error": "local_path and remote_path are required"})

    if not os.path.isfile(local_path):
        return json.dumps({"success": False, "error": f"Local file not found: {local_path}"})

    try:
        client = get_ssh_client(server)
        client.upload(local_path, remote_path)
        return json.dumps({
            "success": True,
            "message": f"Uploaded {local_path} -> {remote_path}",
            "size": os.path.getsize(local_path),
        })
    except PermissionError as e:
        return json.dumps({"success": False, "error": str(e)})
    except ConnectionError as e:
        logger.warning("file_upload SSH failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.exception("file_upload failed")
        return json.dumps({"success": False, "error": str(e)})


def file_download_handler(args: dict[str, Any]) -> str:
    """Download a file from a remote server to local."""
    from pico.tools.remote.ssh_client import get_ssh_client

    remote_path: str = args.get("remote_path", "")
    local_path: str = args.get("local_path", "")
    server: str = args.get("server", "default")

    if not remote_path or not local_path:
        return json.dumps({"success": False, "error": "remote_path and local_path are required"})

    try:
        client = get_ssh_client(server)
        client.download(remote_path, local_path)
        return json.dumps({
            "success": True,
            "message": f"Downloaded {remote_path} -> {local_path}",
            "size": os.path.getsize(local_path) if os.path.isfile(local_path) else 0,
        })
    except PermissionError as e:
        return json.dumps({"success": False, "error": str(e)})
    except ConnectionError as e:
        logger.warning("file_download SSH failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.exception("file_download failed")
        return json.dumps({"success": False, "error": str(e)})


def file_sync_handler(args: dict[str, Any]) -> str:
    """Synchronize a directory between local and remote."""
    from pico.tools.remote.ssh_client import get_ssh_client

    local_dir: str = args.get("local_dir", "")
    remote_dir: str = args.get("remote_dir", "")
    direction: str = args.get("direction", "upload")
    server: str = args.get("server", "default")
    exclude: list[str] = args.get("exclude", ["*.pyc", "__pycache__", ".git", ".venv"])

    if not local_dir or not remote_dir:
        return json.dumps({"success": False, "error": "local_dir and remote_dir are required"})
    if direction not in ("upload", "download"):
        return json.dumps({"success": False, "error": f"Invalid direction: {direction}. Use 'upload' or 'download'"})

    try:
        client = get_ssh_client(server)

        if direction == "upload":
            if not os.path.isdir(local_dir):
                return json.dumps({"success": False, "error": f"Local directory not found: {local_dir}"})
            count = client.upload_dir(local_dir, remote_dir, exclude=exclude)
        else:
            os.makedirs(local_dir, exist_ok=True)
            count = client.download_dir(remote_dir, local_dir, exclude=exclude)

        return json.dumps({
            "success": True,
            "message": f"Synced {count} files ({direction}) between {local_dir} and {remote_dir}",
            "files_transferred": count,
            "direction": direction,
        })
    except PermissionError as e:
        return json.dumps({"success": False, "error": str(e)})
    except ConnectionError as e:
        logger.warning("file_sync SSH failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.exception("file_sync failed")
        return json.dumps({"success": False, "error": str(e)})
