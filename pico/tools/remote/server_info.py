"""Remote server status monitoring tool.

Runs nvidia-smi, free, df, and ps on the remote host to collect GPU, CPU,
memory, and disk information.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _parse_gpu_info(smi_output: str) -> list[dict[str, Any]]:
    """Parse nvidia-smi CSV output into a list of GPU dicts."""
    gpus: list[dict[str, Any]] = []
    if not smi_output.strip():
        return gpus

    for line in smi_output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1].strip(),
                    "memory_used": parts[2].strip(),
                    "memory_total": parts[3].strip(),
                    "utilization": parts[4].strip(),
                })
            except (ValueError, IndexError):
                continue
    return gpus


def _parse_memory_info(free_output: str) -> dict[str, str]:
    """Parse `free -h` output."""
    mem: dict[str, str] = {}
    for line in free_output.strip().splitlines():
        parts = line.split()
        if parts and parts[0].startswith("Mem:"):
            labels = ["total", "used", "free", "shared", "buff/cache", "available"]
            for i, label in enumerate(labels):
                if i + 1 < len(parts):
                    mem[label] = parts[i + 1]
    return mem


def _parse_disk_info(df_output: str) -> list[dict[str, str]]:
    """Parse `df -h` output."""
    disks: list[dict[str, str]] = []
    lines = df_output.strip().splitlines()
    if not lines:
        return disks
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 6:
            disks.append({
                "filesystem": parts[0],
                "size": parts[1],
                "used": parts[2],
                "available": parts[3],
                "use_percent": parts[4],
                "mount": parts[5],
            })
    return disks


def _parse_processes(ps_output: str) -> list[dict[str, str]]:
    """Parse `ps aux --sort=-%mem` top-20 lines."""
    procs: list[dict[str, str]] = []
    lines = ps_output.strip().splitlines()
    if not lines:
        return procs
    headers = lines[0].split()
    for line in lines[1:]:
        parts = line.split(None, len(headers) - 1)
        if len(parts) == len(headers):
            procs.append(dict(zip(headers, parts)))
    return procs


def server_status_handler(args: dict[str, Any]) -> str:
    """Get remote server status.

    Returns JSON with ``gpus``, ``memory``, ``disk``, and ``running_processes``.
    """
    from pico.tools.remote.ssh_client import get_ssh_client

    server: str = args.get("server", "default")

    try:
        client = get_ssh_client(server)

        result: dict[str, Any] = {
            "success": True,
            "server": server,
            "host": client.host,
        }

        # GPU info (non-fatal if nvidia-smi missing)
        gpu_res = client.execute("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null", timeout=15)
        if gpu_res["exit_code"] == 0:
            result["gpus"] = _parse_gpu_info(gpu_res["stdout"])
        else:
            result["gpus"] = []
            result["gpu_error"] = "nvidia-smi not available"

        # Memory
        mem_res = client.execute("free -h", timeout=10)
        result["memory"] = _parse_memory_info(mem_res["stdout"]) if mem_res["exit_code"] == 0 else {}

        # Disk
        disk_res = client.execute("df -h --total 2>/dev/null || df -h", timeout=10)
        result["disk"] = _parse_disk_info(disk_res["stdout"]) if disk_res["exit_code"] == 0 else []

        # Top processes by memory
        ps_res = client.execute("ps aux --sort=-%mem | head -20", timeout=10)
        result["running_processes"] = _parse_processes(ps_res["stdout"]) if ps_res["exit_code"] == 0 else []

        # Load average
        load_res = client.execute("uptime", timeout=10)
        if load_res["exit_code"] == 0:
            result["uptime"] = load_res["stdout"].strip()

        return json.dumps(result, ensure_ascii=False)
    except (ConnectionError, OSError) as e:
        logger.warning("server_status SSH connection failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.exception("server_status failed")
        return json.dumps({"success": False, "error": str(e)})
