"""Training Monitor — web dashboard for tracking YOLO/MMDet training runs.

Scans filesystem for training output directories, parses results.csv / log.json,
serves interactive metrics charts, and optionally proxies TensorBoard.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).parent / "training.html"

# Default scan roots — override via CLI or API
_DEFAULT_SCAN_ROOTS = [
    str(Path.home() / "runs"),
    str(Path.home() / "projects"),
    "/tmp",
]

# In-memory cache of discovered runs
_run_cache: list[dict[str, Any]] = []
_cache_ts: float = 0
_CACHE_TTL = 10  # seconds

# Run metadata (custom display name + notes)
_META_PATH = Path.home() / ".pico-agent" / "monitor_meta.json"


def _load_meta() -> dict[str, dict[str, str]]:
    """Load run metadata from disk. Returns {run_path: {name, notes}}."""
    try:
        if _META_PATH.exists():
            return json.loads(_META_PATH.read_text())
    except json.JSONDecodeError as e:
        logger.warning("Corrupted monitor_meta.json: %s — starting fresh", e)
    except OSError as e:
        logger.warning("Cannot read monitor_meta.json: %s", e)
    return {}


def _save_meta(meta: dict[str, dict[str, str]]) -> None:
    """Atomic write: write to temp file, then rename."""
    _META_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(_META_PATH.parent), suffix=".tmp", prefix="meta_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(_META_PATH))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_meta_cache: dict[str, dict[str, str]] = {}


# ======================================================================
# Result image names (module-level constant)
# ======================================================================

_RESULT_IMAGE_NAMES = (
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
    "results.png", "labels.jpg",
)


# ======================================================================
# YOLO run scanner
# ======================================================================

def _parse_yolo_results_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse YOLO results.csv into per-epoch dicts."""
    rows: list[dict[str, Any]] = []
    for line in csv_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # YOLO results.csv uses space-separated values with leading spaces
        parts = line.split()
        if not rows:
            # First line is header
            header = [h.strip() for h in parts]
            continue
        row: dict[str, Any] = {}
        for i, val in enumerate(parts):
            if i < len(header):
                try:
                    row[header[i]] = float(val)
                except ValueError:
                    row[header[i]] = val
        rows.append(row)
    return rows


def _parse_yolo_results_csv_v2(csv_text: str) -> list[dict[str, Any]]:
    """Parse YOLO results.csv — handles both comma-separated and space-separated."""
    lines = csv_text.strip().splitlines()
    if not lines:
        return []

    # Detect format: if first line has commas, use csv.DictReader
    if "," in lines[0]:
        rows: list[dict[str, Any]] = []
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            cleaned: dict[str, Any] = {}
            for k, v in row.items():
                key = k.strip()
                try:
                    cleaned[key] = float(v.strip())
                except (ValueError, AttributeError):
                    cleaned[key] = v.strip() if isinstance(v, str) else v
            rows.append(cleaned)
        return rows
    else:
        return _parse_yolo_results_csv(csv_text)


# Process scan cache (avoids repeated /proc walks)
_proc_cache: dict[str, set[str]] = {}  # {"cmdline_matches": set(), "cwd_matches": set()}
_proc_cache_ts: float = 0
_PROC_CACHE_TTL = 5  # seconds


def _scan_processes_for_run(run_dir: Path) -> bool:
    """Check if any running process references this run directory.

    Scans /proc once and caches results for _PROC_CACHE_TTL seconds.
    Returns True if a matching process is found.
    """
    global _proc_cache, _proc_cache_ts
    now = time.time()

    if now - _proc_cache_ts > _PROC_CACHE_TTL:
        # Rebuild cache: scan /proc for all Python processes
        cmdline_paths: set[str] = set()
        cwd_python: set[str] = set()
        try:
            for pid_dir in Path("/proc").iterdir():
                if not pid_dir.name.isdigit():
                    continue
                try:
                    cmdline_bytes = (pid_dir / "cmdline").read_bytes()
                    cmdline = cmdline_bytes.decode("utf-8", errors="replace").replace("\0", " ")
                    # Record all paths mentioned in cmdline
                    cmdline_paths.add(cmdline)
                    # For Python processes, also record their CWD
                    if "python" in cmdline.lower():
                        cwd = (pid_dir / "cwd").resolve()
                        cwd_python.add(str(cwd))
                except (OSError, PermissionError, FileNotFoundError):
                    pass
        except (OSError, PermissionError):
            pass

        _proc_cache = {"cmdline_raw": cmdline_paths, "cwd_python": cwd_python}
        _proc_cache_ts = now

    run_resolved = str(run_dir.resolve())
    # Check if any cmdline mentions this run dir
    for cmdline in _proc_cache.get("cmdline_raw", set()):
        if run_resolved in cmdline:
            return True
    # Check if any Python process CWD is exactly this run dir (not parent)
    for cwd in _proc_cache.get("cwd_python", set()):
        if cwd == run_resolved:
            return True
    return False


def _scan_for_runs(scan_roots: list[str]) -> list[dict[str, Any]]:
    """Walk scan roots looking for YOLO training run directories.

    A "run dir" is identified by containing:
      - results.csv (training metrics)
      - args.yaml (training config)
    Optionally contains: weights/, events.out.tfevents*, confusion_matrix.png
    """
    global _run_cache, _cache_ts
    now = time.time()
    if _run_cache and (now - _cache_ts) < _CACHE_TTL:
        return _run_cache

    runs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in scan_roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        try:
            for results_csv in root_path.rglob("results.csv"):
                run_dir = results_csv.parent
                run_key = str(run_dir.resolve())
                if run_key in seen:
                    continue
                seen.add(run_key)

                run_info = _inspect_run_dir(run_dir)
                if run_info:
                    runs.append(run_info)
        except PermissionError:
            continue

    # Sort by modification time (newest first)
    runs.sort(key=lambda r: r.get("modified", ""), reverse=True)
    _run_cache = runs
    _cache_ts = now
    return runs


def _inspect_run_dir(run_dir: Path) -> dict[str, Any] | None:
    """Inspect a single directory for training artifacts."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    # Single stat() call
    csv_stat = results_csv.stat()

    info: dict[str, Any] = {
        "id": str(run_dir.resolve()),
        "name": run_dir.name,
        "path": str(run_dir.resolve()),
        "parent": str(run_dir.parent.resolve()),
        "modified": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(csv_stat.st_mtime),
        ),
        "modified_ts": csv_stat.st_mtime,
    }

    # Parse results.csv
    try:
        text = results_csv.read_text()
        rows = _parse_yolo_results_csv_v2(text)
        info["epochs_logged"] = len(rows)
        if rows:
            last = rows[-1]
            info["last_epoch"] = last
            # Extract key metrics
            info["metrics"] = {}
            for key in ("epoch", "metrics/precision(B)", "metrics/recall(B)",
                        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                        "train/box_loss", "train/cls_loss", "train/dfl_loss",
                        "val/box_loss", "val/cls_loss", "val/dfl_loss",
                        "lr/pg0", "lr/pg1", "lr/pg2"):
                if key in last:
                    info["metrics"][key] = round(last[key], 6) if isinstance(last[key], float) else last[key]
            # Best mAP
            map_key = "metrics/mAP50-95(B)"
            if map_key in rows[0]:
                best_val = max((r.get(map_key, -float("inf")) for r in rows), default=0)
                best_idx = max(range(len(rows)), key=lambda i: rows[i].get(map_key, 0))
                info["best_mAP50-95"] = round(best_val, 6) if best_val > -float("inf") else 0
                info["best_epoch"] = best_idx + 1
            map50_key = "metrics/mAP50(B)"
            if map50_key in rows[0]:
                best_50 = max((r.get(map50_key, -float("inf")) for r in rows), default=0)
                info["best_mAP50"] = round(best_50, 6) if best_50 > -float("inf") else 0
        info["results_data"] = rows
    except Exception as e:
        info["parse_error"] = str(e)
        info["epochs_logged"] = 0

    # Check for args.yaml
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        info["has_args"] = True
        try:
            import yaml
            with open(args_yaml) as f:
                info["args"] = yaml.safe_load(f)
        except Exception:
            info["args_raw"] = args_yaml.read_text()[:2000]
    else:
        info["has_args"] = False

    # Check for weights
    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        weights = []
        for f in sorted(weights_dir.iterdir()):
            if f.suffix == ".pt":
                pt_stat = f.stat()
                weights.append({
                    "name": f.name,
                    "size_mb": round(pt_stat.st_size / 1024 / 1024, 2),
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(pt_stat.st_mtime),
                    ),
                })
        info["weights"] = weights

    # Check for TensorBoard events
    events = list(run_dir.glob("events.out.tfevents.*"))
    info["has_tensorboard"] = len(events) > 0
    info["tensorboard_logdir"] = str(run_dir.resolve()) if events else ""

    # Check for result images
    images = []
    for name in _RESULT_IMAGE_NAMES:
        if (run_dir / name).exists():
            images.append(name)
    for f in sorted(run_dir.iterdir()):
        if f.name not in images and (
            f.name.startswith("train_batch") or f.name.startswith("val_batch")
        ) and f.suffix in (".jpg", ".png"):
            images.append(f.name)
    info["result_images"] = images

    # Determine status — multi-signal active probe
    info["status"] = _detect_run_status(run_dir, info.get("epochs_logged", 0))

    return info


def _detect_run_status(run_dir: Path, epochs_logged: int) -> str:
    """Detect whether a training run is still active.

    Signals (any one triggers "running"):
    1. results.csv or last.pt modified within last 5 minutes
    2. Any running process references this run directory (cmdline or CWD)
    3. A .nfs* lock file exists (NFS mounts)
    """
    _ACTIVE_WINDOW = 300  # 5 minutes
    now = time.time()

    # Signal 1: file mtime — results.csv or weight files recently updated
    for check_file in [run_dir / "results.csv", run_dir / "weights" / "last.pt"]:
        if check_file.exists() and now - check_file.stat().st_mtime < _ACTIVE_WINDOW:
            return "running"

    # Signal 2: process probe (cached /proc scan)
    if _scan_processes_for_run(run_dir):
        return "running"

    # Signal 3: NFS lock files (use glob to avoid full iterdir)
    if list(run_dir.glob(".nfs*")):
        return "running"

    if epochs_logged > 0:
        return "completed"
    return "empty"


# ======================================================================
# Path security helper
# ======================================================================

def _is_within_roots(path: Path, scan_roots: list[str]) -> bool:
    """Check that a resolved path is under one of the configured scan roots."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in scan_roots:
        try:
            root_resolved = Path(root).expanduser().resolve()
            if resolved == root_resolved or str(resolved).startswith(str(root_resolved) + "/"):
                return True
        except OSError:
            continue
    return False


# ======================================================================
# HTTP Handler
# ======================================================================

class _MonitorHandler(BaseHTTPRequestHandler):
    """HTTP handler for training monitor dashboard."""

    scan_roots: list[str] = []

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("Monitor HTTP %s", fmt % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/":
            self._serve_html()
        elif path == "/api/runs":
            self._api_list_runs()
        elif re.match(r"^/api/runs/(.+)$", path):
            remainder = "/".join(path.split("/")[3:])
            # Check if this is an image request: <run_id>/image/<filename>
            parts = remainder.split("/image/")
            if len(parts) == 2:
                self._api_serve_image(parts[0], parts[1])
            else:
                self._api_get_run(remainder)
        elif path == "/api/scan":
            self._api_rescan()
        elif path == "/api/tensorboard/status":
            self._api_tensorboard_status()
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/tensorboard/start":
            self._api_tensorboard_start()
        elif path == "/api/tensorboard/stop":
            self._api_tensorboard_stop()
        elif path == "/api/scan/roots":
            self._api_update_roots()
        elif re.match(r"^/api/runs/(.+)/open$", path):
            self._api_open_run_dir("/".join(path.split("/")[3:-1]))
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        m = re.match(r"^/api/runs/(.+)$", path)
        if m:
            self._api_delete_run(m.group(1))
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        m = re.match(r"^/api/runs/(.+)/meta$", path)
        if m:
            self._api_update_meta("/".join(path.split("/")[3:-1]))
        else:
            self._json_response({"error": "Not found"}, 404)

    # ---- API handlers ----

    def _api_list_runs(self) -> None:
        global _meta_cache
        _meta_cache = _load_meta()
        runs = _scan_for_runs(self.scan_roots)
        # Strip large fields for list view
        summary = []
        for r in runs:
            path = r.get("path", "")
            meta = _meta_cache.get(path, {})
            display_name = meta.get("name") or r["name"]
            summary.append({
                "id": r["id"],
                "name": display_name,
                "original_name": r["name"],
                "path": r["path"],
                "modified": r["modified"],
                "status": r.get("status", "unknown"),
                "epochs_logged": r.get("epochs_logged", 0),
                "metrics": r.get("metrics", {}),
                "best_mAP50": r.get("best_mAP50"),
                "best_mAP50-95": r.get("best_mAP50-95"),
                "best_epoch": r.get("best_epoch"),
                "has_tensorboard": r.get("has_tensorboard", False),
                "has_args": r.get("has_args", False),
                "weights_count": len(r.get("weights", [])),
                "result_images": r.get("result_images", []),
                "notes": meta.get("notes", ""),
            })
        self._json_response(summary)

    def _api_get_run(self, run_path: str) -> None:
        # Decode the path (it was URL-encoded in the id)
        decoded = unquote(run_path)
        # Security: validate path is within scan roots
        if not _is_within_roots(Path(decoded), self.scan_roots):
            self._json_response({"error": "Path not within scan roots"}, 403)
            return
        # Find in cache
        runs = _scan_for_runs(self.scan_roots)
        run = None
        for r in runs:
            if r["id"] == decoded or r["path"] == decoded:
                run = r
                break
        if run is None:
            self._json_response({"error": "Run not found"}, 404)
            return
        # Inject metadata (reload fresh)
        global _meta_cache
        _meta_cache = _load_meta()
        meta = _meta_cache.get(run.get("path", ""), {})
        if meta.get("name"):
            run["original_name"] = run["name"]
            run["name"] = meta["name"]
        run["notes"] = meta.get("notes", "")
        self._json_response(run)

    def _api_rescan(self) -> None:
        global _cache_ts
        _cache_ts = 0  # force rescan
        runs = _scan_for_runs(self.scan_roots)
        self._json_response({"success": True, "count": len(runs)})

    def _api_tensorboard_status(self) -> None:
        info = _get_tensorboard_info()
        self._json_response(info)

    def _api_tensorboard_start(self) -> None:
        body = self._read_body()
        logdir = body.get("logdir", "")
        port = body.get("port", 6006)
        if not logdir:
            self._json_response({"error": "logdir is required"}, 400)
            return
        # Security: validate logdir is within scan roots
        if not _is_within_roots(Path(logdir), self.scan_roots):
            self._json_response({"error": "logdir not within scan roots"}, 403)
            return
        result = _start_tensorboard(logdir, int(port))
        self._json_response(result)

    def _api_tensorboard_stop(self) -> None:
        result = _stop_tensorboard()
        self._json_response(result)

    def _api_update_roots(self) -> None:
        body = self._read_body()
        roots = body.get("roots", [])
        if roots:
            self.scan_roots = roots
            global _cache_ts
            _cache_ts = 0
        self._json_response({"success": True, "roots": self.scan_roots})

    def _api_update_meta(self, run_id_encoded: str) -> None:
        """Update display name and/or notes for a run."""
        decoded = unquote(run_id_encoded)
        body = self._read_body()
        if not body:
            self._json_response({"error": "Empty body"}, 400)
            return
        global _meta_cache
        _meta_cache = _load_meta()
        path = decoded
        # Find actual path from runs — must exist in scan results
        runs = _scan_for_runs(self.scan_roots)
        found = False
        for r in runs:
            if r["id"] == decoded or r["path"] == decoded:
                path = r["path"]
                found = True
                break
        if not found:
            self._json_response({"error": "Run not found in scan results"}, 404)
            return
        entry = _meta_cache.get(path, {})
        if "name" in body:
            entry["name"] = str(body["name"])[:200]
        if "notes" in body:
            entry["notes"] = str(body["notes"])[:2000]
        _meta_cache[path] = entry
        _save_meta(_meta_cache)
        self._json_response({"success": True, "meta": entry})

    def _api_delete_run(self, run_id_encoded: str) -> None:
        """Delete a training run directory and remove it from cache."""
        decoded = unquote(run_id_encoded)
        target = Path(decoded)
        # Security: must be within scan roots
        if not _is_within_roots(target, self.scan_roots):
            self._json_response({"error": "Path not within scan roots"}, 403)
            return
        if not target.exists() or not target.is_dir():
            self._json_response({"error": "Run directory not found"}, 404)
            return
        # Must contain results.csv to be a valid run
        if not (target / "results.csv").exists():
            self._json_response({"error": "Not a valid training run directory"}, 400)
            return
        try:
            shutil.rmtree(str(target))
            # Remove from cache
            global _run_cache, _cache_ts, _meta_cache
            _run_cache = [r for r in _run_cache if r.get("id") != decoded and r.get("path") != decoded]
            _cache_ts = 0  # force rescan on next request
            # Remove from metadata
            resolved = str(target.resolve())
            if resolved in _meta_cache:
                del _meta_cache[resolved]
                _save_meta(_meta_cache)
            self._json_response({"success": True, "deleted": decoded})
        except Exception as e:
            logger.error("Failed to delete %s: %s", decoded, e)
            self._json_response({"error": f"Failed to delete: {e}"}, 500)

    def _api_open_run_dir(self, run_id_encoded: str) -> None:
        """Open run directory in system file manager."""
        decoded = unquote(run_id_encoded)
        target = Path(decoded)
        # Security: must be within scan roots
        if not _is_within_roots(target, self.scan_roots):
            self._json_response({"error": "Path not within scan roots"}, 403)
            return
        if not target.exists() or not target.is_dir():
            self._json_response({"error": "Directory not found"}, 404)
            return
        try:
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", str(target)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._json_response({"success": True, "path": str(target)})
            elif shutil.which("open"):  # macOS
                subprocess.Popen(["open", str(target)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._json_response({"success": True, "path": str(target)})
            else:
                self._json_response({"success": False, "error": "No file manager found (xdg-open/open)"})
        except Exception as e:
            self._json_response({"success": False, "error": str(e)})

    def _api_serve_image(self, run_id: str, filename: str) -> None:
        """Serve result images (confusion_matrix.png, etc.) from run directory."""
        decoded_id = unquote(run_id)
        decoded_name = unquote(filename)
        runs = _scan_for_runs(self.scan_roots)
        run = None
        for r in runs:
            if r["id"] == decoded_id or r["path"] == decoded_id:
                run = r
                break
        if run is None:
            self._json_response({"error": "Run not found"}, 404)
            return
        # Security: only allow known result image names
        _STATIC_ALLOWED = {
            "confusion_matrix.png", "confusion_matrix_normalized.png",
            "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
            "results.png", "labels.jpg",
        }
        # Also allow train_batch*.jpg and val_batch*.jpg
        allowed = decoded_name in _STATIC_ALLOWED or (
            (decoded_name.startswith("train_batch") or decoded_name.startswith("val_batch"))
            and decoded_name.endswith((".jpg", ".png"))
        )
        if not allowed:
            self._json_response({"error": "Image not allowed"}, 403)
            return
        img_path = Path(run["path"]) / decoded_name
        # Security: verify resolved path stays within the run directory
        try:
            resolved = img_path.resolve()
            run_resolved = Path(run["path"]).resolve()
            if not str(resolved).startswith(str(run_resolved) + "/"):
                self._json_response({"error": "Path traversal detected"}, 403)
                return
        except OSError:
            self._json_response({"error": "Invalid path"}, 400)
            return
        if not img_path.exists():
            self._json_response({"error": "Image not found"}, 404)
            return
        content = img_path.read_bytes()
        self.send_response(200)
        ct = "image/jpeg" if decoded_name.endswith(".jpg") else "image/png"
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "max-age=10")
        self.end_headers()
        self.wfile.write(content)

    # ---- helpers ----

    def _serve_html(self) -> None:
        if _HTML_PATH.exists():
            content = _HTML_PATH.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content.encode())))
            self.end_headers()
            self.wfile.write(content.encode())
        else:
            self._json_response({"error": "HTML not found"}, 500)

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self, max_bytes: int = 1_048_576) -> dict:
        """Read and parse JSON body. Caps at max_bytes (default 1MB)."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        if length > max_bytes:
            self._json_response({"error": "Request body too large"}, 413)
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


# ======================================================================
# TensorBoard management
# ======================================================================

_tb_process: subprocess.Popen | None = None
_tb_port: int = 6006
_tb_logdir: str = ""

# Cache tensorboard availability (it's not going to install/uninstall mid-run)
_tb_available_cache: bool | None = None


def _tb_is_available() -> bool:
    """Check if tensorboard binary is accessible (cached)."""
    global _tb_available_cache
    if _tb_available_cache is None:
        _tb_available_cache = shutil.which("tensorboard") is not None
    return _tb_available_cache


def _get_tensorboard_info() -> dict[str, Any]:
    global _tb_process, _tb_port, _tb_logdir
    running = _tb_process is not None and _tb_process.poll() is None
    return {
        "available": _tb_is_available(),
        "running": running,
        "port": _tb_port,
        "logdir": _tb_logdir,
        "url": f"http://127.0.0.1:{_tb_port}" if running else "",
    }


def _start_tensorboard(logdir: str, port: int = 6006) -> dict[str, Any]:
    global _tb_process, _tb_port, _tb_logdir

    # Stop existing
    _stop_tensorboard()

    try:
        _tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port), "--bind_all", "--reload_interval", "5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _tb_port = port
        _tb_logdir = logdir
        time.sleep(2)  # wait for startup
        if _tb_process.poll() is not None:
            stderr = _tb_process.stderr.read().decode() if _tb_process.stderr else ""
            return {"success": False, "error": f"TensorBoard exited: {stderr[:500]}"}
        return {
            "success": True,
            "port": port,
            "url": f"http://127.0.0.1:{port}",
            "logdir": logdir,
        }
    except FileNotFoundError:
        return {"success": False, "error": "tensorboard not found. Install with: pip install tensorboard"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _stop_tensorboard() -> dict[str, Any]:
    global _tb_process
    if _tb_process and _tb_process.poll() is None:
        _tb_process.terminate()
        try:
            _tb_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _tb_process.kill()
        _tb_process = None
        return {"success": True, "message": "TensorBoard stopped"}
    return {"success": True, "message": "TensorBoard was not running"}


# ======================================================================
# Public API
# ======================================================================

def start_monitor(
    host: str = "127.0.0.1",
    port: int = 8766,
    scan_roots: list[str] | None = None,
    open_browser: bool = True,
) -> None:
    """Start the training monitor dashboard.

    Args:
        host: Bind address.
        port: Port number (default 8766, different from session dashboard).
        scan_roots: Directories to scan for training runs.
        open_browser: Auto-open browser.
    """
    roots = scan_roots or _DEFAULT_SCAN_ROOTS

    if host != "127.0.0.1":
        logger.warning("Binding to %s — all endpoints are unauthenticated!", host)

    _MonitorHandler.scan_roots = roots
    HTTPServer.allow_reuse_address = True
    try:
        server = HTTPServer((host, port), _MonitorHandler)
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 98:
            print(f"❌ Port {port} is already in use.")
            print(f"   Another instance may be running, or try: pico-agent monitor --port {port + 1}")
            return
        raise

    url = f"http://{host}:{port}"
    print(f"🏋️ Pico Training Monitor — {url}")
    print(f"   Scanning: {', '.join(roots)}")
    print("   Press Ctrl+C to stop.\n")

    if open_browser:
        import webbrowser
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        _stop_tensorboard()
        server.server_close()


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser(description="Pico Training Monitor")
    _p.add_argument("--port", "-p", type=int, default=8766)
    _p.add_argument("--scan", nargs="*", help="Directories to scan")
    _p.add_argument("--host", default="127.0.0.1")
    _p.add_argument("--no-open", action="store_true")
    _a = _p.parse_args()
    start_monitor(
        host=_a.host,
        port=_a.port,
        scan_roots=_a.scan or None,
        open_browser=not _a.no_open,
    )
