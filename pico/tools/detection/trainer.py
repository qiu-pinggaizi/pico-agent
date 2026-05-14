"""Training tools for object detection (YOLO / MMDetection).

Provides handlers for starting, monitoring, analysing, and resuming training.
All tools support local and remote execution via the ``server`` parameter.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _exec_cmd(command: str, server: str = "", timeout: int = 1800, work_dir: str = "") -> dict[str, Any]:
    """Execute *command* locally or on *server*.

    Returns a dict with at least ``success`` (bool) and ``output`` (str).
    """
    if server:
        from pico.tools.remote.remote_terminal import remote_terminal_handler
        args: dict[str, Any] = {"command": command, "server": server, "timeout": timeout}
        if work_dir:
            args["work_dir"] = work_dir
        return json.loads(remote_terminal_handler(args))

    import subprocess
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=work_dir or None,
        )
        return {
            "success": proc.returncode == 0,
            "output": proc.stdout + proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _remote_cat(file_path: str, server: str) -> str | None:
    """Read a remote file's content via ``cat``.  Returns *None* on failure."""
    result = _exec_cmd(f"cat {file_path}", server=server)
    if result.get("success"):
        return result.get("output", "")
    return None


def _read_local(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _read_file_content(path: str, server: str = "") -> str | None:
    """Unified file reader — local or remote."""
    if server:
        return _remote_cat(path, server)
    return _read_local(path)


# ---------------------------------------------------------------------------
# YOLO training log parsers
# ---------------------------------------------------------------------------

def _parse_yolo_results_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse YOLO ``results.csv`` into a list of per-epoch dicts."""
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(csv_text.strip().splitlines())
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


def _find_yolo_log(project_dir: str, server: str = "") -> str | None:
    """Return the path to the latest ``results.csv`` under *project_dir*."""
    if server:
        result = _exec_cmd(
            f"find {project_dir} -name 'results.csv' -type f | sort | tail -1",
            server=server,
        )
        if result.get("success") and result.get("output", "").strip():
            return result["output"].strip().splitlines()[-1].strip()
        return None

    candidates = sorted(Path(project_dir).rglob("results.csv"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


# ---------------------------------------------------------------------------
# MMDet training log parsers
# ---------------------------------------------------------------------------

def _parse_mmdet_log_json(text: str) -> list[dict[str, Any]]:
    """Parse MMDet ``log.json`` (one JSON object per line)."""
    records: list[dict[str, Any]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _find_mmdet_log(project_dir: str, server: str = "") -> str | None:
    """Return the path to the latest ``log.json`` under *project_dir*."""
    if server:
        result = _exec_cmd(
            f"find {project_dir} -name 'log.json' -type f | sort | tail -1",
            server=server,
        )
        if result.get("success") and result.get("output", "").strip():
            return result["output"].strip().splitlines()[-1].strip()
        return None

    candidates = sorted(Path(project_dir).rglob("log.json"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def _detect_framework(project_dir: str, server: str = "") -> str:
    """Heuristically detect whether *project_dir* contains YOLO or MMDet outputs."""
    yolo_log = _find_yolo_log(project_dir, server)
    if yolo_log:
        return "yolo"
    mmdet_log = _find_mmdet_log(project_dir, server)
    if mmdet_log:
        return "mmdet"
    return "unknown"


# ---------------------------------------------------------------------------
# handler: train_start
# ---------------------------------------------------------------------------

def train_start_handler(args: dict[str, Any]) -> str:
    """Start a training job (YOLO or MMDet).

    Parameters
    ----------
    args : dict
        ``config_path`` (str), ``framework`` (str, default 'yolo'),
        ``server`` (str, default ''), ``gpu`` / ``device`` (str, default '0'),
        ``epochs`` (int, optional), ``batch_size`` (int, optional),
        ``work_dir`` (str, optional), ``background`` (bool, default True).
    """
    config_path: str = args.get("config_path", "")
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")
    device: str = args.get("device", "") or args.get("gpu", "0")
    epochs: int | None = args.get("epochs")
    batch_size: int | None = args.get("batch_size")
    work_dir: str = args.get("work_dir", "")
    background: bool = args.get("background", True)

    if not config_path:
        return json.dumps({"success": False, "error": "config_path is required"})

    try:
        if framework == "yolo":
            cmd_parts = ["yolo", "train", f"data={config_path}"]
            # If config_path looks like a .yaml data file it needs a model too.
            # If it is a .pt model file the caller should pass data separately.
            model_from_args = args.get("model", "")
            if model_from_args:
                cmd_parts.append(f"model={model_from_args}")
            if epochs:
                cmd_parts.append(f"epochs={epochs}")
            if batch_size:
                cmd_parts.append(f"batch={batch_size}")
            if device:
                cmd_parts.append(f"device={device}")
            cmd = " ".join(cmd_parts)
        elif framework == "mmdet":
            nproc = len(device.split(",")) if device else 1
            cmd = (
                f"python -m torch.distributed.launch --nproc_per_node={nproc} "
                f"--master_port=29500 "
                f"tools/dist_train.sh {config_path} {nproc}"
            )
            if work_dir:
                cmd += f" --work-dir {work_dir}"
        else:
            return json.dumps({"success": False, "error": f"Unknown framework: {framework}"})

        timeout = 86400 if background else 1800
        result = _exec_cmd(cmd, server=server, timeout=timeout, work_dir=work_dir or "")

        return json.dumps({
            "success": True,
            "command": cmd,
            "framework": framework,
            "background": background,
            "remote": bool(server),
            "server": server,
            "result": result,
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception("train_start failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: train_monitor
# ---------------------------------------------------------------------------

def train_monitor_handler(args: dict[str, Any]) -> str:
    """Parse training logs and return current progress.

    Parameters
    ----------
    args : dict
        ``log_path`` (str) **or** ``project_dir`` (str), ``framework`` (str),
        ``server`` (str).
    """
    log_path: str = args.get("log_path", "")
    project_dir: str = args.get("project_dir", "") or args.get("log_path", "")
    framework: str = args.get("framework", "")
    server: str = args.get("server", "")

    try:
        # Auto-detect framework if not given
        if not framework:
            framework = _detect_framework(project_dir, server)
            if framework == "unknown":
                return json.dumps({"success": False, "error": "Cannot detect framework. Please specify framework='yolo' or 'mmdet'."})

        if framework == "yolo":
            # Resolve log file
            csv_path = log_path
            if not csv_path or Path(csv_path).is_dir():
                csv_path = _find_yolo_log(log_path or project_dir, server) or ""
            if not csv_path:
                return json.dumps({"success": False, "error": "Cannot find results.csv. Provide log_path or project_dir."})

            text = _read_file_content(csv_path, server)
            if text is None:
                return json.dumps({"success": False, "error": f"Cannot read {csv_path}"})

            rows = _parse_yolo_results_csv(text)
            if not rows:
                return json.dumps({"success": True, "message": "Log file is empty — training may not have started.", "epochs": 0})

            last = rows[-1]
            total_epochs = int(last.get("epoch", len(rows))) + 1
            current_epoch = int(last.get("epoch", len(rows) - 1)) + 1

            # Extract common metrics
            metrics: dict[str, Any] = {"current_epoch": current_epoch, "total_epochs_seen": len(rows)}
            for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                        "train/box_loss", "train/cls_loss", "train/dfl_loss",
                        "val/box_loss", "val/cls_loss", "val/dfl_loss"):
                if key in last:
                    metrics[key] = round(last[key], 6)

            return json.dumps({
                "success": True,
                "framework": "yolo",
                "log_path": csv_path,
                "progress": f"{current_epoch} epochs logged",
                "metrics": metrics,
                "latest_row": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in last.items()},
            }, ensure_ascii=False)

        elif framework == "mmdet":
            json_path = log_path
            if not json_path or Path(json_path).is_dir():
                json_path = _find_mmdet_log(log_path or project_dir, server) or ""
            if not json_path:
                return json.dumps({"success": False, "error": "Cannot find log.json. Provide log_path or project_dir."})

            text = _read_file_content(json_path, server)
            if text is None:
                return json.dumps({"success": False, "error": f"Cannot read {json_path}"})

            records = _parse_mmdet_log_json(text)
            # Filter to epoch-level records
            epoch_records = [r for r in records if "epoch" in r]
            if not epoch_records:
                return json.dumps({"success": True, "message": "No epoch records found.", "total_log_lines": len(records)})

            last = epoch_records[-1]
            current_epoch = last.get("epoch", len(epoch_records))

            metrics: dict[str, Any] = {"current_epoch": current_epoch}
            for key in ("loss", "loss_cls", "loss_bbox", "loss_iou",
                        "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
                        "val_loss", "val_coco/bbox_mAP"):
                if key in last:
                    metrics[key] = round(last[key], 6) if isinstance(last[key], float) else last[key]

            return json.dumps({
                "success": True,
                "framework": "mmdet",
                "log_path": json_path,
                "progress": f"{current_epoch} epochs logged",
                "metrics": metrics,
                "latest_record": last,
            }, ensure_ascii=False)

        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

    except Exception as e:
        logger.exception("train_monitor failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: train_analyze
# ---------------------------------------------------------------------------

def train_analyze_handler(args: dict[str, Any]) -> str:
    """Analyze training trends: overfitting, underfitting, plateau detection.

    Parameters
    ----------
    args : dict
        ``log_path`` (str) **or** ``project_dir`` (str), ``framework`` (str),
        ``server`` (str).
    """
    log_path: str = args.get("log_path", "")
    project_dir: str = args.get("project_dir", "") or args.get("log_path", "")
    framework: str = args.get("framework", "")
    server: str = args.get("server", "")

    try:
        if not framework:
            framework = _detect_framework(project_dir, server)
            if framework == "unknown":
                return json.dumps({"success": False, "error": "Cannot detect framework."})

        if framework == "yolo":
            csv_path = log_path
            if not csv_path or Path(csv_path).is_dir():
                csv_path = _find_yolo_log(log_path or project_dir, server) or ""
            if not csv_path:
                return json.dumps({"success": False, "error": "Cannot find results.csv."})

            text = _read_file_content(csv_path, server)
            if text is None:
                return json.dumps({"success": False, "error": f"Cannot read {csv_path}"})

            rows = _parse_yolo_results_csv(text)
            if len(rows) < 3:
                return json.dumps({"success": True, "message": "Too few epochs to analyse (need >=3).", "epochs": len(rows)})

            # Extract metric series
            map50_key = "metrics/mAP50(B)"
            map5095_key = "metrics/mAP50-95(B)"
            train_loss_keys = [k for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss") if k in rows[0]]
            val_loss_keys = [k for k in ("val/box_loss", "val/cls_loss", "val/dfl_loss") if k in rows[0]]

            map50_values = [r.get(map50_key, 0) for r in rows] if map50_key in rows[0] else []
            map5095_values = [r.get(map5095_key, 0) for r in rows] if map5095_key in rows[0] else []

            analysis: dict[str, Any] = {
                "success": True,
                "framework": "yolo",
                "total_epochs": len(rows),
                "log_path": csv_path,
                "trends": {},
                "diagnosis": [],
                "suggestions": [],
            }

            # --- overfitting detection ---
            if val_loss_keys and train_loss_keys:
                n = min(10, len(rows))
                recent_train = sum(sum(r.get(k, 0) for k in train_loss_keys) for r in rows[-n:]) / n
                recent_val = sum(sum(r.get(k, 0) for k in val_loss_keys) for r in rows[-n:]) / n
                early_val = sum(sum(r.get(k, 0) for k in val_loss_keys) for r in rows[:n]) / n
                early_train = sum(sum(r.get(k, 0) for k in train_loss_keys) for r in rows[:n]) / n

                analysis["trends"]["recent_train_loss"] = round(recent_train, 6)
                analysis["trends"]["recent_val_loss"] = round(recent_val, 6)
                analysis["trends"]["early_train_loss"] = round(early_train, 6)
                analysis["trends"]["early_val_loss"] = round(early_val, 6)

                if recent_val > early_val * 1.15 and recent_train < early_train * 0.85:
                    analysis["diagnosis"].append("overfitting")
                    analysis["suggestions"].append("Consider: increase data augmentation, add dropout/weight-decay, reduce model size, or use early stopping.")
                elif recent_train > early_train * 0.95 and recent_val > early_val * 0.95:
                    analysis["diagnosis"].append("underfitting")
                    analysis["suggestions"].append("Consider: increase model size, train longer, increase learning rate, or improve data quality.")
                else:
                    analysis["diagnosis"].append("healthy")

            # --- mAP plateau ---
            if map5095_values and len(map5095_values) >= 10:
                recent_map = sum(map5095_values[-5:]) / 5
                prev_map = sum(map5095_values[-10:-5]) / 5
                improvement = recent_map - prev_map
                analysis["trends"]["recent_mAP50-95"] = round(recent_map, 6)
                analysis["trends"]["prev_mAP50-95"] = round(prev_map, 6)
                analysis["trends"]["mAP_improvement"] = round(improvement, 6)

                if abs(improvement) < 0.002:
                    analysis["diagnosis"].append("plateau")
                    analysis["suggestions"].append("mAP has plateaued. Consider: learning rate warmup/cosine, mosaic augmentation, larger image size, or switch to a larger model.")

            # Best epoch
            if map5095_values:
                best_idx = max(range(len(map5095_values)), key=lambda i: map5095_values[i])
                analysis["best_epoch"] = best_idx + 1
                analysis["best_mAP50-95"] = round(map5095_values[best_idx], 6)

            return json.dumps(analysis, ensure_ascii=False)

        elif framework == "mmdet":
            json_path = log_path
            if not json_path or Path(json_path).is_dir():
                json_path = _find_mmdet_log(log_path or project_dir, server) or ""
            if not json_path:
                return json.dumps({"success": False, "error": "Cannot find log.json."})

            text = _read_file_content(json_path, server)
            if text is None:
                return json.dumps({"success": False, "error": f"Cannot read {json_path}"})

            records = _parse_mmdet_log_json(text)
            epoch_records = [r for r in records if "epoch" in r]
            if len(epoch_records) < 3:
                return json.dumps({"success": True, "message": "Too few epochs to analyse.", "epochs": len(epoch_records)})

            loss_values = [r.get("loss", 0) for r in epoch_records if "loss" in r]
            map_values = [r.get("coco/bbox_mAP", 0) for r in epoch_records if "coco/bbox_mAP" in r]

            analysis: dict[str, Any] = {
                "success": True,
                "framework": "mmdet",
                "total_epochs": len(epoch_records),
                "log_path": json_path,
                "trends": {},
                "diagnosis": [],
                "suggestions": [],
            }

            if loss_values:
                n = min(10, len(loss_values))
                recent_loss = sum(loss_values[-n:]) / n
                early_loss = sum(loss_values[:n]) / n
                analysis["trends"]["recent_loss"] = round(recent_loss, 6)
                analysis["trends"]["early_loss"] = round(early_loss, 6)

                if recent_loss > early_loss * 1.1:
                    analysis["diagnosis"].append("loss_increasing")
                    analysis["suggestions"].append("Loss is increasing. Lower learning rate or check data pipeline.")

            if map_values:
                best_idx = max(range(len(map_values)), key=lambda i: map_values[i])
                analysis["best_epoch"] = best_idx + 1
                analysis["best_mAP"] = round(map_values[best_idx], 6)

                if len(map_values) >= 10:
                    recent_map = sum(map_values[-5:]) / 5
                    prev_map = sum(map_values[-10:-5]) / 5
                    if abs(recent_map - prev_map) < 0.002:
                        analysis["diagnosis"].append("plateau")
                        analysis["suggestions"].append("mAP plateaued. Try adjusting augmentation or model capacity.")

            if not analysis["diagnosis"]:
                analysis["diagnosis"].append("healthy")

            return json.dumps(analysis, ensure_ascii=False)

        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

    except Exception as e:
        logger.exception("train_analyze failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: train_resume
# ---------------------------------------------------------------------------

def train_resume_handler(args: dict[str, Any]) -> str:
    """Resume training from a checkpoint.

    Parameters
    ----------
    args : dict
        ``checkpoint_path`` (str), ``framework`` (str, default 'yolo'),
        ``config_path`` (str, required for mmdet), ``server`` (str),
        ``additional_epochs`` (int, default 50), ``device`` (str, default '0').
    """
    checkpoint_path: str = args.get("checkpoint_path", "")
    framework: str = args.get("framework", "yolo")
    config_path: str = args.get("config_path", "")
    server: str = args.get("server", "")
    additional_epochs: int = int(args.get("additional_epochs", 50))
    device: str = args.get("device", "") or args.get("gpu", "0")

    if not checkpoint_path:
        return json.dumps({"success": False, "error": "checkpoint_path is required"})

    try:
        if framework == "yolo":
            cmd = f"yolo train resume model={checkpoint_path} epochs={additional_epochs}"
            if device:
                cmd += f" device={device}"
        elif framework == "mmdet":
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet resume"})
            nproc = len(device.split(",")) if device else 1
            cmd = (
                f"python -m torch.distributed.launch --nproc_per_node={nproc} "
                f"--master_port=29500 "
                f"tools/dist_train.sh {config_path} {nproc} --resume-from {checkpoint_path}"
            )
        else:
            return json.dumps({"success": False, "error": f"Unknown framework: {framework}"})

        result = _exec_cmd(cmd, server=server, timeout=86400)

        return json.dumps({
            "success": True,
            "command": cmd,
            "framework": framework,
            "checkpoint": checkpoint_path,
            "additional_epochs": additional_epochs,
            "remote": bool(server),
            "result": result,
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception("train_resume failed")
        return json.dumps({"success": False, "error": str(e)})
