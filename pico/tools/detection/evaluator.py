"""Evaluation tools for object detection (YOLO / MMDetection).

Provides handlers for model evaluation, PR curves, confusion matrices,
bad-case analysis, and model comparison.  All tools support local and
remote execution via the ``server`` parameter.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _exec_cmd(command: str, server: str = "", timeout: int = 600) -> dict[str, Any]:
    """Execute *command* locally or on *server*."""
    if server:
        from pico.tools.remote.remote_terminal import remote_terminal_handler
        return json.loads(remote_terminal_handler({"command": command, "server": server, "timeout": timeout}))

    import subprocess
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return {
            "success": proc.returncode == 0,
            "output": proc.stdout + proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _detect_framework(model_path: str) -> str:
    """Detect framework from model file extension."""
    ext = Path(model_path).suffix.lower()
    if ext in (".pt", ".pth", ".onnx", ".engine"):
        # Could be either — prefer YOLO for .pt/.onnx/.engine
        return "yolo"
    return "unknown"


# ---------------------------------------------------------------------------
# handler: evaluate (eval_model)
# ---------------------------------------------------------------------------

def evaluate_handler(args: dict[str, Any]) -> str:
    """Evaluate a model on a dataset split and return mAP metrics.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``data_dir`` (str), ``config_path`` (str, for mmdet),
        ``framework`` (str, default 'yolo'), ``split`` (str, default 'val'),
        ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    data_dir: str = args.get("data_dir", "")
    config_path: str = args.get("config_path", "")
    framework: str = args.get("framework", "yolo")
    split: str = args.get("split", "val")
    server: str = args.get("server", "")

    if not model_path:
        return json.dumps({"success": False, "error": "model_path is required"})

    try:
        if framework == "yolo":
            # Build YOLO val command
            cmd = f"yolo val model={model_path}"
            if data_dir:
                cmd += f" data={data_dir}"
            cmd += f" split={split}"

            result = _exec_cmd(cmd, server=server, timeout=600)
            if not result.get("success"):
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "Evaluation failed"))})

            # Parse YOLO val output for metrics
            output = result.get("output", "")
            metrics: dict[str, Any] = {"raw_output": output[-2000:] if len(output) > 2000 else output}

            # Try to extract key metrics from YOLO output
            for pattern, name in [
                (r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", "all_class"),
            ]:
                import re
                matches = re.findall(pattern, output)
                if matches:
                    last = matches[-1]
                    metrics["precision"] = float(last[0])
                    metrics["recall"] = float(last[1])
                    metrics["mAP50"] = float(last[2])
                    metrics["mAP50-95"] = float(last[3])
                    break

            # Also try structured metric lines
            for line in output.splitlines():
                line = line.strip()
                if "mAP50" in line and "mAP50-95" in line:
                    nums = re.findall(r"[\d.]+", line)
                    if len(nums) >= 2:
                        metrics["mAP50"] = float(nums[-2])
                        metrics["mAP50-95"] = float(nums[-1])

            return json.dumps({
                "success": True,
                "framework": "yolo",
                "model_path": model_path,
                "split": split,
                "metrics": metrics,
            }, ensure_ascii=False)

        elif framework == "mmdet":
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet evaluation"})

            cmd = f"python tools/test.py {config_path} {model_path} --eval bbox"
            result = _exec_cmd(cmd, server=server, timeout=600)

            if not result.get("success"):
                return json.dumps({"success": False, "error": result.get("error", result.get("output", ""))})

            output = result.get("output", "")
            metrics: dict[str, Any] = {"raw_output": output[-2000:] if len(output) > 2000 else output}

            # Parse MMDet eval output (JSON-like dict at the end)
            import re
            # Look for COCO metrics dict
            dict_match = re.search(r"\{[^{}]*'coco/bbox_mAP'[^{}]*\}", output)
            if dict_match:
                try:
                    eval_dict = ast.literal_eval(dict_match.group())
                    metrics.update({k: round(v, 6) if isinstance(v, float) else v for k, v in eval_dict.items()})
                except Exception:
                    pass

            return json.dumps({
                "success": True,
                "framework": "mmdet",
                "model_path": model_path,
                "config_path": config_path,
                "split": split,
                "metrics": metrics,
            }, ensure_ascii=False)

        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

    except Exception as e:
        logger.exception("evaluate failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: pr_curve
# ---------------------------------------------------------------------------

def pr_curve_handler(args: dict[str, Any]) -> str:
    """Generate precision-recall curves.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``data_dir`` (str), ``config_path`` (str, for mmdet),
        ``output_dir`` (str), ``framework`` (str, default 'yolo'),
        ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    data_dir: str = args.get("data_dir", "")
    config_path: str = args.get("config_path", "")
    output_dir: str = args.get("output_dir", "")
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path:
        return json.dumps({"success": False, "error": "model_path is required"})

    try:
        if framework == "yolo":
            # YOLO val with plots generates PR curves automatically
            cmd = f"yolo val model={model_path}"
            if data_dir:
                cmd += f" data={data_dir}"
            cmd += " plots=True"
            if output_dir:
                cmd += f" project={output_dir}"

            result = _exec_cmd(cmd, server=server, timeout=600)

            # Determine where plots were saved
            plot_dir = output_dir or ""
            if plot_dir:
                # YOLO creates a 'val' subdir
                if server:
                    ls_result = _exec_cmd(f"find {plot_dir} -name 'PR_curve.png' | head -1", server=server)
                    pr_path = ls_result.get("output", "").strip().split("\n")[0] if ls_result.get("success") else ""
                else:
                    from pathlib import Path
                    candidates = list(Path(plot_dir).rglob("PR_curve.png"))
                    pr_path = str(candidates[0]) if candidates else ""
            else:
                pr_path = "runs/val/exp/PR_curve.png (default YOLO output)"

            return json.dumps({
                "success": True,
                "framework": "yolo",
                "model_path": model_path,
                "pr_curve_path": pr_path,
                "output": result.get("output", "")[-1000:] if result.get("output") else "",
            }, ensure_ascii=False)

        elif framework == "mmdet":
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet PR curve"})

            # Use MMDet analysis tools
            out = output_dir or "work_dirs/pr_analysis"
            cmd = (
                f"python tools/analysis_tools/analyze_results.py "
                f"{config_path} {model_path} {out} --show"
            )
            result = _exec_cmd(cmd, server=server, timeout=600)

            return json.dumps({
                "success": True,
                "framework": "mmdet",
                "model_path": model_path,
                "output_dir": out,
                "output": result.get("output", "")[-1000:] if result.get("output") else "",
            }, ensure_ascii=False)

        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

    except Exception as e:
        logger.exception("pr_curve failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: confusion_matrix
# ---------------------------------------------------------------------------

def confusion_matrix_handler(args: dict[str, Any]) -> str:
    """Generate and visualise a confusion matrix.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``data_dir`` (str), ``config_path`` (str),
        ``output_dir`` (str), ``framework`` (str, default 'yolo'),
        ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    data_dir: str = args.get("data_dir", "")
    config_path: str = args.get("config_path", "")
    output_dir: str = args.get("output_dir", "")
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path:
        return json.dumps({"success": False, "error": "model_path is required"})

    try:
        if framework == "yolo":
            cmd = f"yolo val model={model_path}"
            if data_dir:
                cmd += f" data={data_dir}"
            cmd += " plots=True"
            if output_dir:
                cmd += f" project={output_dir}"

            result = _exec_cmd(cmd, server=server, timeout=600)

            cm_path = ""
            if output_dir:
                if server:
                    ls_result = _exec_cmd(f"find {output_dir} -name 'confusion_matrix*.png' | head -1", server=server)
                    cm_path = ls_result.get("output", "").strip().split("\n")[0] if ls_result.get("success") else ""
                else:
                    from pathlib import Path
                    candidates = list(Path(output_dir).rglob("confusion_matrix*.png"))
                    cm_path = str(candidates[0]) if candidates else ""
            else:
                cm_path = "runs/val/exp/confusion_matrix.png (default YOLO output)"

            return json.dumps({
                "success": True,
                "framework": "yolo",
                "model_path": model_path,
                "confusion_matrix_path": cm_path,
                "output": result.get("output", "")[-1000:] if result.get("output") else "",
            }, ensure_ascii=False)

        elif framework == "mmdet":
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet confusion matrix"})

            out = output_dir or "work_dirs/confusion_matrix"
            cmd = (
                f"python tools/analysis_tools/confusion_matrix.py "
                f"{config_path} {model_path} --out {out}"
            )
            result = _exec_cmd(cmd, server=server, timeout=600)

            return json.dumps({
                "success": True,
                "framework": "mmdet",
                "model_path": model_path,
                "output_dir": out,
                "output": result.get("output", "")[-1000:] if result.get("output") else "",
            }, ensure_ascii=False)

        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

    except Exception as e:
        logger.exception("confusion_matrix failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: bad_cases
# ---------------------------------------------------------------------------

def bad_cases_handler(args: dict[str, Any]) -> str:
    """Find and visualise mis-detected / missed objects.

    Runs inference on the validation set and compares predictions to ground
    truth to extract false positives, false negatives, and misclassifications.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``data_dir`` (str), ``output_dir`` (str),
        ``conf_threshold`` (float, default 0.25), ``max_images`` (int, default 50),
        ``framework`` (str, default 'yolo'), ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    data_dir: str = args.get("data_dir", "")
    output_dir: str = args.get("output_dir", "")
    conf_threshold: float = float(args.get("conf_threshold", 0.25))
    max_images: int = int(args.get("max_images", 50))
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path or not output_dir:
        return json.dumps({"success": False, "error": "model_path and output_dir are required"})

    try:
        # Build a Python script that runs inference and compares with labels
        if framework == "yolo":
            script = (
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "from ultralytics import YOLO\n"
                "import numpy as np\n"
                "from collections import defaultdict\n"
                "\n"
                f"model = YOLO('{model_path}')\n"
                f"data_dir = '{data_dir}'\n"
                f"output_dir = '{output_dir}'\n"
                f"conf = {conf_threshold}\n"
                f"max_images = {max_images}\n"
                "os.makedirs(output_dir, exist_ok=True)\n"
                "\n"
                "# Find val images and labels\n"
                "img_dir = None\n"
                "label_dir = None\n"
                "for split in ['val', 'valid', 'test']:\n"
                "    sp = Path(data_dir) / split\n"
                "    if sp.is_dir():\n"
                "        img_dir = sp / 'images' if (sp / 'images').is_dir() else sp\n"
                "        label_dir = sp / 'labels' if (sp / 'labels').is_dir() else sp\n"
                "        break\n"
                "if img_dir is None:\n"
                "    # Try flat structure\n"
                "    img_dir = Path(data_dir) / 'images' if (Path(data_dir) / 'images').is_dir() else Path(data_dir)\n"
                "    label_dir = Path(data_dir) / 'labels' if (Path(data_dir) / 'labels').is_dir() else Path(data_dir)\n"
                "\n"
                "IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}\n"
                "images = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS][:max_images]\n"
                "\n"
                "bad_cases = []\n"
                "for img_path in images:\n"
                "    label_path = label_dir / (img_path.stem + '.txt')\n"
                "    gt_boxes = []\n"
                "    if label_path.is_file():\n"
                "        for line in label_path.read_text().strip().splitlines():\n"
                "            parts = line.strip().split()\n"
                "            if len(parts) >= 5:\n"
                "                gt_boxes.append({'cls': int(parts[0]), 'xc': float(parts[1]), 'yc': float(parts[2]), 'w': float(parts[3]), 'h': float(parts[4])})\n"
                "\n"
                "    results = model(str(img_path), conf=conf, verbose=False)\n"
                "    pred_boxes = []\n"
                "    for r in results:\n"
                "        for box in r.boxes:\n"
                "            pred_boxes.append({'cls': int(box.cls.item()), 'conf': float(box.conf.item()), 'xywhn': box.xywhn[0].tolist()})\n"
                "\n"
                "    n_gt = len(gt_boxes)\n"
                "    n_pred = len(pred_boxes)\n"
                "    # Simple IoU check\n"
                "    def iou(a, b):\n"
                "        ax1, ay1 = a['xc']-a['w']/2, a['yc']-a['h']/2\n"
                "        ax2, ay2 = a['xc']+a['w']/2, a['yc']+a['h']/2\n"
                "        bx1, by1 = b['xywhn'][0]-b['xywhn'][2]/2, b['xywhn'][1]-b['xywhn'][3]/2\n"
                "        bx2, by2 = b['xywhn'][0]+b['xywhn'][2]/2, b['xywhn'][1]+b['xywhn'][3]/2\n"
                "        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)\n"
                "        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)\n"
                "        inter = max(0, ix2-ix1) * max(0, iy2-iy1)\n"
                "        area_a = a['w']*a['h']; area_b = b['xywhn'][2]*b['xywhn'][3]\n"
                "        return inter / (area_a + area_b - inter + 1e-6)\n"
                "\n"
                "    matched_gt = set()\n"
                "    issues = []\n"
                "    for pb in pred_boxes:\n"
                "        best_iou, best_gt = 0, -1\n"
                "        for gi, gb in enumerate(gt_boxes):\n"
                "            if gi in matched_gt:\n"
                "                continue\n"
                "            iou_val = iou(gb, pb)\n"
                "            if iou_val > best_iou:\n"
                "                best_iou = iou_val\n"
                "                best_gt = gi\n"
                "        if best_iou >= 0.5:\n"
                "            matched_gt.add(best_gt)\n"
                "            if gt_boxes[best_gt]['cls'] != pb['cls']:\n"
                "                issues.append({'type': 'misclassify', 'pred_cls': pb['cls'], 'gt_cls': gt_boxes[best_gt]['cls'], 'conf': pb['conf']})\n"
                "        else:\n"
                "            issues.append({'type': 'false_positive', 'pred_cls': pb['cls'], 'conf': pb['conf']})\n"
                "    for gi, gb in enumerate(gt_boxes):\n"
                "        if gi not in matched_gt:\n"
                "            issues.append({'type': 'missed', 'gt_cls': gb['cls']})\n"
                "\n"
                "    if issues:\n"
                "        bad_cases.append({'image': str(img_path), 'issues': issues, 'n_gt': n_gt, 'n_pred': n_pred})\n"
                "\n"
                "# Summary\n"
                "summary = defaultdict(int)\n"
                "for bc in bad_cases:\n"
                "    for issue in bc['issues']:\n"
                "        summary[issue['type']] += 1\n"
                "\n"
                "result = {'success': True, 'total_images_checked': len(images), 'bad_case_count': len(bad_cases), 'issue_summary': dict(summary), 'bad_cases': bad_cases[:20]}\n"
                "print(json.dumps(result))\n"
            )
        elif framework == "mmdet":
            # MMDet requires a different approach — use MMEval or custom script
            return json.dumps({
                "success": False,
                "error": "MMDet bad-case analysis not yet implemented. Use YOLO framework or implement a custom comparison script.",
            })
        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

        # Execute the script
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name

        try:
            if server:
                from pico.tools.remote.file_transfer import file_upload_handler
                remote_path = f"/tmp/_pico_bad_cases_{os.path.basename(tmp_path)}"
                upload_res = json.loads(file_upload_handler({
                    "local_path": tmp_path, "remote_path": remote_path, "server": server,
                }))
                if not upload_res.get("success"):
                    return json.dumps({"success": False, "error": f"Upload failed: {upload_res.get('error', '')}"})
                result = _exec_cmd(f"python3 {remote_path}", server=server, timeout=600)
            else:
                result = _exec_cmd(f"python3 {tmp_path}", server="", timeout=600)

            if result.get("success"):
                output = result.get("output", "").strip()
                # The last line should be JSON
                lines = output.splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    try:
                        return line
                    except Exception:
                        continue
                return json.dumps({"success": True, "output": output[-2000:]})
            else:
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "Bad-case analysis failed"))})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.exception("bad_cases failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: compare_models
# ---------------------------------------------------------------------------

def compare_models_handler(args: dict[str, Any]) -> str:
    """Compare multiple models side by side on the same dataset.

    Parameters
    ----------
    args : dict
        ``model_paths`` (list[str]), ``data_dir`` (str),
        ``framework`` (str, default 'yolo'), ``server`` (str).
    """
    model_paths: list[str] = args.get("model_paths", [])
    data_dir: str = args.get("data_dir", "")
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_paths:
        return json.dumps({"success": False, "error": "model_paths is required"})
    if not data_dir:
        return json.dumps({"success": False, "error": "data_dir is required"})

    try:
        results_list: list[dict[str, Any]] = []

        for mp in model_paths:
            eval_args: dict[str, Any] = {
                "model_path": mp,
                "data_dir": data_dir,
                "framework": framework,
                "split": "val",
                "server": server,
            }
            raw = evaluate_handler(eval_args)
            eval_result = json.loads(raw)
            results_list.append({
                "model_path": mp,
                "evaluation": eval_result,
            })

        # Build comparison table
        comparison: list[dict[str, Any]] = []
        for r in results_list:
            metrics = r["evaluation"].get("metrics", {}) if r["evaluation"].get("success") else {}
            comparison.append({
                "model": Path(r["model_path"]).name,
                "path": r["model_path"],
                "mAP50": metrics.get("mAP50"),
                "mAP50-95": metrics.get("mAP50-95"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "success": r["evaluation"].get("success", False),
            })

        # Find best model
        valid = [c for c in comparison if c.get("mAP50-95") is not None]
        best = max(valid, key=lambda c: c["mAP50-95"]) if valid else None

        return json.dumps({
            "success": True,
            "framework": framework,
            "data_dir": data_dir,
            "models_compared": len(model_paths),
            "comparison": comparison,
            "best_model": best,
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception("compare_models failed")
        return json.dumps({"success": False, "error": str(e)})
