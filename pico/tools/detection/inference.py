"""Inference tools for object detection (YOLO / MMDetection).

Provides handlers for single-image, batch, and video inference.
All tools support local and remote execution via the ``server`` parameter.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _exec_cmd(command: str, server: str = "", timeout: int = 600, work_dir: str = "") -> dict[str, Any]:
    """Execute *command* locally or on *server*."""
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


# ---------------------------------------------------------------------------
# handler: inference_image (infer_image)
# ---------------------------------------------------------------------------

def inference_image_handler(args: dict[str, Any]) -> str:
    """Run inference on a single image.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``image_path`` (str),
        ``conf_threshold`` (float, default 0.25),
        ``output_path`` (str, default ''),
        ``framework`` (str, default 'yolo'),
        ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    image_path: str = args.get("image_path", "")
    conf: float = float(args.get("conf_threshold", 0.25))
    output_path: str = args.get("output_path", "")
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path:
        return json.dumps({"success": False, "error": "model_path is required"})
    if not image_path:
        return json.dumps({"success": False, "error": "image_path is required"})

    try:
        if framework == "yolo":
            # Build a one-liner Python script using ultralytics API
            script = (
                "import json\n"
                "from ultralytics import YOLO\n"
                f"model = YOLO('{model_path}')\n"
                f"results = model('{image_path}', conf={conf}, save={'True' if not output_path else 'False'}"
            )
            if output_path:
                script += f", project='{os.path.dirname(output_path) or '.'}'"
                script += f", name='{os.path.basename(output_path).rsplit('.', 1)[0]}'"
            script += (
                ", verbose=False)\n"
                "detections = []\n"
                "for r in results:\n"
                "    for box in r.boxes:\n"
                "        detections.append({\n"
                "            'class_id': int(box.cls.item()),\n"
                "            'class_name': r.names[int(box.cls.item())] if r.names else str(int(box.cls.item())),\n"
                "            'confidence': round(float(box.conf.item()), 4),\n"
                "            'bbox_xyxy': [round(x, 2) for x in box.xyxy[0].tolist()],\n"
                "            'bbox_xywhn': [round(x, 4) for x in box.xywhn[0].tolist()],\n"
                "        })\n"
                f"print(json.dumps({{'success': True, 'image': '{image_path}', 'num_detections': len(detections), 'detections': detections}}))\n"
            )

            # Execute locally or remotely
            if server:
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(script)
                    tmp_path = f.name
                try:
                    from pico.tools.remote.file_transfer import file_upload_handler
                    remote_path = f"/tmp/_pico_infer_{os.path.basename(tmp_path)}"
                    upload_res = json.loads(file_upload_handler({
                        "local_path": tmp_path, "remote_path": remote_path, "server": server,
                    }))
                    if not upload_res.get("success"):
                        return json.dumps({"success": False, "error": f"Upload failed: {upload_res.get('error', '')}"})
                    result = _exec_cmd(f"python3 {remote_path}", server=server, timeout=300)
                finally:
                    os.unlink(tmp_path)
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(script)
                    tmp_path = f.name
                try:
                    result = _exec_cmd(f"python3 {tmp_path}", server="", timeout=300)
                finally:
                    os.unlink(tmp_path)

            if result.get("success"):
                output = result.get("output", "").strip()
                lines = output.splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "success" in parsed:
                            return json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        continue
                return json.dumps({"success": True, "output": output[-2000:]})
            else:
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "Inference failed"))})

        elif framework == "mmdet":
            script = (
                "import json\n"
                "from mmdet.apis import init_detector, inference_detector\n"
                f"model = init_detector('{args.get('config_path', '')}', '{model_path}', device='cuda:0')\n"
                f"result = inference_detector(model, '{image_path}')\n"
                "pred = result.pred_instances\n"
                "detections = []\n"
                "for i in range(len(pred)):\n"
                "    bbox = pred.bboxes[i].tolist()\n"
                "    detections.append({\n"
                "        'class_id': int(pred.labels[i].item()),\n"
                "        'confidence': round(float(pred.scores[i].item()), 4),\n"
                "        'bbox_xyxy': [round(x, 2) for x in bbox],\n"
                "    })\n"
                f"detections = [d for d in detections if d['confidence'] >= {conf}]\n"
                f"print(json.dumps({{'success': True, 'image': '{image_path}', 'num_detections': len(detections), 'detections': detections}}))\n"
            )

            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                tmp_path = f.name
            try:
                result = _exec_cmd(f"python3 {tmp_path}", server=server, timeout=300)
            finally:
                os.unlink(tmp_path)

            if result.get("success"):
                output = result.get("output", "").strip()
                lines = output.splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "success" in parsed:
                            return json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        continue
                return json.dumps({"success": True, "output": output[-2000:]})
            else:
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "Inference failed"))})

        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

    except Exception as e:
        logger.exception("inference_image failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: inference_batch (infer_batch)
# ---------------------------------------------------------------------------

def inference_batch_handler(args: dict[str, Any]) -> str:
    """Run batch inference on a directory of images.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``image_dir`` (str), ``output_dir`` (str),
        ``conf_threshold`` (float, default 0.25),
        ``framework`` (str, default 'yolo'), ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    image_dir: str = args.get("image_dir", "")
    output_dir: str = args.get("output_dir", "")
    conf: float = float(args.get("conf_threshold", 0.25))
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path or not image_dir or not output_dir:
        return json.dumps({"success": False, "error": "model_path, image_dir, and output_dir are required"})

    try:
        if framework == "yolo":
            script = (
                "import json, os\n"
                "from pathlib import Path\n"
                "from ultralytics import YOLO\n"
                f"model = YOLO('{model_path}')\n"
                f"image_dir = Path('{image_dir}')\n"
                f"output_dir = Path('{output_dir}')\n"
                "output_dir.mkdir(parents=True, exist_ok=True)\n"
                f"conf = {conf}\n"
                "\n"
                "IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}\n"
                "images = [f for f in sorted(image_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS]\n"
                "\n"
                "results_summary = []\n"
                "for img_path in images:\n"
                "    try:\n"
                "        results = model(str(img_path), conf=conf, save=True, project=str(output_dir), name='.', exist_ok=True, verbose=False)\n"
                "        n_det = sum(len(r.boxes) for r in results)\n"
                "        results_summary.append({'image': str(img_path), 'detections': n_det})\n"
                "    except Exception as e:\n"
                "        results_summary.append({'image': str(img_path), 'error': str(e)})\n"
                "\n"
                "total = sum(r.get('detections', 0) for r in results_summary)\n"
                f"print(json.dumps({{'success': True, 'total_images': len(images), 'total_detections': total, 'output_dir': str(output_dir), 'per_image': results_summary[:50]}}))\n"
            )

        elif framework == "mmdet":
            config_path: str = args.get("config_path", "")
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet batch inference"})

            script = (
                "import json, os\n"
                "from pathlib import Path\n"
                "from mmdet.apis import init_detector, inference_detector\n"
                f"model = init_detector('{config_path}', '{model_path}', device='cuda:0')\n"
                f"image_dir = Path('{image_dir}')\n"
                f"output_dir = Path('{output_dir}')\n"
                "output_dir.mkdir(parents=True, exist_ok=True)\n"
                f"conf = {conf}\n"
                "\n"
                "IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}\n"
                "images = [f for f in sorted(image_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS]\n"
                "\n"
                "from mmdet.registry import VISUALIZERS\n"
                "visualizer = VISUALIZERS.build(model.cfg.visualizer)\n"
                "\n"
                "results_summary = []\n"
                "for img_path in images:\n"
                "    try:\n"
                "        result = inference_detector(model, str(img_path))\n"
                "        pred = result.pred_instances\n"
                "        keep = pred.scores >= conf\n"
                "        n_det = keep.sum().item()\n"
                "        # Save visualised image\n"
                "        out_path = output_dir / img_path.name\n"
                "        model.show_result(str(img_path), result, out_file=str(out_path), score_thr=conf)\n"
                "        results_summary.append({'image': str(img_path), 'detections': n_det})\n"
                "    except Exception as e:\n"
                "        results_summary.append({'image': str(img_path), 'error': str(e)})\n"
                "\n"
                "total = sum(r.get('detections', 0) for r in results_summary)\n"
                f"print(json.dumps({{'success': True, 'total_images': len(images), 'total_detections': total, 'output_dir': str(output_dir), 'per_image': results_summary[:50]}}))\n"
            )
        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

        # Execute
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name

        try:
            if server:
                from pico.tools.remote.file_transfer import file_upload_handler
                remote_path = f"/tmp/_pico_batch_{os.path.basename(tmp_path)}"
                upload_res = json.loads(file_upload_handler({
                    "local_path": tmp_path, "remote_path": remote_path, "server": server,
                }))
                if not upload_res.get("success"):
                    return json.dumps({"success": False, "error": f"Upload failed: {upload_res.get('error', '')}"})
                result = _exec_cmd(f"python3 {remote_path}", server=server, timeout=1800)
            else:
                result = _exec_cmd(f"python3 {tmp_path}", server="", timeout=1800)

            if result.get("success"):
                output = result.get("output", "").strip()
                lines = output.splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "success" in parsed:
                            return json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        continue
                return json.dumps({"success": True, "output": output[-2000:]})
            else:
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "Batch inference failed"))})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.exception("inference_batch failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: inference_video (infer_video)
# ---------------------------------------------------------------------------

def inference_video_handler(args: dict[str, Any]) -> str:
    """Run inference on a video file.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``video_path`` (str), ``output_path`` (str),
        ``conf_threshold`` (float, default 0.25),
        ``framework`` (str, default 'yolo'), ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    video_path: str = args.get("video_path", "")
    output_path: str = args.get("output_path", "")
    conf: float = float(args.get("conf_threshold", 0.25))
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path or not video_path:
        return json.dumps({"success": False, "error": "model_path and video_path are required"})

    try:
        if framework == "yolo":
            # Use YOLO CLI or API for video
            out_dir = os.path.dirname(output_path) or "."
            out_name = os.path.splitext(os.path.basename(output_path))[0] if output_path else "video_result"

            script = (
                "import json\n"
                "from ultralytics import YOLO\n"
                f"model = YOLO('{model_path}')\n"
                f"results = model('{video_path}', conf={conf}, save=True, "
                f"project='{out_dir}', name='{out_name}', exist_ok=True)\n"
                "\n"
                "total_frames = len(results)\n"
                "total_detections = sum(len(r.boxes) for r in results)\n"
                f"print(json.dumps({{'success': True, 'video': '{video_path}', 'total_frames': total_frames, 'total_detections': total_detections, 'output_dir': '{out_dir}/{out_name}'}}))\n"
            )

        elif framework == "mmdet":
            config_path: str = args.get("config_path", "")
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet video inference"})

            out_dir = os.path.dirname(output_path) or "."
            out_name = os.path.splitext(os.path.basename(output_path))[0] if output_path else "video_result"

            script = (
                "import json, cv2\n"
                "from pathlib import Path\n"
                "from mmdet.apis import init_detector, inference_detector\n"
                f"model = init_detector('{config_path}', '{model_path}', device='cuda:0')\n"
                f"cap = cv2.VideoCapture('{video_path}')\n"
                f"out_path = '{out_dir}/{out_name}.mp4'\n"
                "fps = cap.get(cv2.CAP_PROP_FPS) or 25\n"
                "w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n"
                "h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
                "fourcc = cv2.VideoWriter_fourcc(*'mp4v')\n"
                "writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))\n"
                f"conf = {conf}\n"
                "frame_count = 0\n"
                "det_count = 0\n"
                "while cap.isOpened():\n"
                "    ret, frame = cap.read()\n"
                "    if not ret:\n"
                "        break\n"
                "    result = inference_detector(model, frame)\n"
                "    pred = result.pred_instances\n"
                "    keep = pred.scores >= conf\n"
                "    det_count += keep.sum().item()\n"
                "    model.show_result(frame, result, out_file=None, show=False)\n"
                "    # Draw boxes manually for video\n"
                "    for i in range(len(pred)):\n"
                "        if pred.scores[i] < conf:\n"
                "            continue\n"
                "        x1, y1, x2, y2 = pred.bboxes[i].int().tolist()\n"
                "        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)\n"
                "        label = f'{pred.labels[i].item()}: {pred.scores[i].item():.2f}'\n"
                "        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)\n"
                "    writer.write(frame)\n"
                "    frame_count += 1\n"
                "cap.release()\n"
                "writer.release()\n"
                f"print(json.dumps({{'success': True, 'video': '{video_path}', 'total_frames': frame_count, 'total_detections': det_count, 'output_path': out_path}}))\n"
            )
        else:
            return json.dumps({"success": False, "error": f"Unsupported framework: {framework}"})

        # Execute
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name

        try:
            if server:
                from pico.tools.remote.file_transfer import file_upload_handler
                remote_path = f"/tmp/_pico_video_{os.path.basename(tmp_path)}"
                upload_res = json.loads(file_upload_handler({
                    "local_path": tmp_path, "remote_path": remote_path, "server": server,
                }))
                if not upload_res.get("success"):
                    return json.dumps({"success": False, "error": f"Upload failed: {upload_res.get('error', '')}"})
                result = _exec_cmd(f"python3 {remote_path}", server=server, timeout=3600)
            else:
                result = _exec_cmd(f"python3 {tmp_path}", server="", timeout=3600)

            if result.get("success"):
                output = result.get("output", "").strip()
                lines = output.splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "success" in parsed:
                            return json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        continue
                return json.dumps({"success": True, "output": output[-2000:]})
            else:
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "Video inference failed"))})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.exception("inference_video failed")
        return json.dumps({"success": False, "error": str(e)})
