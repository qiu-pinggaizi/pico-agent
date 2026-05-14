"""Workflow tools — high-level one-step operations that chain multiple tools.

These are the "easy mode" tools that let users do complex tasks in one call:
  - auto_train: detect dataset → recommend config → generate YAML → start training
  - quick_eval: load model → evaluate → show metrics + bad cases
  - deploy_model: export ONNX → benchmark → optionally upload to remote
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pico.tools.registry import ToolRegistry
from pico.tools.utils import _error, _success

logger = logging.getLogger(__name__)


def _run_cmd(cmd: list[str], cwd: str | None = None, timeout: int = 3600) -> tuple[int, str, str]:
    """Run a subprocess and return (exit_code, stdout, stderr)."""
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return proc.returncode, proc.stdout, proc.stderr


def _detect_dataset_structure(data_dir: Path) -> dict:
    """Auto-detect dataset format and structure.

    Returns dict with: format, class_count, train_count, val_count, classes, yaml_path
    """
    result = {
        "format": "unknown",
        "class_count": 0,
        "train_count": 0,
        "val_count": 0,
        "classes": [],
        "yaml_path": None,
        "image_dir": None,
        "label_dir": None,
    }

    # Check for existing YAML config
    for yaml_file in data_dir.glob("*.yaml"):
        import yaml
        try:
            with open(yaml_file) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and ("names" in cfg or "nc" in cfg):
                result["yaml_path"] = str(yaml_file)
                result["format"] = "yolo_yaml"
                result["class_count"] = cfg.get("nc", 0)
                result["classes"] = list(cfg.get("names", {}).values()) if isinstance(cfg.get("names"), dict) else cfg.get("names", [])
                # Count images
                for split in ["train", "val", "test"]:
                    split_path = cfg.get(split, "")
                    if split_path:
                        img_dir = Path(split_path) if Path(split_path).is_absolute() else data_dir / split_path
                        if img_dir.exists():
                            count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpeg")))
                            result[f"{split}_count"] = count
                return result
        except Exception:
            continue

    # Check for YOLO directory structure: images/train, labels/train
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if images_dir.exists() and labels_dir.exists():
        result["format"] = "yolo_dir"
        result["image_dir"] = str(images_dir)
        result["label_dir"] = str(labels_dir)

        for split in ["train", "val", "test"]:
            split_img = images_dir / split
            _split_lbl = labels_dir / split
            if split_img.exists():
                count = len(list(split_img.glob("*.jpg"))) + len(list(split_img.glob("*.png"))) + len(list(split_img.glob("*.jpeg")))
                result[f"{split}_count"] = count

        # Detect classes from label files
        classes = set()
        for lbl_file in list(labels_dir.rglob("*.txt"))[:100]:
            try:
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes.add(int(parts[0]))
            except (ValueError, IndexError):
                continue
        result["class_count"] = len(classes) if classes else 0

        # Try to find classes.txt
        classes_file = data_dir / "classes.txt"
        if classes_file.exists():
            result["classes"] = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]

        return result

    # Check for COCO structure
    for ann_file in ["annotations.json", "_annotations.coco.json", "result.json"]:
        ann_path = data_dir / ann_file
        if ann_path.exists():
            result["format"] = "coco"
            try:
                import json as _json
                with open(ann_path) as f:
                    coco = _json.load(f)
                cats = coco.get("categories", [])
                result["class_count"] = len(cats)
                result["classes"] = [c.get("name", f"class_{c['id']}") for c in cats]
                imgs = coco.get("images", [])
                result["train_count"] = len(imgs)
            except Exception:
                pass
            return result

    # Check for VOC structure
    xml_files = list(data_dir.rglob("*.xml"))
    if xml_files:
        result["format"] = "voc"
        result["train_count"] = len(xml_files)
        return result

    # Check for flat image directory (images + labels in same dir)
    img_count = len(list(data_dir.glob("*.jpg"))) + len(list(data_dir.glob("*.png")))
    lbl_count = len(list(data_dir.glob("*.txt")))
    if img_count > 0 and lbl_count > 0:
        result["format"] = "yolo_flat"
        result["train_count"] = img_count

    return result


def _generate_yolo_yaml(
    data_dir: Path,
    dataset_info: dict,
    output_path: Path | None = None,
    classes: list[str] | None = None,
    val_split: float = 0.2,
) -> str:
    """Generate a YOLO data.yaml file."""
    import yaml

    if output_path is None:
        output_path = data_dir / "data.yaml"

    nc = dataset_info.get("class_count", 0)
    names = classes or dataset_info.get("classes", [])
    if not names:
        names = [f"class_{i}" for i in range(nc)]

    # Determine paths
    images_dir = dataset_info.get("image_dir", "")
    if images_dir:
        train_path = str(Path(images_dir) / "train")
        val_path = str(Path(images_dir) / "val")
        if not Path(train_path).exists():
            train_path = str(Path(images_dir))
            val_path = str(Path(images_dir))
    else:
        train_path = str(data_dir / "images" / "train")
        val_path = str(data_dir / "images" / "val")
        if not Path(train_path).exists():
            train_path = str(data_dir)
            val_path = str(data_dir)

    config = {
        "path": str(data_dir),
        "train": train_path,
        "val": val_path,
        "nc": nc,
        "names": {i: name for i, name in enumerate(names)},
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return str(output_path)


def _recommend_model(dataset_info: dict) -> dict:
    """Recommend model and hyperparameters based on dataset characteristics."""
    train_count = dataset_info.get("train_count", 0)
    nc = dataset_info.get("class_count", 0)

    # Model size based on dataset size
    if train_count < 500:
        model = "yolov8n"
        epochs = 100
        imgsz = 640
        batch = 8
    elif train_count < 2000:
        model = "yolov8s"
        epochs = 150
        imgsz = 640
        batch = 16
    elif train_count < 10000:
        model = "yolov8m"
        epochs = 200
        imgsz = 640
        batch = 16
    elif train_count < 50000:
        model = "yolov8l"
        epochs = 300
        imgsz = 640
        batch = 8
    else:
        model = "yolov8x"
        epochs = 300
        imgsz = 640
        batch = 8

    return {
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "reasoning": (
            f"Dataset has {train_count} training images and {nc} classes. "
            f"Recommended {model} (small model for faster training, can upgrade if needed)."
        ),
    }


# ---------------------------------------------------------------------------
# auto_train — the main "one command to train" tool
# ---------------------------------------------------------------------------

def auto_train(
    data_dir: str,
    model: str = "",
    epochs: int = 0,
    imgsz: int = 0,
    batch: int = 0,
    output_dir: str = "",
    extra_args: str = "",
    dry_run: bool = False,
) -> str:
    """One-step training: auto-detect dataset → generate config → start training.

    This is the EASY MODE tool. Just point to a data directory and it figures
    out the rest. No need to manually write YAML configs or specify hyperparams.

    Supported dataset formats (auto-detected):
      - YOLO: images/train + labels/train structure
      - YOLO: data.yaml with paths
      - COCO: images + annotations.json
      - VOC: images + XML annotations
      - Flat: images and .txt labels in same directory

    Args:
        data_dir: Path to the dataset directory.
        model: Model to use (e.g. "yolov8n", "yolov8s", "yolov8m"). Leave empty for auto-recommend.
        epochs: Training epochs (0 = auto-recommend).
        imgsz: Image size (0 = auto 640).
        batch: Batch size (0 = auto).
        output_dir: Where to save results (default: runs/detect/train).
        extra_args: Extra ultralytics args, e.g. "lr0=0.01 optimizer=SGD".
        dry_run: If true, only show the plan without starting training.

    Returns:
        JSON with dataset analysis, recommended config, and training command/status.
    """
    try:
        p = Path(data_dir).expanduser().resolve()
        if not p.exists():
            return _error(f"Data directory not found: {data_dir}")

        # Step 1: Detect dataset structure
        ds_info = _detect_dataset_structure(p)
        if ds_info["format"] == "unknown":
            return _error(
                f"Could not detect dataset format in {data_dir}. "
                "Expected one of: YOLO (images/ + labels/), COCO (annotations.json), "
                "VOC (*.xml), or YOLO YAML (data.yaml)."
            )

        if ds_info["class_count"] == 0:
            return _error(
                f"Detected format '{ds_info['format']}' but found 0 classes. "
                "Make sure label files exist and contain class IDs, or add a classes.txt."
            )

        # Step 2: Generate YAML config if needed
        yaml_path = ds_info.get("yaml_path")
        if not yaml_path:
            yaml_path = _generate_yolo_yaml(p, ds_info)

        # Step 3: Recommend hyperparams
        rec = _recommend_model(ds_info)
        if model:
            rec["model"] = model
        if epochs > 0:
            rec["epochs"] = epochs
        if imgsz > 0:
            rec["imgsz"] = imgsz
        if batch > 0:
            rec["batch"] = batch

        # Step 4: Build training command
        out_dir = output_dir or str(p.parent / "runs" / "detect" / "train")

        cmd = [
            "yolo", "detect", "train",
            f"model={rec['model']}.pt",
            f"data={yaml_path}",
            f"epochs={rec['epochs']}",
            f"imgsz={rec['imgsz']}",
            f"batch={rec['batch']}",
            f"project={str(Path(out_dir).parent)}",
            f"name={Path(out_dir).name}",
        ]
        if extra_args:
            cmd.extend(extra_args.split())

        result = {
            "dataset": {
                "path": str(p),
                "format": ds_info["format"],
                "train_images": ds_info["train_count"],
                "val_images": ds_info["val_count"],
                "classes": ds_info["class_count"],
                "class_names": ds_info["classes"][:20],
                "yaml_config": yaml_path,
            },
            "recommendation": rec,
            "command": " ".join(cmd),
        }

        if dry_run:
            result["mode"] = "dry_run"
            result["message"] = "Dry run — add dry_run=false to start training."
            return _success(result)

        # Step 5: Start training
        logger.info("Starting training: %s", " ".join(cmd))
        exit_code, stdout, stderr = _run_cmd(cmd, timeout=86400)  # 24h max

        result["training"] = {
            "exit_code": exit_code,
            "success": exit_code == 0,
            "output_dir": out_dir,
            "stdout_tail": stdout[-3000:] if stdout else "",
            "stderr_tail": stderr[-1500:] if stderr and exit_code != 0 else "",
        }

        if exit_code == 0:
            # Find best weights
            best_weights = Path(out_dir) / "weights" / "best.pt"
            last_weights = Path(out_dir) / "weights" / "last.pt"
            result["training"]["best_weights"] = str(best_weights) if best_weights.exists() else ""
            result["training"]["last_weights"] = str(last_weights) if last_weights.exists() else ""

        return _success(result)

    except Exception as e:
        logger.exception("auto_train failed")
        return _error(f"Auto-train failed: {e}")


# ---------------------------------------------------------------------------
# quick_eval — one-step evaluation
# ---------------------------------------------------------------------------

def quick_eval(
    model_path: str,
    data_yaml: str = "",
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.25,
) -> str:
    """Quickly evaluate a trained model and show key metrics.

    Args:
        model_path: Path to the .pt model weights.
        data_yaml: Path to data.yaml (auto-detected if in same project dir).
        split: Which split to evaluate ("val" or "test").
        imgsz: Image size for evaluation.
        conf: Confidence threshold.

    Returns:
        JSON with mAP, precision, recall, per-class metrics, and confusion matrix path.
    """
    try:
        mp = Path(model_path).expanduser().resolve()
        if not mp.exists():
            return _error(f"Model not found: {model_path}")

        # Auto-detect data.yaml if not provided
        if not data_yaml:
            # Walk up to find data.yaml
            search_dir = mp.parent
            for _ in range(5):
                candidate = search_dir / "data.yaml"
                if candidate.exists():
                    data_yaml = str(candidate)
                    break
                candidate = search_dir / "*.yaml"
                for yf in search_dir.glob("*.yaml"):
                    import yaml
                    try:
                        with open(yf) as f:
                            cfg = yaml.safe_load(f)
                        if isinstance(cfg, dict) and "nc" in cfg:
                            data_yaml = str(yf)
                            break
                    except Exception:
                        continue
                if data_yaml:
                    break
                search_dir = search_dir.parent

        if not data_yaml:
            return _error("Could not auto-detect data.yaml. Please provide data_yaml parameter.")

        # Run evaluation
        cmd = [
            "yolo", "detect", "val",
            f"model={mp}",
            f"data={data_yaml}",
            f"split={split}",
            f"imgsz={imgsz}",
            f"conf={conf}",
        ]

        logger.info("Running evaluation: %s", " ".join(cmd))
        exit_code, stdout, stderr = _run_cmd(cmd, timeout=600)

        # Parse key metrics from stdout
        metrics = {}
        for line in stdout.split("\n"):
            line = line.strip()
            # Look for metric lines like "all    100    0.85    0.78    0.82"
            if "mAP50" in line or "mAP50-95" in line:
                metrics["summary_line"] = line
            # Parse specific metrics
            for key in ["mAP50", "mAP50-95", "Precision", "Recall"]:
                if key.lower() in line.lower():
                    import re
                    nums = re.findall(r"[\d.]+", line)
                    if nums:
                        metrics[key] = float(nums[-1])

        # Find output directory
        val_dir = mp.parent.parent / "val" if mp.parent.name == "weights" else mp.parent

        result = {
            "model": str(mp),
            "data_yaml": data_yaml,
            "exit_code": exit_code,
            "success": exit_code == 0,
            "metrics": metrics,
            "stdout_tail": stdout[-3000:] if stdout else "",
            "stderr_tail": stderr[-1500:] if stderr and exit_code != 0 else "",
        }

        # Find confusion matrix and PR curve images
        for img_name in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png", "results.png"]:
            # Search in common ultralytics output locations
            for search_dir_candidate in [val_dir, mp.parent.parent]:
                img_path = search_dir_candidate / img_name
                if img_path.exists():
                    result[img_name.replace(".png", "_path")] = str(img_path)

        return _success(result)

    except Exception as e:
        logger.exception("quick_eval failed")
        return _error(f"Evaluation failed: {e}")


# ---------------------------------------------------------------------------
# deploy_model — export + benchmark + optional upload
# ---------------------------------------------------------------------------

def deploy_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    benchmark: bool = True,
    remote_server: str = "",
    remote_path: str = "",
) -> str:
    """Export a trained model and optionally upload to remote server.

    Args:
        model_path: Path to the .pt model weights.
        format: Export format ("onnx", "torchscript", "engine", "tflite", "coreml").
        imgsz: Image size.
        benchmark: Run speed benchmark after export.
        remote_server: Remote server name from config (optional).
        remote_path: Path on remote server to upload to (optional).

    Returns:
        JSON with export path, benchmark results, and upload status.
    """
    try:
        mp = Path(model_path).expanduser().resolve()
        if not mp.exists():
            return _error(f"Model not found: {model_path}")

        # Step 1: Export
        cmd = [
            "yolo", "export",
            f"model={mp}",
            f"format={format}",
            f"imgsz={imgsz}",
        ]
        logger.info("Exporting model: %s", " ".join(cmd))
        exit_code, stdout, stderr = _run_cmd(cmd, timeout=600)

        if exit_code != 0:
            return _error(f"Export failed: {stderr[-500:]}")

        # Find exported file
        export_path = mp.with_suffix(f".{format}")
        if not export_path.exists():
            # Try common variations
            for ext in [".onnx", ".torchscript", ".engine", ".tflite"]:
                alt = mp.with_suffix(ext)
                if alt.exists():
                    export_path = alt
                    break

        result = {
            "model": str(mp),
            "export_format": format,
            "export_path": str(export_path) if export_path.exists() else "",
            "export_success": exit_code == 0,
        }

        # Step 2: Benchmark
        if benchmark and export_path.exists():
            try:
                from ultralytics import YOLO
                model = YOLO(str(export_path))
                bm = model.benchmark(imgsz=imgsz)
                result["benchmark"] = {
                    "speed_ms": bm.get("speed", {}),
                    "note": "Lower is faster",
                }
            except Exception as e:
                result["benchmark"] = {"error": str(e)}

        # Step 3: Upload to remote (if configured)
        if remote_server and export_path.exists():
            result["upload"] = {
                "server": remote_server,
                "remote_path": remote_path or f"/tmp/models/{export_path.name}",
                "note": "Use remote_file_upload tool for actual transfer",
            }

        return _success(result)

    except Exception as e:
        logger.exception("deploy_model failed")
        return _error(f"Deploy failed: {e}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry: ToolRegistry) -> None:
    """Register workflow tools."""
    registry.register(
        name="auto_train",
        toolset="detection",
        description=(
            "ONE-STEP TRAINING: Point to a data directory and train a detection model. "
            "Auto-detects dataset format (YOLO/COCO/VOC), generates config, "
            "recommends model size, and starts training. "
            "Use dry_run=true first to see the plan without training."
        ),
        parameters={
            "type": "object",
            "properties": {
                "data_dir": {"type": "string", "description": "Path to dataset directory."},
                "model": {"type": "string", "description": "YOLO model (e.g. 'yolov8n', 'yolov8s'). Empty = auto-recommend.", "default": ""},
                "epochs": {"type": "integer", "description": "Training epochs. 0 = auto-recommend.", "default": 0},
                "imgsz": {"type": "integer", "description": "Image size. 0 = auto 640.", "default": 0},
                "batch": {"type": "integer", "description": "Batch size. 0 = auto.", "default": 0},
                "output_dir": {"type": "string", "description": "Output directory for results.", "default": ""},
                "extra_args": {"type": "string", "description": "Extra ultralytics args, e.g. 'lr0=0.01 optimizer=SGD augment=true'.", "default": ""},
                "dry_run": {"type": "boolean", "description": "If true, show plan without training. Default false.", "default": False},
            },
            "required": ["data_dir"],
        },
        handler=auto_train,
    )

    registry.register(
        name="quick_eval",
        toolset="detection",
        description=(
            "Quickly evaluate a trained model. Shows mAP, precision, recall, "
            "and saves confusion matrix / PR curve. "
            "Auto-detects data.yaml if not provided."
        ),
        parameters={
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to .pt model weights."},
                "data_yaml": {"type": "string", "description": "Path to data.yaml (auto-detected if empty).", "default": ""},
                "split": {"type": "string", "description": "Split to evaluate: 'val' or 'test'.", "default": "val"},
                "imgsz": {"type": "integer", "description": "Image size.", "default": 640},
                "conf": {"type": "number", "description": "Confidence threshold.", "default": 0.25},
            },
            "required": ["model_path"],
        },
        handler=quick_eval,
    )

    registry.register(
        name="deploy_model",
        toolset="detection",
        description=(
            "Export a trained model to deployment format (ONNX, TensorRT, etc.), "
            "run speed benchmark, and optionally prepare for remote upload."
        ),
        parameters={
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to .pt model weights."},
                "format": {"type": "string", "description": "Export format: 'onnx', 'torchscript', 'engine', 'tflite'.", "default": "onnx"},
                "imgsz": {"type": "integer", "description": "Image size.", "default": 640},
                "benchmark": {"type": "boolean", "description": "Run speed benchmark after export.", "default": True},
                "remote_server": {"type": "string", "description": "Remote server name from config (optional).", "default": ""},
                "remote_path": {"type": "string", "description": "Path on remote server.", "default": ""},
            },
            "required": ["model_path"],
        },
        handler=deploy_model,
    )
