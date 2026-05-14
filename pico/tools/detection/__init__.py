"""Object detection tools for Pico Agent.

Supports both YOLO (ultralytics) and MMDetection frameworks.
All tools support local and remote execution via the ``server`` parameter.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from pico.tools.detection.dataset import (
    dataset_explore_handler,
    dataset_split_handler,
    dataset_convert_handler,
)
from pico.tools.detection.annotation import (
    sam_annotate_handler,
    annotation_check_handler,
    annotation_visualize_handler,
    annotation_convert_handler,
)
from pico.tools.detection.config_gen import (
    yolo_config_handler,
    mmdet_config_handler,
    recommend_config_handler,
)
from pico.tools.detection.trainer import (
    train_start_handler,
    train_monitor_handler,
    train_analyze_handler,
    train_resume_handler,
)
from pico.tools.detection.evaluator import (
    evaluate_handler,
    pr_curve_handler,
    confusion_matrix_handler,
    bad_cases_handler,
    compare_models_handler,
)
from pico.tools.detection.inference import (
    inference_image_handler,
    inference_batch_handler,
    inference_video_handler,
)
from pico.tools.detection.exporter import (
    export_onnx_handler,
    export_tensorrt_handler,
    benchmark_handler,
)

logger = logging.getLogger(__name__)


def _check_ultralytics() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def _check_mmdet() -> bool:
    return importlib.util.find_spec("mmdet") is not None


def _check_detection() -> bool:
    """At least one detection framework must be available."""
    return _check_ultralytics() or _check_mmdet()


def register_tools(registry: Any) -> None:
    """Register all detection tools into *registry*."""

    check_det = _check_detection
    check_uv = _check_ultralytics
    check_mm = _check_mmdet

    # ---- dataset tools -----------------------------------------------------
    registry.register(
        name="dataset_explore",
        toolset="detection",
        schema={
            "name": "dataset_explore",
            "description": "Explore a dataset directory: detect format (YOLO/COCO/VOC), count images/annotations, and show class distribution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Path to dataset directory."},
                    "server": {"type": "string", "description": "Remote server name (empty for local).", "default": ""},
                },
                "required": ["data_dir"],
            },
        },
        handler=dataset_explore_handler,
        check_fn=check_det,
    )

    registry.register(
        name="dataset_split",
        toolset="detection",
        schema={
            "name": "dataset_split",
            "description": "Split dataset into train/val/test with stratified sampling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Path to dataset directory."},
                    "output_dir": {"type": "string", "description": "Output directory for split dataset."},
                    "ratios": {"type": "array", "items": {"type": "number"}, "description": "Split ratios [train, val, test]. Default [0.8, 0.15, 0.05].", "default": [0.8, 0.15, 0.05]},
                    "format": {"type": "string", "enum": ["yolo", "coco", "voc"], "description": "Output format.", "default": "yolo"},
                    "server": {"type": "string", "description": "Remote server name.", "default": ""},
                },
                "required": ["data_dir", "output_dir"],
            },
        },
        handler=dataset_split_handler,
        check_fn=check_det,
    )

    registry.register(
        name="dataset_convert",
        toolset="detection",
        schema={
            "name": "dataset_convert",
            "description": "Convert dataset between COCO, VOC, and YOLO formats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Source dataset directory."},
                    "output_dir": {"type": "string", "description": "Output directory."},
                    "source_format": {"type": "string", "enum": ["coco", "voc", "yolo"], "description": "Source format."},
                    "target_format": {"type": "string", "enum": ["coco", "voc", "yolo"], "description": "Target format."},
                    "classes_file": {"type": "string", "description": "Path to classes.txt (for YOLO).", "default": ""},
                    "server": {"type": "string", "description": "Remote server name.", "default": ""},
                },
                "required": ["data_dir", "output_dir", "source_format", "target_format"],
            },
        },
        handler=dataset_convert_handler,
        check_fn=check_det,
    )

    # ---- annotation tools --------------------------------------------------
    registry.register(
        name="sam_annotate",
        toolset="detection",
        schema={
            "name": "sam_annotate",
            "description": "Use SAM (Segment Anything Model) via ultralytics to auto-annotate images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_dir": {"type": "string", "description": "Directory of images to annotate."},
                    "output_dir": {"type": "string", "description": "Output directory for annotations."},
                    "prompts": {"type": "array", "items": {"type": "string"}, "description": "Text prompts for zero-shot detection (e.g., ['person', 'car']).", "default": []},
                    "model": {"type": "string", "description": "SAM model variant (sam_b, sam_l, sam2_b, etc.).", "default": "sam_b"},
                    "server": {"type": "string", "description": "Remote server name.", "default": ""},
                },
                "required": ["image_dir", "output_dir"],
            },
        },
        handler=sam_annotate_handler,
        check_fn=check_uv,
    )

    registry.register(
        name="annotation_check",
        toolset="detection",
        schema={
            "name": "annotation_check",
            "description": "Check annotation quality: out-of-bounds boxes, duplicates, tiny boxes, empty images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Dataset directory with images and labels."},
                    "format": {"type": "string", "enum": ["yolo", "coco", "voc"], "description": "Annotation format.", "default": "yolo"},
                    "server": {"type": "string", "description": "Remote server name.", "default": ""},
                },
                "required": ["data_dir"],
            },
        },
        handler=annotation_check_handler,
        check_fn=check_det,
    )

    registry.register(
        name="annotation_visualize",
        toolset="detection",
        schema={
            "name": "annotation_visualize",
            "description": "Draw bounding boxes on images and save visualizations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_dir": {"type": "string", "description": "Image directory."},
                    "label_dir": {"type": "string", "description": "Label directory (default: same as image_dir).", "default": ""},
                    "output_dir": {"type": "string", "description": "Output directory for visualized images."},
                    "classes_file": {"type": "string", "description": "Path to classes.txt.", "default": ""},
                    "format": {"type": "string", "enum": ["yolo", "coco", "voc"], "default": "yolo"},
                    "max_images": {"type": "integer", "description": "Max images to visualize.", "default": 50},
                    "server": {"type": "string", "description": "Remote server name.", "default": ""},
                },
                "required": ["image_dir", "output_dir"],
            },
        },
        handler=annotation_visualize_handler,
        check_fn=check_det,
    )

    registry.register(
        name="annotation_convert",
        toolset="detection",
        schema={
            "name": "annotation_convert",
            "description": "Convert annotation format (COCO↔VOC↔YOLO).",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input annotation file or directory."},
                    "output_path": {"type": "string", "description": "Output path."},
                    "source_format": {"type": "string", "enum": ["coco", "voc", "yolo"]},
                    "target_format": {"type": "string", "enum": ["coco", "voc", "yolo"]},
                    "classes_file": {"type": "string", "default": ""},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["input_path", "output_path", "source_format", "target_format"],
            },
        },
        handler=annotation_convert_handler,
        check_fn=check_det,
    )

    # ---- config generation -------------------------------------------------
    registry.register(
        name="yolo_config",
        toolset="detection",
        schema={
            "name": "yolo_config",
            "description": "Generate YOLO data.yaml and recommended training config.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Dataset directory."},
                    "output_path": {"type": "string", "description": "Path for generated data.yaml."},
                    "model_size": {"type": "string", "enum": ["n", "s", "m", "l", "x"], "description": "YOLO model size.", "default": "m"},
                    "task": {"type": "string", "enum": ["detect", "segment", "classify"], "default": "detect"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["data_dir", "output_path"],
            },
        },
        handler=yolo_config_handler,
        check_fn=check_uv,
    )

    registry.register(
        name="mmdet_config",
        toolset="detection",
        schema={
            "name": "mmdet_config",
            "description": "Generate MMDetection training config file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Dataset directory (COCO format)."},
                    "output_path": {"type": "string", "description": "Path for generated config."},
                    "model": {"type": "string", "description": "Model name (e.g., rtmdet_s, faster-rcnn).", "default": "rtmdet_s"},
                    "num_classes": {"type": "integer", "description": "Number of classes."},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["data_dir", "output_path", "num_classes"],
            },
        },
        handler=mmdet_config_handler,
        check_fn=check_mm,
    )

    registry.register(
        name="recommend_config",
        toolset="detection",
        schema={
            "name": "recommend_config",
            "description": "Recommend training hyperparameters based on dataset stats and GPU info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Dataset directory."},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "gpu_info": {"type": "object", "description": "GPU info dict (optional, will auto-detect if empty).", "default": {}},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["data_dir"],
            },
        },
        handler=recommend_config_handler,
        check_fn=check_det,
    )

    # ---- training ----------------------------------------------------------
    registry.register(
        name="train_start",
        toolset="detection",
        schema={
            "name": "train_start",
            "description": "Start object detection training (YOLO or MMDet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "config_path": {"type": "string", "description": "Training config file path."},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "epochs": {"type": "integer", "description": "Number of epochs (YOLO override).", "default": 100},
                    "batch_size": {"type": "integer", "description": "Batch size override."},
                    "device": {"type": "string", "description": "Device (e.g., 0, 0,1, cpu).", "default": "0"},
                    "work_dir": {"type": "string", "description": "Working directory for outputs.", "default": ""},
                    "background": {"type": "boolean", "description": "Run training in background.", "default": True},
                    "server": {"type": "string", "description": "Remote server name.", "default": ""},
                },
                "required": ["config_path"],
            },
        },
        handler=train_start_handler,
        check_fn=check_det,
    )

    registry.register(
        name="train_monitor",
        toolset="detection",
        schema={
            "name": "train_monitor",
            "description": "Monitor training progress by parsing logs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_path": {"type": "string", "description": "Path to results.csv (YOLO) or log.json (MMDet)."},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["log_path"],
            },
        },
        handler=train_monitor_handler,
        check_fn=check_det,
    )

    registry.register(
        name="train_analyze",
        toolset="detection",
        schema={
            "name": "train_analyze",
            "description": "Analyze training trends: detect overfitting/underfitting, suggest improvements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_path": {"type": "string", "description": "Path to results.csv or log.json."},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["log_path"],
            },
        },
        handler=train_analyze_handler,
        check_fn=check_det,
    )

    registry.register(
        name="train_resume",
        toolset="detection",
        schema={
            "name": "train_resume",
            "description": "Resume training from checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checkpoint_path": {"type": "string", "description": "Path to checkpoint file."},
                    "config_path": {"type": "string", "description": "Training config (MMDet needs this).", "default": ""},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "additional_epochs": {"type": "integer", "description": "Additional epochs to train.", "default": 50},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["checkpoint_path"],
            },
        },
        handler=train_resume_handler,
        check_fn=check_det,
    )

    # ---- evaluation --------------------------------------------------------
    registry.register(
        name="evaluate",
        toolset="detection",
        schema={
            "name": "evaluate",
            "description": "Evaluate model on test set, return mAP and per-class metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Path to model weights."},
                    "data_dir": {"type": "string", "description": "Dataset directory."},
                    "config_path": {"type": "string", "description": "Config file (MMDet).", "default": ""},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "split": {"type": "string", "enum": ["val", "test"], "default": "val"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path", "data_dir"],
            },
        },
        handler=evaluate_handler,
        check_fn=check_det,
    )

    registry.register(
        name="pr_curve",
        toolset="detection",
        schema={
            "name": "pr_curve",
            "description": "Generate precision-recall curves for each class.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "data_dir": {"type": "string"},
                    "output_dir": {"type": "string", "description": "Output directory for plots.", "default": ""},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path", "data_dir"],
            },
        },
        handler=pr_curve_handler,
        check_fn=check_det,
    )

    registry.register(
        name="confusion_matrix",
        toolset="detection",
        schema={
            "name": "confusion_matrix",
            "description": "Generate and visualize confusion matrix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "data_dir": {"type": "string"},
                    "output_dir": {"type": "string", "default": ""},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path", "data_dir"],
            },
        },
        handler=confusion_matrix_handler,
        check_fn=check_det,
    )

    registry.register(
        name="bad_cases",
        toolset="detection",
        schema={
            "name": "bad_cases",
            "description": "Find and visualize misdetected/missed objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "data_dir": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "conf_threshold": {"type": "number", "description": "Confidence threshold.", "default": 0.25},
                    "max_images": {"type": "integer", "default": 50},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path", "data_dir", "output_dir"],
            },
        },
        handler=bad_cases_handler,
        check_fn=check_det,
    )

    registry.register(
        name="compare_models",
        toolset="detection",
        schema={
            "name": "compare_models",
            "description": "Compare multiple models side by side on the same dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_paths": {"type": "array", "items": {"type": "string"}, "description": "List of model paths."},
                    "data_dir": {"type": "string"},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_paths", "data_dir"],
            },
        },
        handler=compare_models_handler,
        check_fn=check_det,
    )

    # ---- inference ---------------------------------------------------------
    registry.register(
        name="inference_image",
        toolset="detection",
        schema={
            "name": "inference_image",
            "description": "Run inference on a single image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "image_path": {"type": "string"},
                    "conf_threshold": {"type": "number", "default": 0.25},
                    "output_path": {"type": "string", "description": "Save annotated image.", "default": ""},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path", "image_path"],
            },
        },
        handler=inference_image_handler,
        check_fn=check_det,
    )

    registry.register(
        name="inference_batch",
        toolset="detection",
        schema={
            "name": "inference_batch",
            "description": "Run batch inference on a directory of images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "image_dir": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "conf_threshold": {"type": "number", "default": 0.25},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path", "image_dir", "output_dir"],
            },
        },
        handler=inference_batch_handler,
        check_fn=check_det,
    )

    registry.register(
        name="inference_video",
        toolset="detection",
        schema={
            "name": "inference_video",
            "description": "Run inference on a video file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "video_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "conf_threshold": {"type": "number", "default": 0.25},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path", "video_path", "output_path"],
            },
        },
        handler=inference_video_handler,
        check_fn=check_det,
    )

    # ---- export ------------------------------------------------------------
    registry.register(
        name="export_onnx",
        toolset="detection",
        schema={
            "name": "export_onnx",
            "description": "Export model to ONNX format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "output_path": {"type": "string", "default": ""},
                    "opset": {"type": "integer", "default": 17},
                    "dynamic": {"type": "boolean", "description": "Dynamic batch size.", "default": True},
                    "simplify": {"type": "boolean", "default": True},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path"],
            },
        },
        handler=export_onnx_handler,
        check_fn=check_det,
    )

    registry.register(
        name="export_tensorrt",
        toolset="detection",
        schema={
            "name": "export_tensorrt",
            "description": "Export model to TensorRT engine format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "output_path": {"type": "string", "default": ""},
                    "half": {"type": "boolean", "description": "Use FP16.", "default": True},
                    "workspace": {"type": "integer", "description": "TensorRT workspace in GB.", "default": 4},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path"],
            },
        },
        handler=export_tensorrt_handler,
        check_fn=check_det,
    )

    registry.register(
        name="benchmark",
        toolset="detection",
        schema={
            "name": "benchmark",
            "description": "Run speed benchmark on a model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "image_size": {"type": "integer", "default": 640},
                    "batch_sizes": {"type": "array", "items": {"type": "integer"}, "default": [1, 4, 8, 16]},
                    "num_runs": {"type": "integer", "default": 100},
                    "framework": {"type": "string", "enum": ["yolo", "mmdet"], "default": "yolo"},
                    "server": {"type": "string", "default": ""},
                },
                "required": ["model_path"],
            },
        },
        handler=benchmark_handler,
        check_fn=check_det,
    )
