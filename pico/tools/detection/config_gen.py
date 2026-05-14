"""Config generation tools: YOLO data.yaml, MMDet config, hyperparameter recommendations."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _exec_cmd(command: str, server: str = "") -> dict[str, Any]:
    if server:
        from pico.tools.remote.remote_terminal import remote_terminal_handler
        return json.loads(remote_terminal_handler({"command": command, "server": server}))
    import subprocess
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return {"success": proc.returncode == 0, "output": proc.stdout + proc.stderr, "exit_code": proc.returncode}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _detect_dataset_structure(data_dir: str) -> dict[str, Any]:
    """Detect dataset structure and return stats for config generation."""
    data_path = Path(data_dir)
    result: dict[str, Any] = {"data_dir": data_dir}
    for split in ("train", "val", "valid", "test"):
        split_path = data_path / split
        if split_path.is_dir():
            result[f"{split}_dir"] = str(split_path)
            img_count = sum(1 for f in split_path.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            result[f"{split}_images"] = img_count
    for name in ("classes.txt", "classes.names", "names.txt"):
        p = data_path / name
        if p.is_file():
            classes = [line.strip() for line in p.read_text().strip().splitlines() if line.strip()]
            result["classes"] = classes
            result["num_classes"] = len(classes)
            result["classes_file"] = str(p)
            break
    if "total_images" not in result:
        total = 0
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            total += sum(1 for _ in data_path.rglob(f"*{ext}"))
        result["total_images"] = total
    return result


# Model size presets for YOLO
_YOLO_PRESETS: dict[str, dict[str, Any]] = {
    "n": {"epochs": 300, "batch": 16, "imgsz": 640, "lr0": 0.01},
    "s": {"epochs": 300, "batch": 16, "imgsz": 640, "lr0": 0.01},
    "m": {"epochs": 200, "batch": 8, "imgsz": 640, "lr0": 0.01},
    "l": {"epochs": 200, "batch": 4, "imgsz": 640, "lr0": 0.005},
    "x": {"epochs": 150, "batch": 2, "imgsz": 640, "lr0": 0.005},
}


# MMDet model configs
_MMDET_MODELS: dict[str, str] = {
    "rtmdet_s": "mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "rtmdet_m": "mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py",
    "rtmdet_l": "mmdet::rtmdet/rtmdet_l_8xb32-300e_coco.py",
    "rtmdet_tiny": "mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
    "faster-rcnn": "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "faster-rcnn-r101": "mmdet::faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py",
    "yolox_s": "mmdet::yolox/yolox_s_8xb8-300e_coco.py",
    "yolox_m": "mmdet::yolox/yolox_m_8xb8-300e_coco.py",
    "yolox_l": "mmdet::yolox/yolox_l_8xb8-300e_coco.py",
    "detr": "mmdet::detr/detr_r50_8xb2-150e_coco.py",
    "deformable-detr": "mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py",
}


def _run_remote_py(script_body: str, server: str) -> str:
    """Write a Python script to a temp file and execute it remotely, return last line."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_body)
        tmp_path = f.name
    try:
        from pico.tools.remote.file_transfer import file_upload_handler
        upload_result = json.loads(file_upload_handler({"local_path": tmp_path, "remote_path": f"/tmp/_pico_cfg_gen_{os.path.basename(tmp_path)}", "server": server}))
        if not upload_result.get("success"):
            return json.dumps({"success": False, "error": f"Failed to upload script: {upload_result.get('error', '')}"})
        remote_path = f"/tmp/_pico_cfg_gen_{os.path.basename(tmp_path)}"
        result = _exec_cmd(f"python3 {remote_path}", server)
        if result.get("success"):
            lines = result["output"].strip().splitlines()
            return lines[-1] if lines else json.dumps({"success": True, "output": ""})
        return json.dumps({"success": False, "error": result.get("error", result.get("output", ""))})
    finally:
        os.unlink(tmp_path)


def yolo_config_handler(args: dict[str, Any]) -> str:
    """Generate YOLO data.yaml and recommended training parameters."""
    data_dir: str = args.get("data_dir", "")
    output_path: str = args.get("output_path", "")
    model_size: str = args.get("model_size", "m")
    task: str = args.get("task", "detect")
    server: str = args.get("server", "")

    if not data_dir or not output_path:
        return json.dumps({"success": False, "error": "data_dir and output_path are required"})

    try:
        if server:
            script_body = (
                "import json, os\n"
                "from pathlib import Path\n"
                f"data_dir = {data_dir!r}\n"
                f"output_path = {output_path!r}\n"
                f"model_size = {model_size!r}\n"
                f"task = {task!r}\n"
                "IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}\n"
                "classes = []\n"
                "for name in ['classes.txt','classes.names','names.txt']:\n"
                "    p = Path(data_dir) / name\n"
                "    if p.is_file():\n"
                "        classes = [line.strip() for line in p.read_text().strip().splitlines() if line.strip()]\n"
                "        break\n"
                "data_path = Path(data_dir)\n"
                "yaml_lines = [f'path: {data_dir}']\n"
                "for split in ['train','val','valid','test']:\n"
                "    sp = data_path / split\n"
                "    if sp.is_dir():\n"
                "        yaml_lines.append(f'{split}: {sp}')\n"
                "yaml_lines.append(f'nc: {len(classes)}')\n"
                "yaml_lines.append(f'names: {json.dumps(classes)}')\n"
                "os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)\n"
                "with open(output_path, 'w') as f:\n"
                "    f.write('\\n'.join(yaml_lines) + '\\n')\n"
                "PRESETS = {'n':{'epochs':300,'batch':16},'s':{'epochs':300,'batch':16},'m':{'epochs':200,'batch':8},'l':{'epochs':200,'batch':4},'x':{'epochs':150,'batch':2}}\n"
                "preset = PRESETS.get(model_size, PRESETS['m'])\n"
                "seg_suffix = '-seg' if task == 'segment' else ''\n"
                f"print(json.dumps({{'success':True,'config_path':output_path,'num_classes':len(classes),'classes':classes,'recommended_params':{{**preset,'imgsz':640,'lr0':0.01,'model':f'yolo11{{model_size}}{{seg_suffix}}.pt','task':task}},'command':f'yolo train data={{output_path}} model=yolo11{{model_size}}{{seg_suffix}}.pt epochs={{preset[\"epochs\"]}} batch={{preset[\"batch\"]}} imgsz=640 lr0=0.01'}}}}))\n"
            )
            return _run_remote_py(script_body, server)

        # Local
        data_path = Path(data_dir)
        if not data_path.is_dir():
            return json.dumps({"success": False, "error": f"Directory not found: {data_dir}"})

        classes: list[str] = []
        for name in ("classes.txt", "classes.names", "names.txt"):
            p = data_path / name
            if p.is_file():
                classes = [line.strip() for line in p.read_text().strip().splitlines() if line.strip()]
                break

        yaml_lines = [f"path: {data_dir}"]
        for split in ("train", "val", "valid", "test"):
            sp = data_path / split
            if sp.is_dir():
                yaml_lines.append(f"{split}: {sp}")
        yaml_lines.append(f"nc: {len(classes)}")
        yaml_lines.append(f"names: {json.dumps(classes)}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(yaml_lines) + "\n")

        preset = _YOLO_PRESETS.get(model_size, _YOLO_PRESETS["m"])
        seg_suffix = "-seg" if task == "segment" else ""

        return json.dumps({
            "success": True,
            "config_path": output_path,
            "num_classes": len(classes),
            "classes": classes,
            "recommended_params": {
                **preset,
                "model": f"yolo11{model_size}{seg_suffix}.pt",
                "task": task,
            },
            "command": (
                f"yolo train data={output_path} model=yolo11{model_size}{seg_suffix}.pt "
                f"epochs={preset['epochs']} batch={preset['batch']} "
                f"imgsz={preset['imgsz']} lr0={preset['lr0']}"
            ),
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception("yolo_config failed")
        return json.dumps({"success": False, "error": str(e)})


def mmdet_config_handler(args: dict[str, Any]) -> str:
    """Generate MMDetection config file."""
    data_dir: str = args.get("data_dir", "")
    output_path: str = args.get("output_path", "")
    model: str = args.get("model", "rtmdet_s")
    num_classes: int = int(args.get("num_classes", 0))
    server: str = args.get("server", "")

    if not all([data_dir, output_path, num_classes]):
        return json.dumps({"success": False, "error": "data_dir, output_path, num_classes required"})

    try:
        base_config = _MMDET_MODELS.get(model, _MMDET_MODELS.get("rtmdet_s", ""))

        config_lines = [
            f"_base_ = ['{base_config}']",
            "",
            "# Dataset",
            f"data_root = '{data_dir}/'",
            f"metainfo = {{'classes': tuple(range({num_classes})), 'palette': [(i*50%255, i*100%255, i*150%255) for i in range({num_classes})]}}",
            "",
            "train_dataloader = dict(",
            "    dataset=dict(",
            "        data_root=data_root,",
            "        ann_file='annotations/train.json',",
            "        data_prefix=dict(img='images/'),",
            "        metainfo=metainfo,",
            "    )",
            ")",
            "val_dataloader = dict(",
            "    dataset=dict(",
            "        data_root=data_root,",
            "        ann_file='annotations/val.json',",
            "        data_prefix=dict(img='images/'),",
            "        metainfo=metainfo,",
            "    )",
            ")",
            "test_dataloader = val_dataloader",
            "",
            "val_evaluator = dict(ann_file=data_root + 'annotations/val.json')",
            "test_evaluator = val_evaluator",
            "",
            "# Training",
            "train_cfg = dict(max_epochs=100, val_interval=1)",
            "optim_wrapper = dict(optimizer=dict(lr=0.01))",
        ]
        config_content = "\n".join(config_lines) + "\n"

        if server:
            # Write config locally, upload, then return
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            local_tmp = output_path + ".local"
            with open(local_tmp, "w") as f:
                f.write(config_content)
            upload_res = json.loads(__import__("pico.tools.remote.file_transfer", fromlist=["file_upload_handler"]).file_upload_handler(
                {"local_path": local_tmp, "remote_path": output_path, "server": server}
            ))
            os.unlink(local_tmp)
            if not upload_res.get("success"):
                return json.dumps({"success": False, "error": f"Upload failed: {upload_res.get('error', '')}"})

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(config_content)

        return json.dumps({
            "success": True,
            "config_path": output_path,
            "base_config": base_config,
            "num_classes": num_classes,
            "data_dir": data_dir,
            "train_command": (
                f"python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 "
                f"tools/dist_train.py {output_path} 1 --work-dir work_dirs/{model}"
            ),
        })

    except Exception as e:
        logger.exception("mmdet_config failed")
        return json.dumps({"success": False, "error": str(e)})


def recommend_config_handler(args: dict[str, Any]) -> str:
    """Recommend training hyperparameters based on dataset stats and GPU info."""
    data_dir: str = args.get("data_dir", "")
    framework: str = args.get("framework", "yolo")
    gpu_info: dict[str, Any] = args.get("gpu_info", {})
    server: str = args.get("server", "")

    if not data_dir:
        return json.dumps({"success": False, "error": "data_dir is required"})

    try:
        # Collect dataset stats
        if server:
            from pico.tools.detection.dataset import dataset_explore_handler
            explore_result = json.loads(dataset_explore_handler({"data_dir": data_dir, "server": server}))
            stats = explore_result if explore_result.get("success") else {}
        else:
            from pico.tools.detection.dataset import (
                _IMAGE_EXTS,
                _detect_format,
                _find_classes_file,
                _read_classes_file,
            )
            data_path = Path(data_dir)
            fmt = _detect_format(data_dir)
            classes_file = _find_classes_file(data_dir)
            classes = _read_classes_file(classes_file) if classes_file else []
            total_images = sum(1 for _ in data_path.rglob("*") if _.suffix.lower() in _IMAGE_EXTS)
            stats = {"format": fmt, "num_classes": len(classes), "total_images": total_images}

        _num_classes = stats.get("num_classes", 1)
        total_images = stats.get("total_images", 1000)

        # GPU memory estimation (default: 8GB)
        gpu_mem_gb = 8
        if gpu_info and "gpus" in gpu_info:
            for g in gpu_info["gpus"]:
                mem = g.get("memory_total", "8000")
                try:
                    gpu_mem_gb = int("".join(c for c in str(mem) if c.isdigit())) / 1024
                except (ValueError, ZeroDivisionError):
                    gpu_mem_gb = 8
                break

        recommendations: dict[str, Any] = {
            "dataset_stats": stats,
            "gpu_memory_gb": round(gpu_mem_gb, 1),
        }

        if framework == "yolo":
            if total_images < 1000 or gpu_mem_gb < 6:
                model_size = "n"
            elif total_images < 5000 or gpu_mem_gb < 10:
                model_size = "s"
            elif total_images < 20000 or gpu_mem_gb < 16:
                model_size = "m"
            elif total_images < 50000:
                model_size = "l"
            else:
                model_size = "x"

            batch_table = {
                "n": {4: 32, 8: 16, 12: 16, 16: 16, 24: 32, 40: 32},
                "s": {4: 16, 8: 16, 12: 16, 16: 16, 24: 16, 40: 32},
                "m": {4: 8, 8: 8, 12: 16, 16: 16, 24: 16, 40: 16},
                "l": {4: 4, 8: 4, 12: 8, 16: 8, 24: 8, 40: 16},
                "x": {4: 2, 8: 2, 12: 4, 16: 4, 24: 8, 40: 8},
            }
            table = batch_table.get(model_size, batch_table["m"])
            closest_mem = min(table.keys(), key=lambda k: abs(k - gpu_mem_gb))
            batch_size = table[closest_mem]

            if total_images < 500:
                epochs = 300
            elif total_images < 5000:
                epochs = 200
            elif total_images < 20000:
                epochs = 150
            else:
                epochs = 100

            base_lr = 0.01
            lr = base_lr * (batch_size / 16)

            recommendations.update({
                "model_size": model_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "imgsz": 640,
                "lr0": round(lr, 5),
                "model": f"yolo11{model_size}.pt",
                "train_command": (
                    f"yolo train data=data.yaml model=yolo11{model_size}.pt "
                    f"epochs={epochs} batch={batch_size} imgsz=640 lr0={lr:.5f}"
                ),
            })

        elif framework == "mmdet":
            if total_images < 5000 or gpu_mem_gb < 10:
                model_rec = "rtmdet_s"
            elif total_images < 20000:
                model_rec = "rtmdet_m"
            else:
                model_rec = "rtmdet_l"

            batch_size = 8 if gpu_mem_gb >= 12 else (4 if gpu_mem_gb >= 8 else 2)
            epochs = 300 if total_images < 5000 else 200

            recommendations.update({
                "model": model_rec,
                "base_config": _MMDET_MODELS.get(model_rec, ""),
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": 0.01 * (batch_size / 8),
                "train_command": (
                    f"python -m torch.distributed.launch --nproc_per_node=1 "
                    f"tools/dist_train.py config.py {batch_size} --work-dir work_dirs/{model_rec}"
                ),
            })

        return json.dumps({"success": True, **recommendations}, ensure_ascii=False)

    except Exception as e:
        logger.exception("recommend_config failed")
        return json.dumps({"success": False, "error": str(e)})
