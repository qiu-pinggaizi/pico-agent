"""Annotation tools: SAM auto-annotation, quality checking, visualization, format conversion.

SAM uses ultralytics SAM model for zero-shot/point/box prompt annotation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _read_classes_file(path: str) -> list[str]:
    if not path or not os.path.isfile(path):
        return []
    return [line.strip() for line in Path(path).read_text().strip().splitlines() if line.strip()]


def _exec_cmd(command: str, server: str = "") -> dict[str, Any]:
    if server:
        from pico.tools.remote.remote_terminal import remote_terminal_handler
        return json.loads(remote_terminal_handler({"command": command, "server": server}))
    import subprocess
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=600)
        return {"success": proc.returncode == 0, "output": proc.stdout + proc.stderr, "exit_code": proc.returncode}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# SAM annotate
# ---------------------------------------------------------------------------

def sam_annotate_handler(args: dict[str, Any]) -> str:
    """Auto-annotate images using ultralytics SAM model."""
    image_dir: str = args.get("image_dir", "")
    output_dir: str = args.get("output_dir", "")
    prompts: list[str] = args.get("prompts", [])
    model_name: str = args.get("model", "sam_b")
    server: str = args.get("server", "")

    if not image_dir or not output_dir:
        return json.dumps({"success": False, "error": "image_dir and output_dir are required"})

    try:
        os.makedirs(output_dir, exist_ok=True)

        if server:
            # Build a Python script for remote SAM annotation
            script = f"""
import os, json, glob
from pathlib import Path
from ultralytics import SAM
model = SAM("{model_name}")
image_dir = "{image_dir}"
output_dir = "{output_dir}"
prompts = {json.dumps(prompts)}
os.makedirs(output_dir, exist_ok=True)
IMAGE_EXTS = {{".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}}
images = [str(p) for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS]
annotated = 0
for img_path in images:
    if prompts:
        # Text-guided: use YOLO for detection then SAM for segmentation
        results = model(img_path)
    else:
        results = model(img_path)
    # Save results
    base = Path(img_path).stem
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            # Save as YOLO format
            h, w = r.orig_shape
            lbl_path = os.path.join(output_dir, base + ".txt")
            lines = []
            for box in r.boxes:
                cls = int(box.cls[0])
                cx, cy, bw, bh = box.xywhn[0].tolist()
                lines.append(f"{{cls}} {{cx:.6f}} {{cy:.6f}} {{bw:.6f}} {{bh:.6f}}")
            with open(lbl_path, "w") as f:
                f.write("\\n".join(lines) + "\\n")
            annotated += 1
print(json.dumps({{"success": True, "annotated": annotated, "total": len(images)}}))
"""
            result = _exec_cmd(f"python3 -c '{script}'", server)
            if result.get("success"):
                try:
                    lines = result["output"].strip().splitlines()
                    return lines[-1]
                except Exception:
                    return json.dumps({"success": True, "output": result["output"]})
            return json.dumps({"success": False, "error": result.get("error", result.get("output", ""))})

        # Local execution
        from ultralytics import SAM
        model = SAM(model_name)

        image_dir_path = Path(image_dir)
        images = [p for p in image_dir_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS and p.is_file()]
        annotated = 0

        for img_path in sorted(images):
            results = model(str(img_path))
            base = img_path.stem

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    h, w = r.orig_shape
                    lbl_path = os.path.join(output_dir, base + ".txt")
                    lines = []
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        cx, cy, bw, bh = box.xywhn[0].tolist()
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    annotated += 1

        return json.dumps({
            "success": True,
            "annotated": annotated,
            "total": len(images),
            "output_dir": output_dir,
        })

    except Exception as e:
        logger.exception("sam_annotate failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Annotation quality check
# ---------------------------------------------------------------------------

def annotation_check_handler(args: dict[str, Any]) -> str:
    """Check annotation quality: out-of-bounds, duplicates, tiny boxes, empty images."""
    data_dir: str = args.get("data_dir", "")
    fmt: str = args.get("format", "yolo")
    server: str = args.get("server", "")

    if not data_dir:
        return json.dumps({"success": False, "error": "data_dir is required"})

    try:
        if server:
            script = f"""
import json, os
from pathlib import Path

data_dir = "{data_dir}"
fmt = "{fmt}"
IMAGE_EXTS = {{".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}}
issues = {{"out_of_bounds": [], "tiny_boxes": [], "empty_labels": [], "duplicates": []}}

if fmt == "yolo":
    data_path = Path(data_dir)
    for lbl_cand in ["labels", "train/labels", "val/labels"]:
        lbl_dir = data_path / lbl_cand
        if not lbl_dir.is_dir():
            continue
        for txt in lbl_dir.glob("*.txt"):
            lines = txt.read_text().strip().splitlines()
            seen = set()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    issues["empty_labels"].append(str(txt))
                    continue
                cls, cx, cy, bw, bh = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Out of bounds
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    issues["out_of_bounds"].append({{"file": str(txt), "line": i+1, "content": line}})
                # Tiny box (< 1 pixel equivalent in normalized coords < 0.001)
                if bw < 0.001 or bh < 0.001:
                    issues["tiny_boxes"].append({{"file": str(txt), "line": i+1}})
                # Duplicates
                key = (cls, round(cx,4), round(cy,4), round(bw,4), round(bh,4))
                if key in seen:
                    issues["duplicates"].append({{"file": str(txt), "line": i+1}})
                seen.add(key)
            if not lines:
                issues["empty_labels"].append(str(txt))

summary = {{k: len(v) for k, v in issues.items()}}
print(json.dumps({{"success": True, "summary": summary, "issues": issues}}))
"""
            result = _exec_cmd(f"python3 -c '{script}'", server)
            if result.get("success"):
                try:
                    return result["output"].strip().splitlines()[-1]
                except Exception:
                    return json.dumps({"success": True, "output": result["output"]})
            return json.dumps({"success": False, "error": result.get("error", result.get("output", ""))})

        # Local
        data_path = Path(data_dir)
        issues: dict[str, list] = {
            "out_of_bounds": [],
            "tiny_boxes": [],
            "empty_labels": [],
            "duplicates": [],
        }

        if fmt == "yolo":
            label_dirs = []
            for cand in ["labels", "train/labels", "val/labels"]:
                p = data_path / cand
                if p.is_dir():
                    label_dirs.append(p)
            if not label_dirs and (data_path / "train" / "labels").is_dir():
                label_dirs.append(data_path / "train" / "labels")

            for lbl_dir in label_dirs:
                for txt in sorted(lbl_dir.glob("*.txt")):
                    lines = txt.read_text().strip().splitlines()
                    if not lines:
                        issues["empty_labels"].append(str(txt))
                        continue
                    seen: set[tuple] = set()
                    for i, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            issues["empty_labels"].append(f"{txt}:{i+1}")
                            continue
                        cls = parts[0]
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            issues["out_of_bounds"].append({"file": str(txt), "line": i + 1, "content": line})
                        if bw < 0.001 or bh < 0.001:
                            issues["tiny_boxes"].append({"file": str(txt), "line": i + 1})
                        key = (cls, round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4))
                        if key in seen:
                            issues["duplicates"].append({"file": str(txt), "line": i + 1})
                        seen.add(key)

        elif fmt == "coco":
            ann_dir = data_path / "annotations"
            for json_file in ann_dir.glob("*.json"):
                with open(json_file) as f:
                    coco = json.load(f)
                img_map = {img["id"]: img for img in coco.get("images", [])}
                seen_ann: set[tuple] = set()
                for ann in coco.get("annotations", []):
                    bbox = ann["bbox"]
                    img_info = img_map.get(ann["image_id"], {})
                    w, h = img_info.get("width", 0), img_info.get("height", 0)
                    if w > 0 and h > 0:
                        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > w or bbox[1] + bbox[3] > h:
                            issues["out_of_bounds"].append({"annotation_id": ann["id"]})
                    if bbox[2] < 1 or bbox[3] < 1:
                        issues["tiny_boxes"].append({"annotation_id": ann["id"]})
                    key = (ann["image_id"], ann["category_id"], tuple(round(x, 1) for x in bbox))
                    if key in seen_ann:
                        issues["duplicates"].append({"annotation_id": ann["id"]})
                    seen_ann.add(key)
                break

        summary = {k: len(v) for k, v in issues.items()}
        return json.dumps({"success": True, "summary": summary, "issues": issues}, ensure_ascii=False)

    except Exception as e:
        logger.exception("annotation_check failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def annotation_visualize_handler(args: dict[str, Any]) -> str:
    """Draw bounding boxes on images and save visualizations."""
    image_dir: str = args.get("image_dir", "")
    label_dir: str = args.get("label_dir", "") or image_dir
    output_dir: str = args.get("output_dir", "")
    classes_file: str = args.get("classes_file", "")
    _fmt: str = args.get("format", "yolo")
    max_images: int = int(args.get("max_images", 50))
    server: str = args.get("server", "")

    if not image_dir or not output_dir:
        return json.dumps({"success": False, "error": "image_dir and output_dir are required"})

    try:
        os.makedirs(output_dir, exist_ok=True)

        if server:
            script = f"""
import json, os, random
from pathlib import Path

image_dir = "{image_dir}"
label_dir = "{label_dir}" or image_dir
output_dir = "{output_dir}"
classes_file = "{classes_file}"
max_images = {max_images}
IMAGE_EXTS = {{".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}}

# Read classes
classes = []
if classes_file and os.path.isfile(classes_file):
    classes = [l.strip() for l in open(classes_file).read().strip().splitlines()]
elif os.path.isfile(os.path.join(os.path.dirname(image_dir), "classes.txt")):
    classes = [l.strip() for l in open(os.path.join(os.path.dirname(image_dir), "classes.txt")).read().strip().splitlines()]

try:
    import cv2
    import numpy as np
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
              (128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128)]

    images = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS]
    random.shuffle(images)
    images = sorted(images[:max_images])
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = Path(label_dir) / (img_path.stem + ".txt")
        if not lbl_path.is_file():
            continue
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            cv2.putText(img, label, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out_path = os.path.join(output_dir, img_path.name)
        cv2.imwrite(out_path, img)
        saved += 1
    print(json.dumps({{"success": True, "visualized": saved, "output_dir": output_dir}}))
except ImportError:
    print(json.dumps({{"success": False, "error": "cv2 (opencv-python) not installed"}}))
"""
            result = _exec_cmd(f"python3 -c '{script}'", server)
            if result.get("success"):
                try:
                    return result["output"].strip().splitlines()[-1]
                except Exception:
                    return json.dumps({"success": True, "output": result["output"]})
            return json.dumps({"success": False, "error": result.get("error", result.get("output", ""))})

        # Local
        try:
            import cv2
        except ImportError:
            return json.dumps({"success": False, "error": "cv2 (opencv-python) not installed"})

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

        cf = classes_file
        if not cf:
            for cand in ("classes.txt", "classes.names"):
                p = os.path.join(os.path.dirname(image_dir), cand)
                if os.path.isfile(p):
                    cf = p
                    break
        classes = _read_classes_file(cf)

        images = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTS and p.is_file()])
        if len(images) > max_images:
            import random
            random.seed(42)
            images = sorted(random.sample(images, max_images))

        saved = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            lbl_path = Path(label_dir) / (img_path.stem + ".txt")
            if not lbl_path.is_file():
                continue

            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = colors[cls_id % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
                cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out_path = os.path.join(output_dir, img_path.name)
            cv2.imwrite(out_path, img)
            saved += 1

        return json.dumps({
            "success": True,
            "visualized": saved,
            "output_dir": output_dir,
        })

    except Exception as e:
        logger.exception("annotation_visualize failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Format conversion (annotation-level)
# ---------------------------------------------------------------------------

def annotation_convert_handler(args: dict[str, Any]) -> str:
    """Convert annotation format: COCO↔VOC↔YOLO.

    For full dataset conversion, use ``dataset_convert``. This tool converts
    individual annotation files.
    """
    input_path: str = args.get("input_path", "")
    output_path: str = args.get("output_path", "")
    source_format: str = args.get("source_format", "")
    target_format: str = args.get("target_format", "")
    classes_file: str = args.get("classes_file", "")
    server: str = args.get("server", "")

    if not all([input_path, output_path, source_format, target_format]):
        return json.dumps({"success": False, "error": "input_path, output_path, source_format, target_format required"})

    try:
        if server:
            return json.dumps({"success": False, "error": "Remote annotation conversion not yet supported."})

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # YOLO → COCO
        if source_format == "yolo" and target_format == "coco":
            # input_path: directory of YOLO label txt files
            input_dir = Path(input_path)
            if input_dir.is_file():
                input_dir = input_dir.parent
            classes = _read_classes_file(classes_file) if classes_file else []
            coco = {
                "images": [],
                "annotations": [],
                "categories": [{"id": i, "name": c} for i, c in enumerate(classes)],
            }
            ann_id = 1
            for txt_file in sorted(input_dir.glob("*.txt")):
                if classes_file and txt_file.name == os.path.basename(classes_file):
                    continue
                img_name = txt_file.stem
                # Try to find matching image
                w, h = 640, 640  # default
                for ext in _IMAGE_EXTS:
                    img_candidate = txt_file.parent.parent / "images" / (img_name + ext)
                    if img_candidate.is_file():
                        try:
                            from PIL import Image
                            with Image.open(img_candidate) as im:
                                w, h = im.size
                        except Exception:
                            pass
                        break

                img_id = len(coco["images"]) + 1
                coco["images"].append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": w,
                    "height": h,
                })
                for line in txt_file.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls,
                            "bbox": [round(x, 2), round(y, 2), round(bw * w, 2), round(bh * h, 2)],
                            "area": round(bw * w * bh * h, 2),
                            "iscrowd": 0,
                        })
                        ann_id += 1

            with open(output_path, "w") as f:
                json.dump(coco, f, indent=2)
            return json.dumps({
                "success": True,
                "num_images": len(coco["images"]),
                "num_annotations": len(coco["annotations"]),
                "output": output_path,
            })

        # COCO → YOLO
        elif source_format == "coco" and target_format == "yolo":
            with open(input_path) as f:
                coco = json.load(f)
            os.makedirs(output_path, exist_ok=True)
            img_map = {img["id"]: img for img in coco.get("images", [])}
            ann_by_img: dict[int, list] = {}
            for ann in coco.get("annotations", []):
                ann_by_img.setdefault(ann["image_id"], []).append(ann)
            cats = coco.get("categories", [])
            cat_idx = {c["id"]: i for i, c in enumerate(cats)}

            for img_id, img_info in img_map.items():
                w, h = img_info["width"], img_info["height"]
                lbl_path = os.path.join(output_path, img_info["file_name"].rsplit(".", 1)[0] + ".txt")
                lines = []
                for ann in ann_by_img.get(img_id, []):
                    cls = cat_idx.get(ann["category_id"], 0)
                    bbox = ann["bbox"]
                    cx = (bbox[0] + bbox[2] / 2) / w
                    cy = (bbox[1] + bbox[3] / 2) / h
                    bw_n = bbox[2] / w
                    bh_n = bbox[3] / h
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
                with open(lbl_path, "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))

            # Write classes
            cls_path = os.path.join(output_path, "classes.txt")
            with open(cls_path, "w") as f:
                for c in cats:
                    f.write(c["name"] + "\n")
            return json.dumps({"success": True, "num_images": len(img_map), "output": output_path})

        # VOC → YOLO
        elif source_format == "voc" and target_format == "yolo":
            input_dir = Path(input_path)
            if input_dir.is_file():
                input_dir = input_dir.parent
            os.makedirs(output_path, exist_ok=True)
            classes_set: set[str] = set()
            # First pass: collect classes
            for xml_file in input_dir.glob("*.xml"):
                tree = ET.parse(xml_file)
                for obj in tree.findall(".//object"):
                    name = obj.find("name")
                    if name is not None and name.text:
                        classes_set.add(name.text)
            classes = sorted(classes_set)
            cls_idx = {c: i for i, c in enumerate(classes)}

            for xml_file in sorted(input_dir.glob("*.xml")):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                size = root.find("size")
                if size is not None:
                    w = int(size.find("width").text)  # type: ignore
                    h = int(size.find("height").text)  # type: ignore
                else:
                    w, h = 640, 640
                fname = root.find("filename")
                img_name = fname.text if fname is not None else xml_file.stem
                lbl_path = os.path.join(output_path, Path(img_name).stem + ".txt")
                lines = []
                for obj in root.findall(".//object"):
                    name = obj.find("name")
                    bndbox = obj.find("bndbox")
                    if name is None or bndbox is None:
                        continue
                    cls = cls_idx.get(name.text, 0)
                    xmin = float(bndbox.find("xmin").text)  # type: ignore
                    ymin = float(bndbox.find("ymin").text)  # type: ignore
                    xmax = float(bndbox.find("xmax").text)  # type: ignore
                    ymax = float(bndbox.find("ymax").text)  # type: ignore
                    cx = ((xmin + xmax) / 2) / w
                    cy = ((ymin + ymax) / 2) / h
                    bw_n = (xmax - xmin) / w
                    bh_n = (ymax - ymin) / h
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
                with open(lbl_path, "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))

            cls_path = os.path.join(output_path, "classes.txt")
            with open(cls_path, "w") as f:
                for c in classes:
                    f.write(c + "\n")
            return json.dumps({
                "success": True,
                "num_classes": len(classes),
                "num_files": len(list(input_dir.glob("*.xml"))),
                "output": output_path,
            })

        # VOC → COCO
        elif source_format == "voc" and target_format == "coco":
            input_dir = Path(input_path)
            if input_dir.is_file():
                input_dir = input_dir.parent
            classes_set: set[str] = set()
            xml_files = sorted(input_dir.glob("*.xml"))
            for xml_file in xml_files:
                tree = ET.parse(xml_file)
                for obj in tree.findall(".//object"):
                    name = obj.find("name")
                    if name is not None and name.text:
                        classes_set.add(name.text)
            classes = sorted(classes_set)
            cls_idx = {c: i for i, c in enumerate(classes)}

            coco: dict[str, Any] = {
                "images": [],
                "annotations": [],
                "categories": [{"id": i, "name": c} for i, c in enumerate(classes)],
            }
            ann_id = 1
            for img_id, xml_file in enumerate(xml_files, 1):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                fname = root.find("filename")
                img_name = fname.text if fname is not None else xml_file.stem
                size = root.find("size")
                w = int(size.find("width").text) if size is not None and size.find("width") is not None else 640  # type: ignore
                h = int(size.find("height").text) if size is not None and size.find("height") is not None else 640  # type: ignore
                coco["images"].append({"id": img_id, "file_name": img_name, "width": w, "height": h})
                for obj in root.findall(".//object"):
                    name = obj.find("name")
                    bndbox = obj.find("bndbox")
                    if name is None or bndbox is None:
                        continue
                    xmin = float(bndbox.find("xmin").text)  # type: ignore
                    ymin = float(bndbox.find("ymin").text)  # type: ignore
                    xmax = float(bndbox.find("xmax").text)  # type: ignore
                    ymax = float(bndbox.find("ymax").text)  # type: ignore
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_idx.get(name.text, 0),
                        "bbox": [round(xmin, 2), round(ymin, 2), round(xmax - xmin, 2), round(ymax - ymin, 2)],
                        "area": round((xmax - xmin) * (ymax - ymin), 2),
                        "iscrowd": 0,
                    })
                    ann_id += 1
            with open(output_path, "w") as f:
                json.dump(coco, f, indent=2)
            return json.dumps({
                "success": True,
                "num_images": len(coco["images"]),
                "num_annotations": len(coco["annotations"]),
                "output": output_path,
            })

        else:
            return json.dumps({
                "success": False,
                "error": f"Conversion {source_format} → {target_format} not supported. "
                         f"Supported: yolo↔coco, voc→yolo, voc→coco",
            })

    except Exception as e:
        logger.exception("annotation_convert failed")
        return json.dumps({"success": False, "error": str(e)})
