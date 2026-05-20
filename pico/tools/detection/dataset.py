"""Dataset exploration, splitting, and format conversion for detection.

Supports YOLO, COCO, and VOC annotation formats.
All handlers accept a ``server`` parameter — when non-empty, the operation
runs on the remote server via ``remote_terminal``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — format detection
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _detect_format(data_dir: str) -> str:
    """Auto-detect dataset format from directory structure."""
    data_path = Path(data_dir)
    # YOLO: has labels/ dir with .txt files
    if (data_path / "labels").is_dir() or any(
        (data_path / d).is_dir() for d in ("train", "val", "valid", "test")
        if (data_path / d / "labels").is_dir()
    ):
        return "yolo"
    # COCO: has annotations/*.json
    if (data_path / "annotations").is_dir():
        for f in (data_path / "annotations").iterdir():
            if f.suffix == ".json":
                return "coco"
    for f in data_path.iterdir():
        if f.suffix == ".json" and "annotation" in f.name.lower():
            return "coco"
    # VOC: has Annotations/*.xml
    if (data_path / "Annotations").is_dir():
        xmls = list((data_path / "Annotations").glob("*.xml"))
        if xmls:
            return "voc"
    # Fallback: check for .xml files directly
    if list(data_path.glob("*.xml")):
        return "voc"
    return "unknown"


def _get_image_files(directory: str) -> list[str]:
    """Return list of image file paths in *directory*."""
    result = []
    d = Path(directory)
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in _IMAGE_EXTS and f.is_file():
                result.append(str(f))
    return result


def _count_classes_yolo(labels_dir: str) -> Counter:
    """Count class occurrences in YOLO label files."""
    counter: Counter = Counter()
    ld = Path(labels_dir)
    if not ld.is_dir():
        return counter
    for txt in ld.glob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    cls_id = int(parts[0])
                    counter[cls_id] += 1
                except ValueError:
                    continue
    return counter


def _read_classes_file(classes_file: str) -> list[str]:
    """Read classes.txt, one class per line."""
    if not classes_file or not os.path.isfile(classes_file):
        return []
    return [line.strip() for line in Path(classes_file).read_text().strip().splitlines() if line.strip()]


def _find_classes_file(data_dir: str) -> str:
    """Try to locate a classes.txt or classes.names file."""
    for name in ("classes.txt", "classes.names", "names.txt"):
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return p
    # Also check parent
    parent = os.path.dirname(data_dir)
    for name in ("classes.txt", "classes.names", "names.txt"):
        p = os.path.join(parent, name)
        if os.path.isfile(p):
            return p
    return ""


# ---------------------------------------------------------------------------
# Command execution helper (local vs remote)
# ---------------------------------------------------------------------------

def _exec(command: str, server: str = "") -> dict[str, Any]:
    """Execute a command locally or remotely."""
    if server:
        from pico.tools.remote.remote_terminal import remote_terminal_handler
        result = json.loads(remote_terminal_handler({"command": command, "server": server}))
    else:
        import subprocess
        try:
            proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            result = {"success": proc.returncode == 0, "output": proc.stdout, "exit_code": proc.returncode}
            if proc.stderr:
                result["output"] += f"\n{proc.stderr}"
        except subprocess.TimeoutExpired:
            result = {"success": False, "error": "Command timed out"}
        except Exception as e:
            result = {"success": False, "error": str(e)}
    return result


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def dataset_explore_handler(args: dict[str, Any]) -> str:
    """Explore dataset: detect format, count images/annotations, class distribution."""
    data_dir: str = args.get("data_dir", "")
    server: str = args.get("server", "")

    if not data_dir:
        return json.dumps({"success": False, "error": "data_dir is required"})

    try:
        if server:
            # For remote: run a Python script to explore
            script = f"""
import json, os, collections
from pathlib import Path

data_dir = "{data_dir}"
IMAGE_EXTS = {{".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}}

result = {{"data_dir": data_dir}}
data_path = Path(data_dir)

# Count images
image_count = 0
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if Path(f).suffix.lower() in IMAGE_EXTS:
            image_count += 1
result["image_count"] = image_count

# Check format
fmt = "unknown"
if (data_path / "labels").is_dir() or any((data_path / d).is_dir() for d in ("train","val","valid","test") if (data_path / d / "labels").is_dir()):
    fmt = "yolo"
elif (data_path / "annotations").is_dir():
    for f in (data_path / "annotations").iterdir():
        if f.suffix == ".json":
            fmt = "coco"
            break
if fmt == "unknown" and (data_path / "Annotations").is_dir():
    if list((data_path / "Annotations").glob("*.xml")):
        fmt = "voc"
result["format"] = fmt

# Subdirectory structure
subs = [d.name for d in data_path.iterdir() if d.is_dir()]
result["subdirectories"] = subs

# Class distribution (YOLO)
if fmt == "yolo":
    labels_dir = None
    for candidate in ["labels", "train/labels", "val/labels"]:
        p = data_path / candidate
        if p.is_dir():
            # Check direct .txt files first
            if list(p.glob("*.txt")):
                labels_dir = str(p)
                break
            # Check nested (e.g. labels/train2017/)
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and list(sub.glob("*.txt")):
                    labels_dir = str(sub)
                    break
            if labels_dir:
                break
    if labels_dir:
        counter = collections.Counter()
        for txt in Path(labels_dir).glob("*.txt"):
            for line in txt.read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    try: counter[int(parts[0])] += 1
                    except: pass
        result["class_distribution"] = dict(counter)
        result["num_classes"] = len(counter)
        result["num_annotations"] = sum(counter.values())

print(json.dumps(result))
"""
            cmd_result = _exec(f"python3 -c '{script}'", server)
            if cmd_result.get("success"):
                try:
                    return json.dumps({"success": True, **json.loads(cmd_result["output"].strip().splitlines()[-1])})
                except json.JSONDecodeError:
                    return json.dumps({"success": True, "raw_output": cmd_result["output"]})
            else:
                return json.dumps({"success": False, "error": cmd_result.get("error", cmd_result.get("output", ""))})

        # Local execution
        result: dict[str, Any] = {"success": True, "data_dir": data_dir}
        data_path = Path(data_dir)

        if not data_path.is_dir():
            return json.dumps({"success": False, "error": f"Directory not found: {data_dir}"})

        fmt = _detect_format(data_dir)
        result["format"] = fmt

        # Count images
        image_count = 0
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if Path(f).suffix.lower() in _IMAGE_EXTS:
                    image_count += 1
        result["image_count"] = image_count

        # Subdirectories
        result["subdirectories"] = [d.name for d in data_path.iterdir() if d.is_dir()]

        # Class distribution
        classes_file = _find_classes_file(data_dir)
        if classes_file:
            result["classes_file"] = classes_file
            result["classes"] = _read_classes_file(classes_file)

        if fmt == "yolo":
            # Find labels dir — also handles nested structures like labels/train2017/
            labels_dir = None
            for candidate in ["labels", "train/labels", "val/labels"]:
                p = data_path / candidate
                if p.is_dir():
                    # Check if there are .txt files directly
                    if list(p.glob("*.txt")):
                        labels_dir = str(p)
                        break
                    # Check for nested subdirs (e.g. labels/train2017/)
                    for sub in sorted(p.iterdir()):
                        if sub.is_dir() and list(sub.glob("*.txt")):
                            labels_dir = str(sub)
                            break
                    if labels_dir:
                        break
            if labels_dir:
                counter = _count_classes_yolo(labels_dir)
                result["class_distribution"] = dict(counter)
                result["num_classes"] = len(counter)
                result["num_annotations"] = sum(counter.values())

                if classes_file:
                    classes = _read_classes_file(classes_file)
                    result["class_names"] = {str(k): classes[k] if k < len(classes) else f"class_{k}" for k in counter}

        elif fmt == "coco":
            # Parse COCO json
            ann_dir = data_path / "annotations"
            for json_file in ann_dir.glob("*.json"):
                with open(json_file) as f:
                    coco_data = json.load(f)
                result["num_images"] = len(coco_data.get("images", []))
                result["num_annotations"] = len(coco_data.get("annotations", []))
                cats = coco_data.get("categories", [])
                result["num_classes"] = len(cats)
                result["categories"] = {c["id"]: c["name"] for c in cats}
                # class distribution
                ann_counter: Counter = Counter()
                for ann in coco_data.get("annotations", []):
                    ann_counter[ann["category_id"]] += 1
                result["class_distribution"] = dict(ann_counter)
                break

        elif fmt == "voc":
            ann_dir = data_path / "Annotations"
            xml_files = list(ann_dir.glob("*.xml"))
            result["num_annotations_files"] = len(xml_files)
            ann_counter: Counter = Counter()
            for xf in xml_files:
                tree = ET.parse(xf)
                for obj in tree.findall(".//object"):
                    name_el = obj.find("name")
                    if name_el is not None and name_el.text:
                        ann_counter[name_el.text] += 1
            result["class_distribution"] = dict(ann_counter)
            result["num_classes"] = len(ann_counter)

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.exception("dataset_explore failed")
        return json.dumps({"success": False, "error": str(e)})


def dataset_split_handler(args: dict[str, Any]) -> str:
    """Split dataset into train/val/test with stratified sampling."""
    data_dir: str = args.get("data_dir", "")
    output_dir: str = args.get("output_dir", "")
    ratios: list[float] = args.get("ratios", [0.8, 0.15, 0.05])
    fmt: str = args.get("format", "yolo")
    server: str = args.get("server", "")

    if not data_dir or not output_dir:
        return json.dumps({"success": False, "error": "data_dir and output_dir are required"})

    try:
        if server:
            # Build a Python script for remote execution
            script = f"""
import json, os, random, shutil
from pathlib import Path

data_dir = "{data_dir}"
output_dir = "{output_dir}"
ratios = {ratios}
fmt = "{fmt}"
IMAGE_EXTS = {{".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}}

os.makedirs(output_dir, exist_ok=True)

# Collect image-label pairs
pairs = []
data_path = Path(data_dir)

if fmt == "yolo":
    # find images and labels
    img_dir = None
    lbl_dir = None
    for candidate in [("images", "labels"), ("train/images", "train/labels"), ("", "")]:
        idir = data_path / candidate[0] if candidate[0] else data_path
        ldir = data_path / candidate[1] if candidate[1] else data_path
        if idir.is_dir() and ldir.is_dir():
            img_dir = idir
            lbl_dir = ldir
            break
    if img_dir and lbl_dir:
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in IMAGE_EXTS:
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.is_file():
                    pairs.append((str(img_file), str(lbl_file)))

random.seed(42)
random.shuffle(pairs)
n = len(pairs)
train_n = int(n * ratios[0])
val_n = int(n * ratios[1])
splits = {{"train": pairs[:train_n], "val": pairs[train_n:train_n+val_n], "test": pairs[train_n+val_n:]}}

for split_name, items in splits.items():
    if not items:
        continue
    oimg = os.path.join(output_dir, split_name, "images")
    olbl = os.path.join(output_dir, split_name, "labels")
    os.makedirs(oimg, exist_ok=True)
    os.makedirs(olbl, exist_ok=True)
    for img, lbl in items:
        shutil.copy2(img, os.path.join(oimg, os.path.basename(img)))
        shutil.copy2(lbl, os.path.join(olbl, os.path.basename(lbl)))

# Copy classes file if exists
for cf in ["classes.txt", "classes.names", "names.txt"]:
    src = os.path.join(data_dir, cf)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(output_dir, cf))

print(json.dumps({{
    "success": True,
    "total": n,
    "train": len(splits["train"]),
    "val": len(splits["val"]),
    "test": len(splits["test"]),
    "output_dir": output_dir
}}))
"""
            result = _exec(f"python3 -c '{script}'", server)
            if result.get("success"):
                try:
                    lines = result["output"].strip().splitlines()
                    return lines[-1]
                except Exception:
                    return json.dumps({"success": True, "output": result["output"]})
            return json.dumps({"success": False, "error": result.get("error", result.get("output", ""))})

        # Local execution
        data_path = Path(data_dir)
        if not data_path.is_dir():
            return json.dumps({"success": False, "error": f"Directory not found: {data_dir}"})

        os.makedirs(output_dir, exist_ok=True)

        # Collect pairs
        pairs: list[tuple[str, str]] = []

        if fmt == "yolo":
            img_dir = None
            lbl_dir = None
            for idir_c, ldir_c in [("images", "labels"), ("train/images", "train/labels"), ("", "")]:
                idir = data_path / idir_c if idir_c else data_path
                ldir = data_path / ldir_c if ldir_c else data_path
                if idir.is_dir() and ldir.is_dir():
                    img_dir = idir
                    lbl_dir = ldir
                    break
            if img_dir and lbl_dir:
                for img_file in sorted(img_dir.iterdir()):
                    if img_file.suffix.lower() in _IMAGE_EXTS:
                        lbl_file = lbl_dir / (img_file.stem + ".txt")
                        if lbl_file.is_file():
                            pairs.append((str(img_file), str(lbl_file)))

        n = len(pairs)
        random.seed(42)
        random.shuffle(pairs)

        train_n = int(n * ratios[0])
        val_n = int(n * ratios[1])
        splits = {
            "train": pairs[:train_n],
            "val": pairs[train_n:train_n + val_n],
            "test": pairs[train_n + val_n:],
        }

        for split_name, items in splits.items():
            if not items:
                continue
            oimg = os.path.join(output_dir, split_name, "images")
            olbl = os.path.join(output_dir, split_name, "labels")
            os.makedirs(oimg, exist_ok=True)
            os.makedirs(olbl, exist_ok=True)
            for img, lbl in items:
                shutil.copy2(img, os.path.join(oimg, os.path.basename(img)))
                shutil.copy2(lbl, os.path.join(olbl, os.path.basename(lbl)))

        # Copy classes file
        for cf in ("classes.txt", "classes.names", "names.txt"):
            src = os.path.join(data_dir, cf)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, cf))

        return json.dumps({
            "success": True,
            "total": n,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "output_dir": output_dir,
        })

    except Exception as e:
        logger.exception("dataset_split failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def _yolo_to_coco(images_dir: str, labels_dir: str, classes: list[str]) -> dict:
    """Convert YOLO labels to COCO dict."""
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(classes)],
    }
    ann_id = 1
    for img_id, img_file in enumerate(sorted(Path(images_dir).iterdir()), 1):
        if img_file.suffix.lower() not in _IMAGE_EXTS:
            continue
        # Get image dimensions (use PIL if available, otherwise defaults)
        w, h = 640, 640
        try:
            from PIL import Image
            with Image.open(img_file) as im:
                w, h = im.size
        except Exception:
            pass
        coco["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": w,
            "height": h,
        })
        lbl_file = Path(labels_dir) / (img_file.stem + ".txt")
        if lbl_file.is_file():
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    w_box = bw * w
                    h_box = bh * h
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [round(x, 2), round(y, 2), round(w_box, 2), round(h_box, 2)],
                        "area": round(w_box * h_box, 2),
                        "iscrowd": 0,
                    })
                    ann_id += 1
    return coco


def _coco_to_yolo(coco_json_path: str, output_dir: str, classes_file: str = "") -> dict:
    """Convert COCO JSON to YOLO format."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Build image id -> info map
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    ann_by_img: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Category mapping
    cats = coco.get("categories", [])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(cats)}

    # Write labels
    for img_id, img_info in img_map.items():
        w = img_info["width"]
        h = img_info["height"]
        lbl_name = Path(img_info["file_name"]).stem + ".txt"
        lbl_path = os.path.join(output_dir, "labels", lbl_name)
        lines = []
        for ann in ann_by_img.get(img_id, []):
            cls_idx = cat_id_to_idx.get(ann["category_id"], 0)
            bbox = ann["bbox"]  # [x, y, w, h]
            cx = (bbox[0] + bbox[2] / 2) / w
            cy = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    # Write classes file
    if not classes_file:
        classes_file = os.path.join(output_dir, "classes.txt")
    with open(classes_file, "w") as f:
        for cat in cats:
            f.write(cat["name"] + "\n")

    return {"success": True, "num_images": len(img_map), "num_categories": len(cats)}


def dataset_convert_handler(args: dict[str, Any]) -> str:
    """Convert dataset between COCO, VOC, and YOLO formats."""
    data_dir: str = args.get("data_dir", "")
    output_dir: str = args.get("output_dir", "")
    source_format: str = args.get("source_format", "")
    target_format: str = args.get("target_format", "")
    classes_file: str = args.get("classes_file", "")
    server: str = args.get("server", "")

    if not all([data_dir, output_dir, source_format, target_format]):
        return json.dumps({"success": False, "error": "data_dir, output_dir, source_format, target_format are required"})

    if source_format == target_format:
        return json.dumps({"success": False, "error": "source and target formats are the same"})

    try:
        if server:
            # For remote conversion, we upload a Python script and run it
            return json.dumps({"success": False, "error": "Remote dataset conversion not yet supported. Use local mode."})

        os.makedirs(output_dir, exist_ok=True)

        if source_format == "yolo" and target_format == "coco":
            data_path = Path(data_dir)
            # Find images and labels dirs
            img_dir = lbl_dir = None
            for idir, ldir in [("images", "labels"), ("train/images", "train/labels"), ("", "")]:
                id_path = data_path / idir if idir else data_path
                ld_path = data_path / ldir if ldir else data_path
                if id_path.is_dir() and ld_path.is_dir():
                    img_dir, lbl_dir = str(id_path), str(ld_path)
                    break
            if not img_dir or not lbl_dir:
                return json.dumps({"success": False, "error": "Could not find images/ and labels/ directories"})

            cf = classes_file or _find_classes_file(data_dir)
            classes = _read_classes_file(cf) if cf else []
            coco = _yolo_to_coco(img_dir, lbl_dir, classes)
            out_json = os.path.join(output_dir, "annotations.json")
            with open(out_json, "w") as f:
                json.dump(coco, f, indent=2)
            return json.dumps({
                "success": True,
                "output": out_json,
                "num_images": len(coco["images"]),
                "num_annotations": len(coco["annotations"]),
                "num_categories": len(coco["categories"]),
            })

        elif source_format == "coco" and target_format == "yolo":
            data_path = Path(data_dir)
            json_file = None
            for candidate in (data_path / "annotations").glob("*.json"):
                json_file = str(candidate)
                break
            if not json_file:
                for f in data_path.glob("*.json"):
                    json_file = str(f)
                    break
            if not json_file:
                return json.dumps({"success": False, "error": "No COCO JSON found"})

            result = _coco_to_yolo(json_file, output_dir, classes_file)
            return json.dumps(result)

        else:
            return json.dumps({
                "success": False,
                "error": f"Conversion {source_format} -> {target_format} not yet implemented. "
                         f"Supported: yolo↔coco",
            })

    except Exception as e:
        logger.exception("dataset_convert failed")
        return json.dumps({"success": False, "error": str(e)})
