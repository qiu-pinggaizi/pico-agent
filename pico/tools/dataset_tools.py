"""Dataset tools — search, download, and manage datasets from online platforms.

Supports HuggingFace Hub, Kaggle, Roboflow, and general web search.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from pico.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _success(data: Any) -> str:
    return json.dumps({"success": True, **(data if isinstance(data, dict) else {"result": data})}, ensure_ascii=False)


def _error(msg: str) -> str:
    return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


def _get_download_dir() -> Path:
    """Get the default dataset download directory."""
    from pico.config import get_config
    cfg = get_config()
    base = Path(cfg.base_dir) if cfg.base_dir else Path.home() / ".pico-agent"
    dl_dir = base / "datasets"
    dl_dir.mkdir(parents=True, exist_ok=True)
    return dl_dir


def _search_huggingface(query: str, max_results: int) -> list[dict]:
    """Search HuggingFace datasets via their API."""
    import httpx
    try:
        resp = httpx.get(
            "https://huggingface.co/api/datasets",
            params={"search": query, "limit": max_results, "sort": "downloads", "direction": -1},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        datasets = resp.json()
        results = []
        for ds in datasets:
            results.append({
                "id": ds.get("id", ""),
                "source": "huggingface",
                "downloads": ds.get("downloads", 0),
                "likes": ds.get("likes", 0),
                "tags": ds.get("tags", [])[:5],
                "description": (ds.get("description") or "")[:200],
                "url": f"https://huggingface.co/datasets/{ds.get('id', '')}",
            })
        return results
    except Exception as e:
        logger.warning("HuggingFace search failed: %s", e)
        return []


def _search_kaggle(query: str, max_results: int) -> list[dict]:
    """Search Kaggle datasets via their API."""
    import httpx
    try:
        api_user = os.environ.get("KAGGLE_USERNAME")
        api_key = os.environ.get("KAGGLE_KEY")
        if not api_user or not api_key:
            return [{"note": "Kaggle search requires KAGGLE_USERNAME and KAGGLE_KEY env vars. Get them from kaggle.com → Settings → API"}]

        resp = httpx.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"search": query, "page": 1, "pageSize": max_results},
            auth=(api_user, api_key),
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        datasets = resp.json()
        results = []
        for ds in datasets:
            results.append({
                "ref": ds.get("ref", ""),
                "source": "kaggle",
                "title": ds.get("title", ""),
                "size": ds.get("totalBytes", 0),
                "downloadCount": ds.get("downloadCount", 0),
                "voteCount": ds.get("voteCount", 0),
                "tags": [t.get("name", "") for t in ds.get("tags", [])[:5]],
                "url": f"https://www.kaggle.com/datasets/{ds.get('ref', '')}",
            })
        return results
    except Exception as e:
        logger.warning("Kaggle search failed: %s", e)
        return []


def _search_roboflow_universe(query: str, max_results: int) -> list[dict]:
    """Search Roboflow Universe datasets via web scraping (API requires auth)."""
    import httpx
    try:
        # Roboflow Universe has a public search endpoint
        resp = httpx.get(  # noqa: F841
            "https://universe.roboflow.com/datasets/search",
            params={"q": query, "limit": max_results},
            headers={"User-Agent": "Mozilla/5.0 (compatible; PicoAgent/0.1)"},
            timeout=15,
            follow_redirects=True,
        )
        # Roboflow Universe doesn't have a clean public API for search
        # Fall back to returning search URL
        return [{
            "note": f"Browse Roboflow Universe results at: https://universe.roboflow.com/search?q={query.replace(' ', '+')}",
            "source": "roboflow",
            "hint": "Use web_extract on the URL above to browse results. To download, use the Roboflow API with a project ID.",
        }]
    except Exception as e:
        logger.warning("Roboflow search failed: %s", e)
        return []


def dataset_search(query: str, sources: str = "huggingface", max_results: int = 10) -> str:
    """Search for datasets across multiple platforms.

    Args:
        query: Search query (e.g., "cat dog detection", "coco", "traffic sign").
        sources: Comma-separated sources: "huggingface", "kaggle", "roboflow", "all".
        max_results: Max results per source (default 10).

    Returns:
        JSON with search results grouped by source.
    """
    try:
        source_list = [s.strip().lower() for s in sources.split(",")]
        all_results: dict[str, list] = {}

        searchers = {
            "huggingface": _search_huggingface,
            "kaggle": _search_kaggle,
            "roboflow": _search_roboflow_universe,
        }

        if "all" in source_list:
            source_list = list(searchers.keys())

        for source in source_list:
            if source in searchers:
                results = searchers[source](query, max_results)
                if results:
                    all_results[source] = results

        total = sum(len(v) for v in all_results.values())
        logger.info("dataset_search(%s): %d results from %s", query, total, list(all_results.keys()))
        return _success({"query": query, "sources": list(all_results.keys()), "total": total, "results": all_results})

    except Exception as e:
        logger.error("dataset_search failed: %s", e)
        return _error(f"Dataset search failed: {e}")


def dataset_download(
    source: str,
    name: str,
    output_dir: str = "",
    subset: str = "",
    split: str = "",
) -> str:
    """Download a dataset from a supported platform.

    Args:
        source: Platform — "huggingface", "kaggle", "roboflow", or "url".
        name: Dataset identifier:
              - huggingface: "username/dataset-name" (e.g., "imagenet-1k")
              - kaggle: "owner/dataset-slug" (e.g., "zillow/zecon")
              - roboflow: "workspace/project/version" (e.g., "roboflow-100/cells-uyemf/2")
              - url: Direct URL to a zip/tar file
        output_dir: Where to save (default: ~/.pico-agent/datasets/<name>).
        subset: HuggingFace subset/config name (optional).
        split: HuggingFace split to download (optional, e.g., "train", "test").

    Returns:
        JSON with download path and status.
    """
    try:
        if not output_dir:
            safe_name = name.replace("/", "_").replace("\\", "_")
            output_dir = str(_get_download_dir() / safe_name)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if source == "huggingface":
            return _download_huggingface(name, output_path, subset, split)
        elif source == "kaggle":
            return _download_kaggle(name, output_path)
        elif source == "roboflow":
            return _download_roboflow(name, output_path)
        elif source == "url":
            return _download_url(name, output_path)
        else:
            return _error(f"Unsupported source: {source}. Use: huggingface, kaggle, roboflow, url")

    except Exception as e:
        logger.error("dataset_download failed: %s", e)
        return _error(f"Download failed: {e}")


def _download_huggingface(dataset_id: str, output_path: Path, subset: str, split: str) -> str:
    """Download dataset from HuggingFace using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        return _error("huggingface_hub not installed. Run: pip install huggingface_hub")

    try:
        if split:
            # Download specific split
            kwargs = {"repo_id": dataset_id, "repo_type": "dataset", "local_dir": str(output_path)}
            if subset:
                kwargs["filename"] = f"data/{subset}/{split}.parquet" if not split.endswith(".parquet") else f"data/{subset}/{split}"
            else:
                kwargs["filename"] = f"data/{split}.parquet" if not split.endswith(".parquet") else f"data/{split}"
            path = hf_hub_download(**kwargs)
            return _success({"source": "huggingface", "dataset": dataset_id, "path": path, "split": split})
        else:
            # Download full dataset
            path = snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=str(output_path),
                allow_patterns=None,  # download everything
            )
            return _success({"source": "huggingface", "dataset": dataset_id, "path": path})

    except Exception as e:
        return _error(f"HuggingFace download failed: {e}")


def _download_kaggle(dataset_ref: str, output_path: Path) -> str:
    """Download dataset from Kaggle using kaggle CLI."""
    try:
        # Check kaggle CLI
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return _error("Kaggle CLI not installed or not configured. Run: pip install kaggle, and set KAGGLE_USERNAME/KAGGLE_KEY env vars.")
    except FileNotFoundError:
        return _error("Kaggle CLI not found. Run: pip install kaggle")

    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(output_path), "--unzip"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return _error(f"Kaggle download failed: {result.stderr.strip()}")

        # List downloaded files
        files = [str(f.relative_to(output_path)) for f in output_path.rglob("*") if f.is_file()][:20]

        return _success({
            "source": "kaggle",
            "dataset": dataset_ref,
            "path": str(output_path),
            "files": files,
            "file_count": len(list(output_path.rglob("*"))),
        })

    except subprocess.TimeoutExpired:
        return _error("Kaggle download timed out (>600s)")
    except Exception as e:
        return _error(f"Kaggle download failed: {e}")


def _download_roboflow(project_ref: str, output_path: Path) -> str:
    """Download dataset from Roboflow Universe using roboflow SDK or API."""
    try:
        # Try roboflow SDK first
        from roboflow import Roboflow
        rf_key = os.environ.get("ROBOFLOW_API_KEY")
        if not rf_key:
            return _error("ROBOFLOW_API_KEY not set. Get it from app.roboflow.com → Settings → API")

        # Parse workspace/project/version
        parts = project_ref.split("/")
        if len(parts) < 2:
            return _error(f"Invalid Roboflow ref: {project_ref}. Use format: workspace/project/version")

        workspace = parts[0]
        project = parts[1]
        version = int(parts[2]) if len(parts) > 2 else None

        rf = Roboflow(api_key=rf_key)
        ws = rf.workspace(workspace)
        proj = ws.project(project)
        _dataset = proj.version(version or 1).download("yolov8", location=str(output_path))

        return _success({
            "source": "roboflow",
            "dataset": project_ref,
            "path": str(output_path),
        })

    except ImportError:
        return _error("roboflow SDK not installed. Run: pip install roboflow")
    except Exception as e:
        return _error(f"Roboflow download failed: {e}")


def _download_url(url: str, output_path: Path) -> str:
    """Download a dataset from a direct URL (zip/tar/gz)."""
    import httpx

    try:
        # Determine filename from URL
        filename = url.split("/")[-1].split("?")[0] or "download"
        filepath = output_path / filename

        # Stream download
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        file_size = filepath.stat().st_size

        # Auto-extract archives
        extracted = False
        if filename.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(output_path)
            extracted = True
        elif filename.endswith((".tar.gz", ".tgz", ".tar")):
            import tarfile
            with tarfile.open(filepath, "r:*") as tf:
                tf.extractall(output_path)
            extracted = True

        result = {
            "source": "url",
            "url": url,
            "path": str(output_path),
            "filename": filename,
            "size_bytes": file_size,
            "extracted": extracted,
        }

        if extracted:
            files = [str(f.relative_to(output_path)) for f in output_path.rglob("*") if f.is_file() and f.name != filename][:20]
            result["files"] = files

        return _success(result)

    except Exception as e:
        return _error(f"URL download failed: {e}")


def dataset_info(path: str) -> str:
    """Show information about a local dataset directory.

    Args:
        path: Path to the dataset directory.

    Returns:
        JSON with file count, total size, file types, and directory structure.
    """
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return _error(f"Path not found: {path}")
        if not p.is_dir():
            return _error(f"Not a directory: {path}")

        total_size = 0
        file_count = 0
        dir_count = 0
        extensions: dict[str, int] = {}
        sample_files: list[str] = []

        for item in p.rglob("*"):
            if item.is_file():
                file_count += 1
                size = item.stat().st_size
                total_size += size
                ext = item.suffix.lower() or "(no ext)"
                extensions[ext] = extensions.get(ext, 0) + 1
                if len(sample_files) < 30:
                    sample_files.append(str(item.relative_to(p)))
            elif item.is_dir():
                dir_count += 1

        # Format size
        if total_size > 1_073_741_824:
            size_str = f"{total_size / 1_073_741_824:.1f} GB"
        elif total_size > 1_048_576:
            size_str = f"{total_size / 1_048_576:.1f} MB"
        elif total_size > 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size} B"

        return _success({
            "path": str(p),
            "file_count": file_count,
            "dir_count": dir_count,
            "total_size": size_str,
            "total_size_bytes": total_size,
            "extensions": dict(sorted(extensions.items(), key=lambda x: -x[1])[:15]),
            "sample_files": sample_files,
        })

    except Exception as e:
        return _error(f"Failed to read dataset info: {e}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry: ToolRegistry) -> None:
    """Register dataset tools."""
    registry.register(
        name="dataset_search",
        toolset="web",
        description=(
            "Search for datasets on HuggingFace, Kaggle, Roboflow. "
            "Returns dataset listings with download counts, tags, and URLs. "
            "Use this BEFORE downloading to find the right dataset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'cat detection', 'COCO', 'traffic sign'."},
                "sources": {"type": "string", "description": "Comma-separated sources: 'huggingface', 'kaggle', 'roboflow', 'all'. Default 'huggingface'.", "default": "huggingface"},
                "max_results": {"type": "integer", "description": "Max results per source (default 10).", "default": 10},
            },
            "required": ["query"],
        },
        handler=dataset_search,
    )

    registry.register(
        name="dataset_download",
        toolset="web",
        description=(
            "Download a dataset from HuggingFace, Kaggle, Roboflow, or a direct URL. "
            "For HuggingFace: name='username/dataset-id'. "
            "For Kaggle: name='owner/slug' (needs KAGGLE_USERNAME + KAGGLE_KEY env vars). "
            "For URL: provide direct download link to zip/tar file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Platform: 'huggingface', 'kaggle', 'roboflow', 'url'."},
                "name": {"type": "string", "description": "Dataset identifier or URL."},
                "output_dir": {"type": "string", "description": "Save location (default: ~/.pico-agent/datasets/<name>).", "default": ""},
                "subset": {"type": "string", "description": "HuggingFace subset/config name (optional).", "default": ""},
                "split": {"type": "string", "description": "HuggingFace split (optional, e.g. 'train', 'test').", "default": ""},
            },
            "required": ["source", "name"],
        },
        handler=dataset_download,
    )

    registry.register(
        name="dataset_info",
        toolset="web",
        description=(
            "Show info about a local dataset directory: file count, total size, file types, and sample files. "
            "Use after downloading to verify the dataset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the dataset directory."},
            },
            "required": ["path"],
        },
        handler=dataset_info,
    )
