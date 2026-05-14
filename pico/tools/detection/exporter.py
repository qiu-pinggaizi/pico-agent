"""Export tools for object detection models (YOLO / MMDetection).

Provides handlers for ONNX export, TensorRT export, and speed benchmarks.
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


# ---------------------------------------------------------------------------
# handler: export_onnx
# ---------------------------------------------------------------------------

def export_onnx_handler(args: dict[str, Any]) -> str:
    """Export a model to ONNX format.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``output_path`` (str, default ''),
        ``opset`` (int, default 17), ``dynamic`` (bool, default True),
        ``simplify`` (bool, default True),
        ``framework`` (str, default 'yolo'), ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    output_path: str = args.get("output_path", "")
    opset: int = int(args.get("opset", 17))
    dynamic: bool = bool(args.get("dynamic", True))
    simplify: bool = bool(args.get("simplify", True))
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path:
        return json.dumps({"success": False, "error": "model_path is required"})

    try:
        if framework == "yolo":
            script = (
                "import json, os\n"
                "from ultralytics import YOLO\n"
                f"model = YOLO('{model_path}')\n"
                f"onnx_path = model.export(format='onnx', opset={opset}, dynamic={dynamic}, simplify={simplify})\n"
                "onnx_path = str(onnx_path)\n"
                "file_size = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0\n"
                f"print(json.dumps({{'success': True, 'onnx_path': onnx_path, 'file_size_mb': round(file_size/1024/1024, 2), 'opset': {opset}, 'dynamic': {dynamic}, 'simplify': {simplify}}}))\n"
            )
        elif framework == "mmdet":
            config_path: str = args.get("config_path", "")
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet ONNX export"})

            out = output_path or model_path.replace(".pth", ".onnx").replace(".pt", ".onnx")
            imgsz: int = int(args.get("imgsz", 640))
            script = (
                "import json, os, torch\n"
                "from mmdet.apis import init_detector\n"
                f"model = init_detector('{config_path}', '{model_path}', device='cpu')\n"
                "model.eval()\n"
                f"dummy = torch.randn(1, 3, {imgsz}, {imgsz})\n"
                f"torch.onnx.export(model, dummy, '{out}', opset_version={opset}, dynamic_axes={{'input': {{0: 'batch'}}, 'output': {{0: 'batch'}}}})\n"
                "file_size = os.path.getsize('{out}') if os.path.exists('{out}') else 0\n"
                f"print(json.dumps({{'success': True, 'onnx_path': '{out}', 'file_size_mb': round(file_size/1024/1024, 2)}}))\n"
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
                remote_path = f"/tmp/_pico_onnx_{os.path.basename(tmp_path)}"
                upload_res = json.loads(file_upload_handler({
                    "local_path": tmp_path, "remote_path": remote_path, "server": server,
                }))
                if not upload_res.get("success"):
                    return json.dumps({"success": False, "error": f"Upload failed: {upload_res.get('error', '')}"})
                result = _exec_cmd(f"python3 {remote_path}", server=server, timeout=300)
            else:
                result = _exec_cmd(f"python3 {tmp_path}", server="", timeout=300)

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
                return json.dumps({"success": True, "output": output[-1000:]})
            else:
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "ONNX export failed"))})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.exception("export_onnx failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: export_tensorrt
# ---------------------------------------------------------------------------

def export_tensorrt_handler(args: dict[str, Any]) -> str:
    """Export a model to TensorRT engine format.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``output_path`` (str, default ''),
        ``half`` (bool, default True), ``workspace`` (int, default 4),
        ``framework`` (str, default 'yolo'), ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    output_path: str = args.get("output_path", "")
    half: bool = bool(args.get("half", True))
    workspace: int = int(args.get("workspace", 4))
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")
    imgsz: int = int(args.get("imgsz", 640))

    if not model_path:
        return json.dumps({"success": False, "error": "model_path is required"})

    try:
        if framework == "yolo":
            script = (
                "import json, os\n"
                "from ultralytics import YOLO\n"
                f"model = YOLO('{model_path}')\n"
                f"engine_path = model.export(format='engine', half={half}, workspace={workspace}, imgsz={imgsz})\n"
                "engine_path = str(engine_path)\n"
                "file_size = os.path.getsize(engine_path) if os.path.exists(engine_path) else 0\n"
                f"print(json.dumps({{'success': True, 'engine_path': engine_path, 'file_size_mb': round(file_size/1024/1024, 2), 'half_precision': {half}, 'workspace_gb': {workspace}}}))\n"
            )
        elif framework == "mmdet":
            config_path: str = args.get("config_path", "")
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet TensorRT export"})

            out = output_path or model_path.replace(".pth", ".engine").replace(".pt", ".engine")
            script = (
                "import json, os, subprocess\n"
                "# MMDet TensorRT export via mmdeploy\n"
                f"cmd = ('python -m mmdeploy.apis.openvino.onnx2trt '\n"
                f"       '{model_path.replace('.pth','.onnx').replace('.pt','.onnx')}' \n"
                f"       '{out}' --fp16' if {half} else '--fp32')\n"
                "result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)\n"
                "file_size = os.path.getsize('{out}') if os.path.exists('{out}') else 0\n"
                f"print(json.dumps({{'success': result.returncode == 0, 'engine_path': '{out}', 'file_size_mb': round(file_size/1024/1024, 2), 'output': result.stdout + result.stderr}}))\n"
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
                remote_path = f"/tmp/_pico_trt_{os.path.basename(tmp_path)}"
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
                lines = output.splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "success" in parsed:
                            return json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        continue
                return json.dumps({"success": True, "output": output[-1000:]})
            else:
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "TensorRT export failed"))})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.exception("export_tensorrt failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# handler: benchmark
# ---------------------------------------------------------------------------

def benchmark_handler(args: dict[str, Any]) -> str:
    """Run speed benchmark on a model.

    Measures inference latency across different batch sizes.

    Parameters
    ----------
    args : dict
        ``model_path`` (str), ``image_size`` (int, default 640),
        ``batch_sizes`` (list[int], default [1, 4, 8, 16]),
        ``num_runs`` (int, default 100),
        ``framework`` (str, default 'yolo'), ``server`` (str).
    """
    model_path: str = args.get("model_path", "")
    image_size: int = int(args.get("image_size", 640))
    batch_sizes: list[int] = args.get("batch_sizes", [1, 4, 8, 16])
    num_runs: int = int(args.get("num_runs", 100))
    framework: str = args.get("framework", "yolo")
    server: str = args.get("server", "")

    if not model_path:
        return json.dumps({"success": False, "error": "model_path is required"})

    try:
        if framework == "yolo":
            script = (
                "import json, time\n"
                "from ultralytics import YOLO\n"
                "import torch\n"
                f"model = YOLO('{model_path}')\n"
                f"imgsz = {image_size}\n"
                f"num_runs = {num_runs}\n"
                f"batch_sizes = {batch_sizes}\n"
                "\n"
                "results = []\n"
                "for bs in batch_sizes:\n"
                "    # Warmup\n"
                "    for _ in range(5):\n"
                "        dummy = torch.randn(bs, 3, imgsz, imgsz).to(model.device if hasattr(model, 'device') else 'cpu')\n"
                "        model(dummy, verbose=False)\n"
                "    # Benchmark\n"
                "    torch.cuda.synchronize() if torch.cuda.is_available() else None\n"
                "    start = time.perf_counter()\n"
                "    for _ in range(num_runs):\n"
                "        dummy = torch.randn(bs, 3, imgsz, imgsz).to(model.device if hasattr(model, 'device') else 'cpu')\n"
                "        model(dummy, verbose=False)\n"
                "    torch.cuda.synchronize() if torch.cuda.is_available() else None\n"
                "    elapsed = time.perf_counter() - start\n"
                "    avg_ms = (elapsed / num_runs) * 1000\n"
                "    throughput = (bs * num_runs) / elapsed\n"
                "    results.append({\n"
                "        'batch_size': bs,\n"
                "        'avg_latency_ms': round(avg_ms, 2),\n"
                "        'throughput_fps': round(throughput, 1),\n"
                "        'per_image_ms': round(avg_ms / bs, 2),\n"
                "    })\n"
                "\n"
                "# GPU info\n"
                "gpu_name = 'N/A'\n"
                "gpu_mem = 0\n"
                "if torch.cuda.is_available():\n"
                "    gpu_name = torch.cuda.get_device_name(0)\n"
                "    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3\n"
                "\n"
                f"print(json.dumps({{'success': True, 'model': '{model_path}', 'image_size': imgsz, 'num_runs': num_runs, 'gpu': gpu_name, 'gpu_memory_gb': round(gpu_mem, 1), 'results': results}}))\n"
            )
        elif framework == "mmdet":
            config_path: str = args.get("config_path", "")
            if not config_path:
                return json.dumps({"success": False, "error": "config_path is required for MMDet benchmark"})

            script = (
                "import json, time\n"
                "from mmdet.apis import init_detector\n"
                "import torch\n"
                f"model = init_detector('{config_path}', '{model_path}', device='cuda:0')\n"
                "model.eval()\n"
                f"imgsz = {image_size}\n"
                f"num_runs = {num_runs}\n"
                f"batch_sizes = {batch_sizes}\n"
                "\n"
                "results = []\n"
                "for bs in batch_sizes:\n"
                "    dummy = torch.randn(bs, 3, imgsz, imgsz).cuda()\n"
                "    # Warmup\n"
                "    with torch.no_grad():\n"
                "        for _ in range(5):\n"
                "            model(dummy)\n"
                "    torch.cuda.synchronize()\n"
                "    start = time.perf_counter()\n"
                "    with torch.no_grad():\n"
                "        for _ in range(num_runs):\n"
                "            model(dummy)\n"
                "    torch.cuda.synchronize()\n"
                "    elapsed = time.perf_counter() - start\n"
                "    avg_ms = (elapsed / num_runs) * 1000\n"
                "    throughput = (bs * num_runs) / elapsed\n"
                "    results.append({\n"
                "        'batch_size': bs,\n"
                "        'avg_latency_ms': round(avg_ms, 2),\n"
                "        'throughput_fps': round(throughput, 1),\n"
                "        'per_image_ms': round(avg_ms / bs, 2),\n"
                "    })\n"
                "\n"
                "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'\n"
                "gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3 if torch.cuda.is_available() else 0\n"
                f"print(json.dumps({{'success': True, 'model': '{model_path}', 'image_size': imgsz, 'num_runs': num_runs, 'gpu': gpu_name, 'gpu_memory_gb': round(gpu_mem, 1), 'results': results}}))\n"
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
                remote_path = f"/tmp/_pico_bench_{os.path.basename(tmp_path)}"
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
                return json.dumps({"success": False, "error": result.get("error", result.get("output", "Benchmark failed"))})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.exception("benchmark failed")
        return json.dumps({"success": False, "error": str(e)})
