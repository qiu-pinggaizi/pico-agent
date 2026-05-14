# Pico Agent — 远程服务器架构设计

## 一、两种运行模式

```
模式 A: 本地 Agent + 远程执行（推荐日常使用）
┌─────────────────────┐         SSH          ┌─────────────────────┐
│  本地机器 (你的电脑)  │ ────────────────────▶ │  远程 GPU 服务器      │
│                     │                       │                     │
│  ┌───────────────┐  │   terminal_ssh.py     │  ┌───────────────┐  │
│  │  Pico Agent   │  │ ── SSH 连接 ────────▶ │  │  训练脚本      │  │
│  │  (CLI 交互)   │  │   执行命令/传文件     │  │  ultralytics   │  │
│  │  LLM 调用     │  │                       │  │  mmdet         │  │
│  │  决策/对话    │  │   scp/rsync 上传      │  │  GPU 训练      │  │
│  └───────────────┘  │   日志实时回传        │  └───────────────┘  │
└─────────────────────┘                       └─────────────────────┘

模式 B: Agent 直接运行在远程服务器
┌─────────────────────────────────────────────┐
│  远程 GPU 服务器                              │
│                                             │
│  ┌───────────────┐    ┌───────────────┐     │
│  │  Pico Agent   │───▶│  训练脚本      │     │
│  │  (SSH 进入后)  │    │  本地执行      │     │
│  │  CLI 交互      │    │  GPU 直连      │     │
│  └───────────────┘    └───────────────┘     │
└─────────────────────────────────────────────┘
```

## 二、远程服务器配置

### 配置文件 ~/.pico-agent/config.yaml

```yaml
# 本地 LLM 配置（用于 Agent 决策）
model:
  provider: openai
  model: gpt-4o
  api_key: sk-xxx

# 远程服务器列表
servers:
  gpu-server-1:
    host: 192.168.1.100
    port: 22
    user: jinjie
    key_path: ~/.ssh/id_rsa          # 或 password
    work_dir: /data/jinjie/pico-work
    gpu_count: 4
    gpu_type: "NVIDIA A100 80GB"
    conda_env: "detect"              # 自动激活的 conda 环境
    description: "主力训练服务器，4卡A100"

  gpu-server-2:
    host: 10.0.0.50
    port: 22
    user: jinjie
    key_path: ~/.ssh/id_rsa
    work_dir: /home/jinjie/pico-work
    gpu_count: 2
    gpu_type: "NVIDIA V100 32GB"
    conda_env: "mmdet"
    description: "MMDetection 专用服务器"

# 默认服务器
default_server: gpu-server-1
```

### SSH 工具实现

```python
# tools/remote/ssh_client.py
class SSHClient:
    """paramiko 封装，支持远程命令执行、文件传输"""
    
    def connect(self, server_config: dict): ...
    def execute(self, command: str, timeout: int = 300) -> dict:
        """远程执行命令，返回 {stdout, stderr, exit_code}"""
    def upload(self, local_path: str, remote_path: str): ...
    def download(self, remote_path: str, local_path: str): ...
    def upload_dir(self, local_dir: str, remote_dir: str): ...
    def stream_logs(self, remote_log_path: str, callback):
        """实时流式读取远程日志"""
```

### 远程工具集

```python
# tools/remote/remote_terminal.py
remote_terminal(command: str, server: str = "default", timeout: int = 300)
# 通过 SSH 在远程服务器执行命令

# tools/remote/file_transfer.py
file_upload(local_path: str, remote_path: str, server: str = "default")
file_download(remote_path: str, local_path: str, server: str = "default")
file_sync(local_dir: str, remote_dir: str, direction: str = "upload", server: str = "default")
# direction: "upload" (本地→远程) 或 "download" (远程→本地)

# tools/remote/server_info.py
server_status(server: str = "default")
# 返回: {"gpu_usage": [...], "memory": {...}, "disk": {...}, "running_jobs": [...]}
```

## 三、数据集远程工作流

```
本地数据集                    远程服务器
./datasets/traffic/           /data/jinjie/pico-work/datasets/traffic/
  ├── images/                   ├── images/
  │   ├── train/                │   ├── train/
  │   └── val/                  │   └── val/
  ├── labels/                   ├── labels/
  │   ├── train/                │   ├── train/
  │   └── val/                  │   └── val/
  └── data.yaml                 └── data.yaml

Step 1: 本地准备数据 → file_sync(upload) → 上传到远程
Step 2: 远程启动训练 → remote_terminal(train start)
Step 3: 远程监控进度 → remote_terminal(train monitor) 或 stream_logs
Step 4: 下载结果 → file_download → 下载 weights/results
```

### 自动同步策略
- **上传时机**: 开始训练前、数据集更新后
- **下载时机**: 训练完成、用户要求下载
- **增量同步**: rsync 只传输变化的文件

## 四、双框架适配层

### YOLO (Ultralytics) — 本地/远程通用
```python
# YOLO 命令行方式，通过 terminal/remote_terminal 执行
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 device=0,1,2,3
```

### MMDetection — 远程为主
```python
# MMDetection 用 Python API，需要在远程服务器有 mmdet 环境
tools/train.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py --gpus 4
# 或分布式
bash tools/dist_train.sh configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py 4
```

### 框架选择策略
```
用户意图 → Agent 判断
  ├── "YOLO"/"ultralytics" → YOLO 工具链
  ├── "mmdet"/"MMDetection"/"rtmdet" → MMDetection 工具链
  └── 未指定 → 根据任务推荐
      ├── 简单检测任务 → YOLO（更易用）
      └── 复杂/学术任务 → MMDetection（模型更丰富）
```

## 五、分布式训练支持

### YOLO 多卡
```bash
# 自动 DataParallel（ultralytics 内置）
yolo train data=data.yaml model=yolov8s.pt device=0,1,2,3

# DDP 模式
yolo train data=data.yaml model=yolov8s.pt device=0,1,2,3 amp=true
```

### MMDetection 多卡
```bash
# torchrun 分布式
bash tools/dist_train.sh config.py 4 --launcher pytorch

# SLURM 集群
bash tools/slurm_train.sh partition job_name config.py
```

### Agent 自动处理
```python
# trainer.py 中自动根据 server config 生成正确的训练命令
def generate_train_command(config, server_info):
    if config.framework == "yolo":
        gpus = ",".join(str(i) for i in range(server_info["gpu_count"]))
        return f"yolo train data={config.data} model={config.model} device={gpus}"
    elif config.framework == "mmdet":
        gpu_count = server_info["gpu_count"]
        return f"bash tools/dist_train.sh {config.config_file} {gpu_count}"
```

## 六、SAM 自动标注工作流

```
无标注图片 → SAM 自动标注 → YOLO/MMDet 格式

Step 1: Agent 用 SAM 的 auto 模式生成初步标注
        - 自动模式：SAM 对每张图自动预测所有目标
        - 点击模式：用户指定点/框，SAM 精确分割

Step 2: 转换为目标格式
        - COCO JSON → YOLO txt (归一化坐标)
        - COCO JSON → VOC XML
        - COCO JSON → MMDet format

Step 3: 人工审核（可选）
        Agent 生成标注预览图供用户确认
```

## 七、项目结构（最终版）

```
pico-agent/
├── pico/
│   ├── __init__.py
│   ├── agent.py             # 核心对话循环
│   ├── llm.py               # LLM Provider 适配
│   ├── config.py            # YAML 配置管理
│   ├── session.py           # SQLite 会话存储
│   ├── memory.py            # 跨会话记忆
│   ├── compression.py       # 上下文压缩
│   ├── delegation.py        # 子 Agent 委派
│   ├── cli.py               # prompt_toolkit CLI
│   ├── tools/
│   │   ├── __init__.py      # 工具自动发现
│   │   ├── registry.py      # 工具注册表
│   │   ├── file_tools.py    # 本地文件操作
│   │   ├── terminal.py      # 本地终端
│   │   ├── web_tools.py     # Web 搜索/提取
│   │   ├── vision.py        # 图像分析
│   │   ├── remote/          # ★ 远程服务器工具
│   │   │   ├── __init__.py
│   │   │   ├── ssh_client.py    # SSH 连接封装 (paramiko)
│   │   │   ├── remote_terminal.py # 远程命令执行
│   │   │   ├── file_transfer.py  # 文件上传/下载/同步
│   │   │   └── server_info.py   # 远程服务器状态查询
│   │   └── detection/       # ★ 检测专用工具
│   │       ├── __init__.py
│   │       ├── dataset.py       # 数据集探索/划分/转换
│   │       ├── annotation.py    # 标注：SAM 自动标注、格式转换、质量检查
│   │       ├── config_gen.py    # 配置生成（YOLO/MMDet）
│   │       ├── trainer.py       # 训练：启动/监控/断点续训（本地+远程）
│   │       ├── evaluator.py     # 评估：mAP、PR 曲线、混淆矩阵、bad case
│   │       ├── inference.py     # 推理：单图/批量/视频
│   │       └── exporter.py      # 导出：ONNX/TensorRT/速度测试
│   └── skills/
│       └── detection.md     # 检测全流程技能文档
├── pyproject.toml
├── config.yaml              # 默认配置（含 servers 配置示例）
└── README.md
```

## 八、新增依赖

```
# 原有
openai>=1.0
anthropic>=0.30
pyyaml>=6.0
prompt_toolkit>=3.0
rich>=13.0
duckduckgo-search>=5

# 新增：远程服务器
paramiko>=3.0           # SSH 客户端

# 新增：检测框架（可选，按需安装）
ultralytics>=8.0        # YOLOv8/v11
mmcv>=2.0               # MMDetection 基础
mmdet>=3.0              # MMDetection

# 新增：图像/数据处理
opencv-python>=4.8
Pillow>=10.0
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
numpy>=1.24

# 新增：导出（可选）
onnx>=1.14
onnxruntime>=1.16
```
