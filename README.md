# Pico Agent 🤖

A lightweight AI agent framework with tool use, session management, dataset/code download, and one-step detection training.

## Quick Start

```bash
# Install
cd ~/projects/pico-agent
pip install -e .

# Configure
export PICO_API_KEY="your-api-key"
export PICO_MODEL="gpt-4o"  # or claude-3-5-sonnet, etc.

# Start
pico-agent
```

## Usage — Three Levels of Easy

### 🚀 Level 1: One Command (CLI Shortcuts)

```bash
# Train a detection model — just point to your data
pico-agent train ./datasets/traffic

# Preview training plan without starting
pico-agent train ./data --model yolov8s --dry-run

# Evaluate a trained model
pico-agent eval runs/detect/train/weights/best.pt

# Search GitHub for repos
pico-agent search "RT-DETR detection" --source github

# Search and download a dataset
pico-agent download "cat dog detection" --source huggingface

# Clone a repo and install deps
pico-agent clone https://github.com/user/repo --install --browse

# Run inference on an image
pico-agent infer best.pt photo.jpg
```

### 💬 Level 2: Natural Language (REPL)

```
>>> 帮我在 ./datasets/traffic 上训练一个检测模型
Agent: [auto_train] 检测到 YOLO 格式数据集，2000 张图片，5 个类别...
       推荐使用 yolov8s，150 个 epoch，batch size 16...
       训练已启动！

>>> 搜一下 GitHub 上的 SAM 仓库，clone 下来
Agent: [code_search] 找到 facebookresearch/segment-anything (45K⭐)...
       [code_clone] 已 clone 到 ~/.pico-agent/repos/segment-anything/
       [code_install] 正在安装依赖...

>>> 下载一个行人检测数据集
Agent: [dataset_search] 搜到以下数据集...
       [dataset_download] 已下载到 ~/.pico-agent/datasets/pedestrian/
```

### 🔧 Level 3: Advanced (Direct Tool Control)

```
>>> 用 MMDetection 训练 faster-rcnn，数据在 ./data/coco，远程服务器 m5pro
Agent: [dataset_info] 检查数据集
       [mmdet_config] 生成 Faster-RCNN 配置
       [remote_file_sync] 同步数据到远程服务器
       [remote_terminal] 启动分布式训练
       [train_monitor] 监控训练进度...
```

## Project Structure

```
pico-agent/
├── pico/
│   ├── agent.py              # Core conversation loop
│   ├── llm.py                # OpenAI + Anthropic providers
│   ├── cli.py                # CLI with shortcuts + REPL
│   ├── config.py             # YAML config
│   ├── session.py            # SQLite session storage
│   ├── memory.py             # Cross-session memory
│   ├── compression.py        # Context auto-compression
│   ├── delegation.py         # Sub-agent delegation
│   └── tools/
│       ├── registry.py       # Tool registration + dispatch
│       ├── file_tools.py     # read_file, write_file, search_files
│       ├── terminal.py       # Shell command execution
│       ├── web_tools.py      # web_search, web_extract
│       ├── vision.py         # Image analysis
│       ├── dataset_tools.py  # 🔍 Search + download datasets
│       ├── code_tools.py     # 🔍 Search + clone + install repos
│       ├── workflow_tools.py # 🚀 auto_train, quick_eval, deploy_model
│       ├── detection/        # 🔬 Advanced detection tools (25 tools)
│       └── remote/           # 🌐 Remote server tools
├── pyproject.toml
└── config.example.yaml
```

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| **Easy Mode** | `auto_train`, `quick_eval`, `deploy_model` | One-step operations |
| **Dataset** | `dataset_search`, `dataset_download`, `dataset_info` | Find & download data |
| **Code** | `code_search`, `code_clone`, `code_install`, `code_browse`, `code_exec` | Find & setup repos |
| **Detection** | `yolo_config`, `train_start`, `evaluate`, `bad_cases`, etc. | Fine-grained control |
| **Annotation** | `sam_annotate`, `convert_annotation` | Label data with SAM |
| **Export** | `export_onnx`, `export_trt`, `benchmark` | Model deployment |
| **Remote** | `remote_terminal`, `remote_file_*` | Remote server ops |

## Supported Dataset Formats (Auto-detected)

| Format | Structure | Example |
|--------|-----------|---------|
| **YOLO** | `images/train/` + `labels/train/` | Ultralytics standard |
| **YOLO YAML** | `data.yaml` with paths | Pre-configured YOLO |
| **COCO** | `images/` + `annotations.json` | COCO-style JSON |
| **VOC** | `images/` + `*.xml` annotations | Pascal VOC |

## Supported Download Sources

| Source | Auth Required | Tools |
|--------|--------------|-------|
| **HuggingFace** | No (optional token for private) | `dataset_search`, `dataset_download` |
| **Kaggle** | Yes (`KAGGLE_USERNAME` + `KAGGLE_KEY`) | `dataset_search`, `dataset_download` |
| **Roboflow** | Yes (`ROBOFLOW_API_KEY`) | `dataset_download` |
| **URL** | No | `dataset_download(source="url")` |
| **GitHub** | Optional (`GITHUB_TOKEN` for higher rate limits) | `code_search`, `code_clone` |

## Installation Options

```bash
pip install -e .                       # Core (file, terminal, web, vision)
pip install -e ".[datasets]"           # + HuggingFace, Kaggle, Roboflow
pip install -e ".[remote]"             # + SSH remote servers
pip install -e ".[detection]"          # + YOLO, OpenCV, MMDetection
pip install -e ".[all]"                # Everything
```

## Environment Variables

```bash
# LLM (required)
export PICO_API_KEY="sk-..."
export PICO_MODEL="gpt-4o"                     # or claude-3-5-sonnet-20241022
export PICO_API_BASE="https://api.openai.com/v1"  # or custom proxy

# Optional: higher rate limits
export GITHUB_TOKEN="ghp_..."

# Optional: Kaggle datasets
export KAGGLE_USERNAME="yourname"
export KAGGLE_KEY="..."

# Optional: Roboflow datasets
export ROBOFLOW_API_KEY="..."
```

## Configuration

Create `~/.pico-agent/config.yaml`:

```yaml
# LLM settings
model: gpt-4o
api_base: https://api.openai.com/v1

# Remote servers (for detection workflows)
servers:
  gpu-server:
    host: 192.168.1.100
    user: admin
    key_file: ~/.ssh/id_rsa
```

## Architecture

```
User Input
    ↓
┌─────────────┐
│   CLI Layer  │ ← Shortcuts (train/eval/search/clone)
└──────┬──────┘
       ↓
┌─────────────┐
│  Agent Loop  │ ← LLM ↔ Tools cycle (max_turns)
└──────┬──────┘
       ↓
┌─────────────┐     ┌──────────────┐
│ Tool Registry│────→│  Tool Handler │
└─────────────┘     └──────────────┘
       ↓
┌─────────────┐     ┌──────────────┐
│    LLM       │←──→│   Session     │
│  Provider    │     │   + Memory    │
└─────────────┘     └──────────────┘
```

## License

MIT
