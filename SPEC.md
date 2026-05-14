# Pico Agent — 完整实现计划

## 项目路径
`/home/jinjie/projects/pico-agent/`

## 需要创建的文件

```
pico-agent/
├── pyproject.toml
├── config.example.yaml
├── README.md
├── pico/
│   ├── __init__.py
│   ├── agent.py              # 核心对话循环 AIAgent
│   ├── llm.py                # LLM Provider (OpenAI compatible + Anthropic)
│   ├── config.py             # YAML 配置加载 ~/.pico-agent/config.yaml
│   ├── session.py            # SQLite 会话存储
│   ├── memory.py             # 跨会话记忆 (memory.md)
│   ├── compression.py        # 上下文自动压缩
│   ├── delegation.py         # 子 Agent 委派
│   ├── cli.py                # prompt_toolkit CLI + rich 输出
│   ├── tools/
│   │   ├── __init__.py       # 自动发现 tools/ 下所有 .py
│   │   ├── registry.py       # ToolRegistry: register / get_schemas / dispatch
│   │   ├── file_tools.py     # read_file, write_file, search_files
│   │   ├── terminal.py       # terminal(command, timeout, background)
│   │   ├── web_tools.py      # web_search (DuckDuckGo), web_extract
│   │   ├── vision.py         # vision_analyze (base64 图片发给 LLM)
│   │   ├── remote/
│   │   │   ├── __init__.py
│   │   │   ├── ssh_client.py     # paramiko SSH 封装
│   │   │   ├── remote_terminal.py # 远程命令执行
│   │   │   ├── file_transfer.py  # upload/download/sync (rsync)
│   │   │   └── server_info.py    # 远程 GPU/CPU/内存状态
│   │   └── detection/
│   │       ├── __init__.py
│   │       ├── dataset.py        # explore/split/convert
│   │       ├── annotation.py     # SAM 标注、格式转换、质量检查、可视化
│   │       ├── config_gen.py     # YOLO/MMDet 配置生成 + 超参数推荐
│   │       ├── trainer.py        # 训练启动/监控/日志分析/断点续训 (本地+远程)
│   │       ├── evaluator.py      # mAP/PR 曲线/混淆矩阵/bad case/模型对比
│   │       ├── inference.py      # 单图/批量/视频推理
│   │       └── exporter.py       # ONNX/TensorRT/CoreML 导出 + benchmark
│   └── skills/
│       └── detection.md      # 检测全流程技能提示词
```

## 核心模块设计

### config.py
- 加载 `~/.pico-agent/config.yaml`，不存在则用默认值
- 环境变量覆盖：PICO_API_KEY, PICO_MODEL, PICO_BASE_URL
- 默认配置：
  ```yaml
  model:
    provider: openai
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1
    api_key: ${PICO_API_KEY}
  agent:
    max_turns: 50
    max_tokens: 128000
    compression_threshold: 0.80
  servers: {}
  default_server: ""
  ```

### llm.py
```python
class LLMResponse:
    content: str                    # 文本回复
    tool_calls: list[ToolCall]      # 工具调用
    usage: dict                     # token 用量

class LLMProvider(ABC):
    def chat(self, messages, tools=None, system=None) -> LLMResponse: ...

class OpenAIProvider(LLMProvider):
    """兼容 OpenAI/DeepSeek/vLLM/任何 OpenAI 格式 API"""

class AnthropicProvider(LLMProvider):
    """Claude 原生 API"""
```

### agent.py — 核心对话循环
```python
class AIAgent:
    def __init__(self, config, tool_registry, session, memory): ...
    
    def run(self, user_message: str) -> str:
        """一次对话交互（可能包含多轮工具调用）"""
        messages = self._build_messages(user_message)
        
        for turn in range(self.config.max_turns):
            # 检查是否需要压缩
            messages = self.compression.maybe_compress(messages)
            
            response = self.llm.chat(
                messages=messages,
                tools=self.tools.get_schemas(),
                system=self._build_system_prompt()
            )
            
            if response.tool_calls:
                for call in response.tool_calls:
                    result = self.tools.dispatch(call.name, call.arguments)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                continue
            
            # 纯文本回复 → 保存并返回
            self.session.add_message("assistant", response.content)
            return response.content
        
        return "达到最大迭代次数，请重试。"
```

### session.py — SQLite
- 表：sessions(id, title, created_at, updated_at)
- 表：messages(id, session_id, role, content, tool_call_id, timestamp)
- 操作：create_session, add_message, get_messages, list_sessions, search

### memory.py
- 文件：`~/.pico-agent/memory.md`
- 操作：add(content), remove(keyword), load() -> str
- 每次对话注入系统提示

### compression.py
- 计算消息列表 token 数（简单估算：len(text)/4）
- 超过阈值时，将前半部分用 LLM 总结为一条摘要

### delegation.py
```python
def delegate_task(goal: str, context: str, toolsets: list[str] = None) -> str:
    """创建隔离子 Agent 执行任务，只返回 summary"""
    child = AIAgent(config, ToolRegistry(toolsets), MemorylessSession(), NoMemory())
    result = child.run(f"{goal}\n\nContext: {context}")
    return result
```

### cli.py
- prompt_toolkit REPL
- 斜杠命令：/new, /help, /quit, /sessions, /memory, /server
- rich 彩色输出（表格、代码高亮、进度条）
- Ctrl+C 取消，Ctrl+D 退出

### tools/registry.py
```python
class ToolRegistry:
    def register(self, name, toolset, schema, handler, check_fn=None): ...
    def get_schemas(self, toolsets=None) -> list[dict]: ...
    def dispatch(self, name, args) -> str: ...  # 返回 JSON string
    def discover(self, tools_dir): ...  # 自动导入 tools/ 下模块
```

### tools/remote/ssh_client.py
```python
import paramiko

class SSHClient:
    def __init__(self, host, port, user, key_path=None, password=None): ...
    def execute(self, command, timeout=300) -> dict:
        """返回 {"stdout": ..., "stderr": ..., "exit_code": ...}"""
    def upload(self, local_path, remote_path): ...
    def download(self, remote_path, local_path): ...
    def upload_dir(self, local_dir, remote_dir): ...
    def stream_command(self, command, callback): ...

# 全局连接池
_ssh_clients: dict[str, SSHClient] = {}

def get_ssh_client(server_name: str) -> SSHClient: ...
```

### tools/remote/remote_terminal.py
```python
remote_terminal(command: str, server: str = "default", timeout: int = 300, work_dir: str = None)
# 自动处理 conda activate（如果配置了 conda_env）
# 返回 {"output": "...", "exit_code": 0}
```

### tools/remote/file_transfer.py
```python
file_upload(local_path, remote_path, server="default")
file_download(remote_path, local_path, server="default")
file_sync(local_dir, remote_dir, direction="upload", server="default", exclude=["*.pyc", "__pycache__"])
```

### tools/remote/server_info.py
```python
server_status(server="default")
# 运行 nvidia-smi, free -h, df -h, ps aux
# 返回 {"gpus": [...], "memory": {...}, "disk": {...}, "running_processes": [...]}
```

### tools/detection/ 核心逻辑

**dataset.py:**
- explore: 扫描目录结构，检测 YOLO/MMDet/COCO/VOC 格式，统计图片数、标注数、类别分布
- split: 分层采样划分 train/val/test
- convert: COCO↔VOC↔YOLO 格式互转

**annotation.py:**
- sam_annotate: 调用 ultralytics SAM 模型自动标注
- check: 检查标注质量（越界框、重复框、小框、空图）
- visualize: 在图片上绘制标注框
- convert_format: 标注格式转换

**config_gen.py:**
- yolo_config: 生成 data.yaml + 推荐训练参数
- mmdet_config: 生成 mmdet 训练配置文件
- recommend: 根据数据集统计 + GPU 信息推荐超参数

**trainer.py:**
- start: 启动训练（本地 terminal 或远程 remote_terminal）
- monitor: 解析训练日志（YOLO 的 results.csv，MMDet 的 log.json）
- analyze: 分析训练趋势，检测过拟合/欠拟合
- resume: 断点续训

**evaluator.py:**
- evaluate: 运行模型评估，返回 mAP 等指标
- pr_curve: 生成 PR 曲线
- confusion_matrix: 生成混淆矩阵
- bad_cases: 找出错误案例并可视化
- compare: 多模型对比

**inference.py:**
- image: 单图推理
- batch: 批量推理
- video: 视频推理

**exporter.py:**
- onnx: ONNX 导出
- tensorrt: TensorRT 导出
- benchmark: 速度基准测试

## 检测工具的远程适配策略

检测工具（trainer/evaluator/inference/exporter）需要同时支持本地和远程两种执行方式：

```python
# 通用执行器
def execute_command(command: str, local: bool = True, server: str = "default") -> dict:
    if local:
        return terminal(command)
    else:
        return remote_terminal(command, server=server)

# 检测工具内部根据配置自动选择
def train_start(config_path, framework="yolo", server="default", ...):
    use_remote = server != "" and config.get("servers", {}).get(server)
    if use_remote:
        # 上传数据和配置 → 远程执行
        file_sync(data_dir, remote_data_dir, "upload", server)
        return remote_terminal(train_command, server=server)
    else:
        return terminal(train_command)
```

## Skills 文件

pico/skills/detection.md 内容：
- 触发条件
- 全流程编排策略（数据→配置→训练→评估→推理→导出）
- 双框架（YOLO/MMDetection）最佳实践
- 远程服务器工作流
- SAM 标注工作流
- 分布式训练指导
- 常见问题诊断表

## 实现要求

1. Python 3.10+，类型标注
2. pyproject.toml 管理依赖和入口点
3. 入口点：`pico-agent` CLI 命令
4. 所有工具 handler 返回 JSON 字符串
5. 工具执行失败返回 {"success": false, "error": "..."}
6. 检测工具的 check_fn 检查 ultralytics/mmdet 是否安装
7. 远程工具的 check_fn 检查 paramiko 是否安装 + config 中是否有 servers
8. 日志用 logging 模块
9. 确保 `pip install -e .` 后 `pico-agent "hello"` 可运行（不依赖检测框架）
