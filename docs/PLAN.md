# Pico Agent — 最小可行 AI Agent 系统

## 项目目标
构建一个功能完整的 AI Agent 框架（类 Hermes Agent 的最小版本），具备：
- 对话循环 + 工具调用
- 文件操作、终端执行、Web 搜索
- 会话持久化（SQLite）
- 跨会话记忆
- 上下文自动压缩
- CLI 交互界面
- 子 Agent 委派

## 架构设计

```
pico-agent/
├── pico/
│   ├── __init__.py          # 包入口
│   ├── agent.py             # 核心对话循环 (AIAgent)
│   ├── llm.py               # LLM Provider 适配层
│   ├── config.py            # YAML 配置管理
│   ├── session.py           # SQLite 会话存储
│   ├── memory.py            # 跨会话记忆
│   ├── compression.py       # 上下文压缩
│   ├── delegation.py        # 子 Agent 委派
│   ├── cli.py               # prompt_toolkit CLI
│   ├── tools/
│   │   ├── __init__.py      # 自动发现 + 注册
│   │   ├── registry.py      # 工具注册表
│   │   ├── file_tools.py    # read_file, write_file, search_files
│   │   ├── terminal.py      # shell 命令执行
│   │   └── web_tools.py     # 网页搜索与提取
│   └── prompts/
│       └── system.md        # 系统提示词
├── config.yaml              # 默认配置
├── requirements.txt         # 依赖
└── README.md                # 使用说明
```

## 模块详细设计

### 1. config.py — 配置管理
- 加载 `~/.pico-agent/config.yaml`（不存在则用默认值）
- 支持环境变量覆盖（`PICO_API_KEY`, `PICO_MODEL` 等）
- 提供 `get_config() -> dict` 全局访问

### 2. llm.py — LLM Provider 适配
- `LLMProvider` 抽象基类
- `OpenAIProvider` — 兼容 OpenAI/DeepSeek/vLLM 等所有 OpenAI 格式 API
- `AnthropicProvider` — Claude 系列原生支持
- 统一接口：`chat(messages, tools, system) -> LLMResponse`
- 流式输出支持（可选）

### 3. tools/registry.py — 工具注册表
- `ToolRegistry` 类：register / get_schemas / dispatch
- 工具自动发现：扫描 `tools/` 下所有 `.py` 文件中的 `registry.register()` 调用
- 每个工具有 schema（OpenAI function calling 格式）、handler、check_fn

### 4. tools/file_tools.py — 文件操作
- `read_file(path, offset, limit)` — 读取文件（带行号）
- `write_file(path, content)` — 写入文件
- `search_files(pattern, path, target)` — 内容搜索 / 文件名搜索

### 5. tools/terminal.py — 终端执行
- `terminal(command, timeout)` — 执行 shell 命令
- 超时控制（默认 120 秒）
- 输出截断（防止过长）

### 6. tools/web_tools.py — Web 工具
- `web_search(query)` — 使用 DuckDuckGo 搜索（免费，无需 API key）
- `web_extract(url)` — 提取网页正文内容

### 7. agent.py — 核心对话循环
```
AIAgent:
  - __init__(config, tools, session, memory)
  - run(user_message) -> str
    1. 加载系统提示 + 记忆上下文
    2. while iterations < max_turns:
       a. 调用 LLM
       b. 如有 tool_calls → 依次执行，结果追加 → continue
       c. 如为文本回复 → 保存到会话，return
    3. 上下文压缩（接近 token 上限时触发）
```

### 8. session.py — 会话存储
- SQLite 数据库 `~/.pico-agent/sessions.db`
- 表：sessions (id, title, created_at), messages (id, session_id, role, content, timestamp)
- 支持：创建会话、追加消息、查询历史、列出最近会话、搜索

### 9. memory.py — 持久记忆
- Markdown 文件 `~/.pico-agent/memory.md`
- 记忆注入到每次对话的系统提示中
- 提供 add/remove/list 操作

### 10. compression.py — 上下文压缩
- 当消息 token 数 > 阈值时触发
- 将旧消息压缩为摘要（用 LLM 总结）
- 保留最近 N 条完整消息

### 11. delegation.py — 子 Agent 委派
- `delegate_task(goal, context, toolsets) -> str`
- 创建隔离的 AIAgent 实例执行子任务
- 只返回摘要给父 Agent

### 12. cli.py — CLI 交互
- prompt_toolkit 实现交互式 REPL
- 支持：多行输入、历史记录、Ctrl+C 取消
- 斜杠命令：/new, /help, /quit, /sessions, /memory
- 彩色输出（rich 库）

## 依赖
```
openai>=1.0           # LLM 调用（兼容大多数 provider）
anthropic>=0.30       # Claude 原生支持
pyyaml>=6.0           # 配置管理
prompt_toolkit>=3.0   # CLI 交互
rich>=13.0            # 彩色输出
duckduckgo-search>=5  # Web 搜索
```

## 测试策略
- 先用 `python -c "from pico.agent import AIAgent"` 验证导入
- 配置一个 API key 后运行 `pico-agent "hello"` 测试基本对话
- 测试工具调用：让 agent 读取文件、执行命令
- 测试会话持久化：退出后重新进入，查看历史
