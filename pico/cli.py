"""Pico Agent CLI — prompt_toolkit REPL with rich output.

Usage:
    pico-agent                          # Start interactive REPL
    pico-agent "hello"                  # Single-shot mode
    pico-agent train ./data             # Quick training shortcut
    pico-agent eval model.pt            # Quick evaluation shortcut
    pico-agent search "YOLO detection"  # Search GitHub repos
    pico-agent download "cat detection" # Search & download datasets
    pico-agent clone <url>              # Clone a repo
    pico-agent --session <id>           # Resume a specific session
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pico.agent import AIAgent

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
_RED = "\033[31m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _print_error(msg: str) -> None:
    """Print an error message in red to stderr."""
    if sys.stderr.isatty():
        print(f"{_RED}{_BOLD}Error:{_RESET} {_RED}{msg}{_RESET}", file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------

def _rich_print(text: str) -> None:
    """Print with rich markdown rendering if available, else plain print."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print(Markdown(text))
    except ImportError:
        print(text)


def _rich_print_json(data: dict) -> None:
    """Print a dict as formatted JSON."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()
        console.print(Syntax(json.dumps(data, indent=2, ensure_ascii=False), "json"))
    except ImportError:
        print(json.dumps(data, indent=2, ensure_ascii=False))


def _rich_print_sessions(sessions: list[Any]) -> None:
    """Print session list as a rich table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Sessions", show_lines=False)
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Title", style="bold")
        table.add_column("Updated", style="cyan")

        for s in sessions:
            table.add_row(s.id[:8], s.title, s.updated_at)
        console.print(table)
    except ImportError:
        for s in sessions:
            print(f"  {s.id[:8]}  {s.title}  ({s.updated_at})")


def _rich_print_memory(memory_content: str) -> None:
    """Print memory content."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        if memory_content.strip():
            console.print(Markdown(memory_content))
        else:
            console.print("[dim]Memory is empty.[/dim]")
    except ImportError:
        print(memory_content if memory_content.strip() else "(empty)")


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def _handle_slash_command(cmd: str, agent: AIAgent) -> str | None:
    """Handle a slash command. Returns a display string, or None if not recognized."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/quit" or command == "/exit":
        raise SystemExit(0)

    if command == "/help":
        return (
            "# Pico Agent Commands\n\n"
            "**Session:**\n"
            "- `/new` — Start a new session\n"
            "- `/sessions` — List recent sessions\n"
            "- `/compact` — Compress context to save tokens\n"
            "- `/clear` — Clear screen + start new session\n"
            "- `/usage` — Show token usage and cost for current session\n\n"
            "**Memory:**\n"
            "- `/memory` — Show persistent memory\n"
            "- `/memory add <text>` — Add a memory entry\n"
            "- `/memory rm <keyword>` — Remove memory entries\n\n"
            "**Knowledge:**\n"
            "- `/kb` — Show training knowledge base summary\n"
            "- `/kb best [task]` — Show best training run\n"
            "- `/kb list [task]` — List recent training runs\n\n"
            "**Diagnostics:**\n"
            "- `/doctor` — Health check (API, config, tools, disk)\n"
            "- `/model [name]` — Show or switch LLM model\n"
            "- `/tokenjuice` — Show compression rules\n\n"
            "**Tools:**\n"
            "- `/tools` — List available tools\n\n"
            "**UI:**\n"
            "- `/ui [port]` — Launch web dashboard\n"
            "- `/monitor [port]` — Launch training monitor\n\n"
            "**Quick Actions:**\n"
            "- Just describe what you want to do! The agent will figure out the tools.\n"
            "  Examples:\n"
            "  - `train my dataset at ./data` — auto-detect & train\n"
            "  - `search GitHub for YOLO v9` — find repos\n"
            "  - `download COCO dataset` — search & download\n"
            "  - `evaluate model at runs/detect/train/weights/best.pt`\n\n"
            "- `/help` — Show this help\n"
            "- `/quit` — Exit\n"
        )

    if command == "/new":
        old_id = agent.session_id
        agent.session_id = ""
        return f"Started new session (was: {old_id[:8] if old_id else 'none'})"

    if command == "/sessions":
        if hasattr(agent.session, "list_sessions"):
            sessions = agent.session.list_sessions(limit=20)
            if sessions:
                _rich_print_sessions(sessions)
                return None  # already printed
            return "No sessions found."
        return "Session listing not available (memoryless session)."

    if command == "/memory":
        if arg.startswith("add "):
            memory_text = arg[4:].strip()
            if memory_text:
                agent.memory.add(memory_text)
                return f"Added to memory: {memory_text}"
            return "Usage: /memory add <text>"

        if arg.startswith("rm "):
            keyword = arg[3:].strip()
            if keyword:
                removed = agent.memory.remove(keyword)
                return f"Removed entries with '{keyword}'" if removed else f"No entries matched '{keyword}'"
            return "Usage: /memory rm <keyword>"

        memory_content = agent.memory.load()
        _rich_print_memory(memory_content)
        return None

    if command == "/tools":
        schemas = agent.tools.get_schemas()
        if not schemas:
            return "No tools registered."
        lines = ["# Available Tools\n"]
        for s in schemas:
            name = s.get("name", "?")
            desc = s.get("description", "No description").split("\n")[0]
            lines.append(f"- **{name}** — {desc}")
        return "\n".join(lines)

    if command == "/compact":
        from pico.compression import _messages_token_count
        if isinstance(agent.session, SessionStore) and agent.session_id:
            msgs = agent.session.get_messages_as_dicts(agent.session_id)
            msgs = [m for m in msgs if m["role"] != "system"]
            before = _messages_token_count(msgs)
            compressed = agent.compressor.maybe_compress(msgs)
            after = _messages_token_count(compressed)
            return f"Compressed: {before} → {after} tokens ({before - after} saved, {len(msgs)} → {len(compressed)} messages)"
        return "Compression not available in memoryless mode."

    if command == "/clear":
        import os
        os.system("clear" if os.name == "posix" else "cls")
        old_id = agent.session_id
        agent.session_id = ""
        return f"Screen cleared. Started new session (was: {old_id[:8] if old_id else 'none'})"

    if command == "/usage":
        if isinstance(agent.session, SessionStore) and agent.session_id:
            sess = agent.session.get_session(agent.session_id)
            if sess:
                lines = [
                    "# Session Usage\n",
                    f"- **Session**: `{sess.id[:8]}` — {sess.title}",
                    f"- **Messages**: {sess.message_count}",
                    f"- **API calls**: {sess.api_call_count}",
                    f"- **Input tokens**: {sess.input_tokens:,}",
                    f"- **Output tokens**: {sess.output_tokens:,}",
                    f"- **Cache read**: {sess.cache_read_tokens:,}",
                    f"- **Est. cost**: ${sess.estimated_cost_usd:.4f}",
                    f"- **Source**: {sess.source}",
                ]
                return "\n".join(lines)
            return "No active session."
        return "Usage tracking not available in memoryless mode."

    if command == "/ui":
        port = int(arg.strip()) if arg.strip().isdigit() else 8765
        from pico.ui.server import start_ui
        import threading
        t = threading.Thread(
            target=start_ui,
            kwargs={"port": port, "open_browser": True},
            daemon=True,
        )
        t.start()
        return f"Dashboard started at http://127.0.0.1:{port}"

    if command == "/monitor":
        port = int(arg.strip()) if arg.strip().isdigit() else 8766
        from pico.ui.training_server import start_monitor
        import threading
        t = threading.Thread(
            target=start_monitor,
            kwargs={"port": port, "open_browser": True},
            daemon=True,
        )
        t.start()
        return f"Training monitor started at http://127.0.0.1:{port}"

    # ---- /doctor: Health check ----
    if command == "/doctor":
        import shutil
        checks: list[str] = []
        # 1. API key
        if agent.config.api_key:
            masked = agent.config.api_key[:4] + "..." + agent.config.api_key[-4:] if len(agent.config.api_key) > 8 else "***"
            checks.append(f"✅ API key configured ({masked})")
        else:
            checks.append("❌ No API key set — set OPENAI_API_KEY or edit ~/.pico-agent/config.yaml")
        # 2. Provider / model
        checks.append(f"✅ Provider: {agent.config.provider} | Model: {agent.config.model}")
        # 3. Base URL
        checks.append(f"✅ Base URL: {agent.config.base_url}")
        # 4. YOLO
        yolo_path = shutil.which("yolo")
        if yolo_path:
            checks.append(f"✅ YOLO CLI: {yolo_path}")
        else:
            checks.append("⚠️ YOLO CLI not in PATH (install with: pip install ultralytics)")
        # 5. SSH (paramiko)
        try:
            import paramiko  # noqa: F401
            checks.append("✅ paramiko (SSH) available")
        except ImportError:
            checks.append("⚠️ paramiko not installed (remote features disabled)")
        # 6. Tools count
        tool_count = len(agent.tools.get_schemas())
        checks.append(f"✅ {tool_count} tools registered")
        # 7. Disk space
        try:
            import shutil as sh
            usage = sh.disk_usage(str(Path.home()))
            free_gb = usage.free / (1024**3)
            if free_gb < 5:
                checks.append(f"⚠️ Low disk space: {free_gb:.1f} GB free")
            else:
                checks.append(f"✅ Disk space: {free_gb:.1f} GB free")
        except Exception:
            checks.append("⚠️ Could not check disk space")
        # 8. TokenJuice
        rules_count = len(agent.tokenjuice.rules)
        checks.append(f"✅ TokenJuice: {rules_count} compression rules active")
        # 9. Training KB
        try:
            kb_count = len(agent.training_kb.list_runs(limit=1000))
            checks.append(f"✅ Training KB: {kb_count} runs recorded")
        except Exception as e:
            checks.append(f"⚠️ Training KB error: {e}")
        # 10. Sessions
        if isinstance(agent.session, SessionStore):
            stats = agent.session.get_stats() if hasattr(agent.session, 'get_stats') else {}
            checks.append(f"✅ Sessions DB: {stats.get('total_sessions', '?')} sessions, {stats.get('total_messages', '?')} messages")

        return "# Health Check\n\n" + "\n".join(checks)

    # ---- /model: Show or switch model ----
    if command == "/model":
        if not arg:
            return (
                f"# Current Model\n\n"
                f"- **Provider**: {agent.config.provider}\n"
                f"- **Model**: `{agent.config.model}`\n"
                f"- **Base URL**: {agent.config.base_url}\n"
                f"- **Max tokens**: {agent.config.max_tokens:,}\n"
                f"- **Temperature**: {agent.config.temperature}\n\n"
                f"Switch with: `/model <model_name>`\n"
                f"Example: `/model gpt-4o` or `/model deepseek-chat`"
            )
        # Switch model at runtime
        new_model = arg.strip()
        old_model = agent.config.model
        agent.config.data.setdefault("model", {})["model"] = new_model
        # Recreate LLM provider with new model
        try:
            from pico.llm import create_provider
            agent.llm = create_provider(agent.config)
            agent.compressor.llm = agent.llm
            return f"✅ Switched model: `{old_model}` → `{new_model}`"
        except Exception as e:
            # Rollback
            agent.config.data["model"]["model"] = old_model
            return f"❌ Failed to switch to `{new_model}`: {e}"

    # ---- /kb: Training Knowledge Base ----
    if command == "/kb":
        if arg.startswith("best"):
            task = arg[4:].strip()
            run = agent.training_kb.get_best_run(task=task)
            if not run:
                return f"No training runs found{' for task ' + task if task else ''}."
            return (
                f"# Best Training Run\n\n"
                f"- **Task**: {run.task or 'untitled'}\n"
                f"- **Model**: {run.model_arch}\n"
                f"- **mAP50**: {run.map50:.3f}\n"
                f"- **mAP50-95**: {run.map50_95:.3f}\n"
                f"- **Precision**: {run.precision_:.3f}\n"
                f"- **Recall**: {run.recall_:.3f}\n"
                f"- **Epochs**: {run.epochs}\n"
                f"- **Dataset**: {run.dataset_name or run.dataset_path}\n"
                f"- **Weights**: `{run.best_weights}`\n"
            )
        if arg.startswith("list"):
            task = arg[4:].strip()
            runs = agent.training_kb.list_runs(task=task)
            if not runs:
                return f"No training runs found{' for task ' + task if task else ''}."
            lines = [f"# Training Runs{' — ' + task if task else ''}\n"]
            for r in runs:
                from datetime import datetime
                ts = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M")
                lines.append(
                    f"- `{ts}` **{r.task or 'untitled'}** ({r.model_arch}) — "
                    f"mAP50={r.map50:.3f}, mAP50-95={r.map50_95:.3f} | {r.epochs}ep"
                )
            return "\n".join(lines)
        # Default: summary
        return agent.training_kb.summary_text()

    # ---- /tokenjuice: Show compression rules ----
    if command == "/tokenjuice":
        return (
            f"# TokenJuice Compression\n\n"
            f"**Status**: {'✅ Enabled' if agent.tokenjuice.enabled else '❌ Disabled'}\n\n"
            f"```\n{agent.tokenjuice.get_rules_summary()}\n```\n\n"
            f"Custom rules: `~/.pico-agent/tokenjuice/rules/*.json`\n"
            f"Project rules: `.pico-agent/tokenjuice/rules/*.json`"
        )

    return None


# ---------------------------------------------------------------------------
# Path validation helpers
# ---------------------------------------------------------------------------

def _validate_dir(path: str, label: str = "Directory") -> str | None:
    """Validate that a path is an existing directory. Returns error string or None."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"{label} not found: {path}"
    if not p.is_dir():
        return f"{label} is not a directory: {path}"
    return None


def _validate_file(path: str, label: str = "File") -> str | None:
    """Validate that a path is an existing file. Returns error string or None."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"{label} not found: {path}"
    if not p.is_file():
        return f"{label} is not a file: {path}"
    return None


# ---------------------------------------------------------------------------
# Shortcut commands — directly invoke tools (bypass LLM when possible)
# ---------------------------------------------------------------------------

def _shortcut_train(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent train <data_dir> [--model X] [--epochs N] [--dry-run]"""
    data_dir = args.data_dir
    err = _validate_dir(data_dir, "Dataset directory")
    if err:
        _print_error(err)
        return 1

    # Direct tool call — skip LLM
    tool_args: dict[str, Any] = {"data_dir": str(Path(data_dir).expanduser().resolve())}
    if args.model:
        tool_args["model"] = args.model
    if args.epochs:
        tool_args["epochs"] = args.epochs
    if args.batch:
        tool_args["batch"] = args.batch
    tool_args["dry_run"] = args.dry_run

    result = agent.tools.dispatch("auto_train", tool_args)
    try:
        data = json.loads(result)
        if data.get("success"):
            _rich_print_json(data)
            return 0
        else:
            _print_error(data.get("error", "Unknown error"))
            return 1
    except json.JSONDecodeError:
        print(result)
        return 0


def _shortcut_eval(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent eval <model_path> [--data data.yaml]"""
    err = _validate_file(args.model_path, "Model file")
    if err:
        _print_error(err)
        return 1

    tool_args: dict[str, Any] = {"model_path": str(Path(args.model_path).expanduser().resolve())}
    if args.data:
        tool_args["data_yaml"] = args.data

    result = agent.tools.dispatch("quick_eval", tool_args)
    try:
        data = json.loads(result)
        if data.get("success"):
            _rich_print_json(data)
            return 0
        else:
            _print_error(data.get("error", "Unknown error"))
            return 1
    except json.JSONDecodeError:
        print(result)
        return 0


def _shortcut_search(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent search <query> [--source github|huggingface|all]"""
    source = args.source or "all"
    if source == "all":
        prompt = f"Search for '{args.query}' across GitHub repos and HuggingFace datasets. Show me the top results."
    elif source == "github":
        prompt = f"Search GitHub for '{args.query}' repositories. Show top results with stars and clone URLs."
    else:
        prompt = f"Search {source} for '{args.query}' datasets. Show me the results."
    return _single_shot(agent, prompt)


def _shortcut_download(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent download <query_or_url> [--source huggingface|kaggle|url]"""
    source = args.source or "huggingface"
    if args.query_or_url.startswith("http"):
        prompt = f"Download this dataset: {args.query_or_url}"
    else:
        prompt = (
            f"Find and download a dataset matching '{args.query_or_url}' from {source}. "
            "Search first, then download the best match."
        )
    return _single_shot(agent, prompt)


def _shortcut_clone(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent clone <url> [--install]"""
    # Direct tool call
    tool_args: dict[str, Any] = {"url": args.url}
    result = agent.tools.dispatch("code_clone", tool_args)
    try:
        data = json.loads(result)
        if not data.get("success"):
            _print_error(data.get("error", "Unknown error"))
            return 1

        _rich_print_json(data)

        if args.install and data.get("success"):
            clone_path = data.get("path", "")
            if clone_path:
                print("\nInstalling dependencies...")
                install_result = agent.tools.dispatch("code_install", {"repo_path": clone_path})
                install_data = json.loads(install_result)
                _rich_print_json(install_data)
                return 0 if install_data.get("success") else 1
        return 0
    except json.JSONDecodeError:
        print(result)
        return 0


def _shortcut_infer(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent infer <model_path> <image_path>"""
    err = _validate_file(args.model_path, "Model file")
    if err:
        _print_error(err)
        return 1
    err = _validate_file(args.image_path, "Image file")
    if err:
        _print_error(err)
        return 1

    prompt = f"Run inference with model {args.model_path} on image {args.image_path}. Show the results."
    return _single_shot(agent, prompt)


def _shortcut_ui(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent ui [--port N] [--host ADDR] [--no-open]"""
    host = args.host or "127.0.0.1"
    port = args.port or 8765

    from pico.ui.server import start_ui

    start_ui(
        host=host,
        port=port,
        db_path=str(agent.session.db_path) if hasattr(agent.session, "db_path") else None,
        open_browser=not args.no_open,
    )
    return 0


def _shortcut_monitor(agent: AIAgent, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent monitor [--port N] [--scan DIR ...] [--no-open]"""
    port = args.port or 8766
    scan_roots = args.scan if args.scan else None

    from pico.ui.training_server import start_monitor

    start_monitor(
        port=port,
        scan_roots=scan_roots,
        open_browser=not args.no_open,
    )
    return 0


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------

def _single_shot(agent: AIAgent, message: str) -> int:
    """Run a single message through the LLM and print the response.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    message = message.strip()
    if not message:
        _print_error("empty message. Provide a question or command.")
        return 1
    try:
        response = agent.run(message)
        _rich_print(response)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as e:
        # Catch all errors including LLM provider exceptions
        # (OpenAIError, APIError, etc.) that don't inherit from
        # OSError/ValueError/RuntimeError. Broad catch is correct
        # for a CLI entry point — we never want raw tracebacks.
        msg = str(e)
        if "Missing" in msg and ("credentials" in msg.lower() or "api key" in msg.lower()):
            _print_error(
                "No API key configured. Set OPENAI_API_KEY or PICO_API_KEY environment variable,\n"
                "       or add api_key to ~/.pico-agent/config.yaml\n\n"
                "Run `pico-agent --help` for more information."
            )
        else:
            _print_error(str(e))
        return 1


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------

def _repl(agent: AIAgent) -> int:
    """Interactive REPL with prompt_toolkit."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        history_path = agent.config.data.get("_history_path", "")
        history = FileHistory(history_path) if history_path else None
        session = PromptSession(history=history)
    except ImportError:
        session = None

    from pico import __version__
    print(
        f"🤖 Pico Agent v{__version__} — AI-powered coding & detection assistant\n"
        "   Ask me anything, or type /help for commands. Ctrl+D to exit.\n"
    )

    while True:
        try:
            if session:
                user_input = session.prompt(">>> ").strip()
            else:
                user_input = input(">>> ").strip()
        except KeyboardInterrupt:
            # Ctrl+C at prompt: print newline and continue (don't exit)
            print()
            continue
        except EOFError:
            # Ctrl+D: exit cleanly
            print("\nGoodbye!")
            return 0

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            try:
                result = _handle_slash_command(user_input, agent)
                if result is not None:
                    _rich_print(result)
            except SystemExit:
                print("Goodbye!")
                return 0
            continue

        # Regular message → run agent
        try:
            if sys.stderr.isatty():
                print(f"{_DIM}Thinking...{_RESET}", file=sys.stderr)
            response = agent.run(user_input)
            _rich_print(response)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            logger.exception("Agent error")
            _print_error(str(e))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Supports both interactive REPL, single-shot messages, and shortcut commands.

    Examples:
        pico-agent                              # Interactive REPL
        pico-agent "what's 2+2?"                # Single-shot
        pico-agent train ./datasets/traffic     # Quick train
        pico-agent train ./data --model yolov8s --dry-run
        pico-agent eval runs/train/weights/best.pt
        pico-agent search "RT-DETR" --source github
        pico-agent download "coco 2017" --source huggingface
        pico-agent clone https://github.com/user/repo --install
        pico-agent infer best.pt image.jpg
    """
    if argv is None:
        argv = sys.argv[1:]

    KNOWN_COMMANDS = {"train", "eval", "search", "download", "clone", "infer", "ui", "monitor"}

    # --- Manual global flag extraction (avoids argparse subparser issues) ---
    session_id_arg = ""
    verbose = False
    config_path = None
    explicit_config = False
    positional: list[str] = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--help", "-h"):
            if positional and positional[0] in KNOWN_COMMANDS:
                positional.append(a)
                i += 1
            else:
                _print_help()
                sys.exit(0)
        elif a in ("--version", "-V"):
            from pico import __version__
            print(f"pico-agent {__version__}")
            sys.exit(0)
        elif a in ("--session", "-s") and i + 1 < len(argv):
            session_id_arg = argv[i + 1]; i += 2
        elif a.startswith("--session="):
            session_id_arg = a.split("=", 1)[1]; i += 1
        elif a in ("--verbose", "-v"):
            verbose = True; i += 1
        elif a in ("--config", "-c") and i + 1 < len(argv):
            config_path = argv[i + 1]; explicit_config = True; i += 2
        elif a.startswith("--config="):
            config_path = a.split("=", 1)[1]; explicit_config = True; i += 1
        elif a == "--":
            positional.extend(argv[i + 1:])
            break
        else:
            if a.startswith("-") and positional and positional[0] in KNOWN_COMMANDS:
                # Defer to subcommand parser (e.g. --port, --scan, --model)
                positional.append(a)
            elif a.startswith("-"):
                _print_error(f"unknown option: {a}")
                _print_help()
                sys.exit(1)
            else:
                positional.append(a)
            i += 1

    _setup_logging(verbose)

    # --- Validate explicit --config path (Bug #2) ---
    if explicit_config and config_path is not None:
        cfg_p = Path(config_path).expanduser().resolve()
        if not cfg_p.exists():
            _print_error(f"config file not found: {config_path}")
            sys.exit(1)

    # --- Route: command vs single-shot vs REPL ---
    command = None
    command_args: argparse.Namespace | None = None

    if positional and positional[0] in KNOWN_COMMANDS:
        command = positional[0]
        command_args = _parse_command_args(command, positional[1:])
    # else: single-shot message or REPL

    # Lazy imports to keep startup fast
    from pico.agent import AIAgent
    from pico.config import CONFIG_DIR, load_config
    from pico.memory import Memory
    from pico.session import SessionStore
    from pico.tools import discover_and_register
    from pico.tools.registry import ToolRegistry

    try:
        config = load_config(config_path, strict=explicit_config)
    except ValueError as e:
        _print_error(str(e))
        sys.exit(1)

    # Set working directory if configured
    if config.work_dir:
        import os
        work = os.path.expanduser(config.work_dir)
        if os.path.isdir(work):
            os.chdir(work)
            logger.info("Working directory: %s", work)
        else:
            _print_error(f"work_dir not found: {config.work_dir}")
            sys.exit(1)

    # Set up history file
    history_path = str(CONFIG_DIR / "history")
    config.data["_history_path"] = history_path

    session_store = SessionStore()
    memory = Memory()
    registry = ToolRegistry()
    discover_and_register(registry)

    agent = AIAgent(
        config=config,
        tool_registry=registry,
        session=session_store,
        memory=memory,
        session_id=session_id_arg,
    )

    exit_code = 0

    # --- Piped stdin: read and process as single-shot (Bug #3) ---
    if command is None and not positional and not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()
        if piped_input:
            exit_code = _single_shot(agent, piped_input)
        session_store.close()
        sys.exit(exit_code or 0)

    try:
        if command == "train" and command_args:
            exit_code = _shortcut_train(agent, command_args)
        elif command == "eval" and command_args:
            exit_code = _shortcut_eval(agent, command_args)
        elif command == "search" and command_args:
            exit_code = _shortcut_search(agent, command_args)
        elif command == "download" and command_args:
            exit_code = _shortcut_download(agent, command_args)
        elif command == "clone" and command_args:
            exit_code = _shortcut_clone(agent, command_args)
        elif command == "infer" and command_args:
            exit_code = _shortcut_infer(agent, command_args)
        elif command == "ui" and command_args:
            exit_code = _shortcut_ui(agent, command_args)
        elif command == "monitor" and command_args:
            exit_code = _shortcut_monitor(agent, command_args)
        elif positional:
            message = " ".join(positional)
            exit_code = _single_shot(agent, message)
        else:
            exit_code = _repl(agent)
    finally:
        session_store.close()

    sys.exit(exit_code or 0)


def _print_help() -> None:
    """Print help text (not auto-generated, to avoid argparse issues)."""
    from pico import __version__ as _v
    print(
        "╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  Pico Agent v{_v} — AI-powered coding & detection assistant  ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
        "\n"
        "USAGE\n"
        "  pico-agent [flags] [command ... | message]\n"
        "\n"
        "FLAGS\n"
        "  -h, --help              Show this help message and exit\n"
        "  -V, --version           Show version and exit\n"
        "  -s, --session SESSION   Resume a specific session by ID\n"
        "  -v, --verbose           Enable debug logging (default: warnings only)\n"
        "  -c, --config CONFIG     Path to a YAML config file (default: ~/.pico-agent/config.yaml)\n"
        "\n"
        "QUICK START\n"
        "  pico-agent                              Start interactive REPL\n"
        '  pico-agent "what is the weather?"       Single-shot question\n'
        "  pico-agent train ./datasets/traffic     Auto-detect & train a model\n"
        "  pico-agent train ./data --model yolov8s --dry-run\n"
        "  pico-agent eval runs/train/weights/best.pt\n"
        '  pico-agent search "RT-DETR" --source github\n'
        '  pico-agent download "coco 2017" --source huggingface\n'
        "  pico-agent clone https://github.com/user/repo --install\n"
        "  pico-agent infer best.pt image.jpg\n"
        "  pico-agent ui                    Launch web dashboard\n"
        "  pico-agent monitor               Launch training monitor\n"
        "\n"
        "SHORTCUT COMMANDS\n"
        "  train <dir>       Auto-detect dataset format and start YOLO training\n"
        "                    Options: --model, --epochs, --batch, --dry-run\n"
        "  eval <model>      Evaluate a trained model and show mAP / PR curves\n"
        "                    Options: --data <data.yaml>\n"
        "  search <query>    Search GitHub repos and HuggingFace datasets\n"
        "                    Options: --source github|huggingface|kaggle|all\n"
        "  download <query>  Search, download, and verify a dataset\n"
        "                    Options: --source huggingface|kaggle|roboflow|url\n"
        "  clone <url>       Clone a Git repository (pulls if already cloned)\n"
        "                    Options: --install to auto-detect and install deps\n"
        "  infer <model> <img>  Run inference on an image with a .pt model\n"
        "  ui                 Launch web dashboard to visualize sessions\n"
        "                     Options: --port, --host, --no-open\n"
        "  monitor            Launch training monitor (scan for YOLO runs)\n"
        "                     Options: --port, --scan <dirs>, --no-open\n"
        "\n"
        "REPL SHORTCUTS (inside interactive mode)\n"
        "  /help             Show in-REPL help\n"
        "  /new              Start a new conversation session\n"
        "  /sessions         List recent sessions\n"
        "  /memory           Show persistent cross-session memory\n"
        "  /memory add <txt> Add an entry to persistent memory\n"
        "  /memory rm <key>  Remove memory entries matching keyword\n"
        "  /ui [port]        Launch web dashboard (default port: 8765)\n"
        "  /quit, /exit      Exit the REPL (also: Ctrl+D)\n"
        "\n"
        "CONFIGURATION\n"
        "  Config file : ~/.pico-agent/config.yaml\n"
        "  Environment : PICO_API_KEY, PICO_MODEL, PICO_BASE_URL, PICO_PROVIDER\n"
        "  Run `pico-agent --help` or visit https://hermes-agent.nousresearch.com for docs\n"
    )


def _parse_command_args(command: str, args: list[str]) -> argparse.Namespace:
    """Parse subcommand-specific arguments."""
    parser = argparse.ArgumentParser(prog=f"pico-agent {command}")

    if command == "train":
        parser.add_argument("data_dir", help="Path to dataset directory")
        parser.add_argument("--model", "-m", default="", help="YOLO model (e.g. yolov8n, yolov8s)")
        parser.add_argument("--epochs", "-e", type=int, default=0, help="Training epochs (0=auto)")
        parser.add_argument("--batch", "-b", type=int, default=0, help="Batch size (0=auto)")
        parser.add_argument("--dry-run", "-n", action="store_true", help="Show plan without training")
    elif command == "eval":
        parser.add_argument("model_path", help="Path to .pt model weights")
        parser.add_argument("--data", "-d", default="", help="Path to data.yaml")
    elif command == "search":
        parser.add_argument("query", help="Search query")
        parser.add_argument("--source", choices=["github", "huggingface", "kaggle", "all"], default="all")
    elif command == "download":
        parser.add_argument("query_or_url", help="Search query or direct URL")
        parser.add_argument("--source", choices=["huggingface", "kaggle", "roboflow", "url"], default="huggingface")
    elif command == "clone":
        parser.add_argument("url", help="Git URL")
        parser.add_argument("--install", "-i", action="store_true", help="Auto-install dependencies")
        parser.add_argument("--browse", "-b", action="store_true", help="Show directory structure after clone")
    elif command == "infer":
        parser.add_argument("model_path", help="Path to .pt model weights")
        parser.add_argument("image_path", help="Path to image file")
    elif command == "ui":
        parser.add_argument("--port", "-p", type=int, default=8765, help="Port number (default: 8765)")
        parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
        parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    elif command == "monitor":
        parser.add_argument("--port", "-p", type=int, default=8766, help="Port number (default: 8766)")
        parser.add_argument("--scan", nargs="*", help="Directories to scan for training runs")
        parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
