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
from typing import Any

logger = logging.getLogger(__name__)


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

def _handle_slash_command(cmd: str, agent: Any) -> str | None:
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
            "- `/sessions` — List recent sessions\n\n"
            "**Memory:**\n"
            "- `/memory` — Show persistent memory\n"
            "- `/memory add <text>` — Add a memory entry\n"
            "- `/memory rm <keyword>` — Remove memory entries\n\n"
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

def _shortcut_train(agent: Any, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent train <data_dir> [--model X] [--epochs N] [--dry-run]"""
    data_dir = args.data_dir
    err = _validate_dir(data_dir, "Dataset directory")
    if err:
        print(f"Error: {err}", file=sys.stderr)
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
            print(f"Error: {data.get('error', 'Unknown error')}", file=sys.stderr)
            return 1
    except json.JSONDecodeError:
        print(result)
        return 0


def _shortcut_eval(agent: Any, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent eval <model_path> [--data data.yaml]"""
    err = _validate_file(args.model_path, "Model file")
    if err:
        print(f"Error: {err}", file=sys.stderr)
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
            print(f"Error: {data.get('error', 'Unknown error')}", file=sys.stderr)
            return 1
    except json.JSONDecodeError:
        print(result)
        return 0


def _shortcut_search(agent: Any, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent search <query> [--source github|huggingface|all]"""
    source = args.source or "all"
    if source == "all":
        prompt = f"Search for '{args.query}' across GitHub repos and HuggingFace datasets. Show me the top results."
    elif source == "github":
        prompt = f"Search GitHub for '{args.query}' repositories. Show top results with stars and clone URLs."
    else:
        prompt = f"Search {source} for '{args.query}' datasets. Show me the results."
    return _single_shot(agent, prompt)


def _shortcut_download(agent: Any, args: argparse.Namespace) -> int:
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


def _shortcut_clone(agent: Any, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent clone <url> [--install]"""
    # Direct tool call
    tool_args: dict[str, Any] = {"url": args.url}
    result = agent.tools.dispatch("code_clone", tool_args)
    try:
        data = json.loads(result)
        if not data.get("success"):
            print(f"Error: {data.get('error', 'Unknown error')}", file=sys.stderr)
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


def _shortcut_infer(agent: Any, args: argparse.Namespace) -> int:
    """Shortcut: pico-agent infer <model_path> <image_path>"""
    err = _validate_file(args.model_path, "Model file")
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    err = _validate_file(args.image_path, "Image file")
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    prompt = f"Run inference with model {args.model_path} on image {args.image_path}. Show the results."
    return _single_shot(agent, prompt)


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------

def _single_shot(agent: Any, message: str) -> int:
    """Run a single message through the LLM and print the response.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        response = agent.run(message)
        _rich_print(response)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------

def _repl(agent: Any) -> int:
    """Interactive REPL with prompt_toolkit."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        history_path = agent.config.data.get("_history_path", "")
        history = FileHistory(history_path) if history_path else None
        session = PromptSession(history=history)
    except ImportError:
        session = None

    print("Pico Agent — type /help for commands, Ctrl+D to exit\n")

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
            response = agent.run(user_input)
            _rich_print(response)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            logger.exception("Agent error")
            print(f"Error: {e}", file=sys.stderr)


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

    KNOWN_COMMANDS = {"train", "eval", "search", "download", "clone", "infer"}

    # --- Manual global flag extraction (avoids argparse subparser issues) ---
    session_id_arg = ""
    verbose = False
    config_path = None
    positional: list[str] = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--help", "-h"):
            # If a command follows, let the command parser handle --help
            # (e.g. `pico-agent train --help` → show train help, not global help)
            rest_after = argv[i + 1:]
            if positional and positional[0] in KNOWN_COMMANDS:
                positional.append(a)
                i += 1
            elif rest_after and rest_after[0] in KNOWN_COMMANDS:
                positional.append(a)
                i += 1
            else:
                _print_help()
                sys.exit(0)
        elif a in ("--session", "-s") and i + 1 < len(argv):
            session_id_arg = argv[i + 1]; i += 2
        elif a.startswith("--session="):
            session_id_arg = a.split("=", 1)[1]; i += 1
        elif a in ("--verbose", "-v"):
            verbose = True; i += 1
        elif a in ("--config", "-c") and i + 1 < len(argv):
            config_path = argv[i + 1]; i += 2
        elif a.startswith("--config="):
            config_path = a.split("=", 1)[1]; i += 1
        elif a == "--":
            positional.extend(argv[i + 1:]); break
        else:
            positional.append(a); i += 1

    _setup_logging(verbose)

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

    config = load_config(config_path)

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
    print(
        "usage: pico-agent [-h] [-s SESSION] [-v] [-c CONFIG] [command ... | message]\n\n"
        "Pico Agent — AI agent with tool use, dataset/code download, and detection\n"
        "training\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  -s, --session SESSION Resume a specific session by ID\n"
        "  -v, --verbose         Enable debug logging\n"
        "  -c, --config CONFIG   Path to config YAML file\n\n"
        "Shortcut commands:\n"
        "  train    Auto-detect dataset and start training\n"
        "  eval     Evaluate a trained model\n"
        "  search   Search GitHub / HuggingFace\n"
        "  download Search and download a dataset\n"
        "  clone    Clone a Git repository\n"
        "  infer    Run inference on an image\n\n"
        "Examples:\n"
        "  pico-agent                              # Interactive REPL\n"
        '  pico-agent "what\'s 2+2?"                # Single-shot\n'
        "  pico-agent train ./datasets/traffic     # Quick train\n"
        "  pico-agent train ./data --model yolov8s --dry-run\n"
        "  pico-agent eval runs/train/weights/best.pt\n"
        '  pico-agent search "RT-DETR" --source github\n'
        '  pico-agent download "coco 2017" --source huggingface\n'
        "  pico-agent clone https://github.com/user/repo --install\n"
        "  pico-agent infer best.pt image.jpg"
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

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
