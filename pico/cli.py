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
import logging
import sys
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
# Shortcut commands — bypass REPL, directly invoke tools
# ---------------------------------------------------------------------------

def _shortcut_train(agent: Any, args: argparse.Namespace) -> None:
    """Shortcut: pico-agent train <data_dir> [--model X] [--epochs N] [--dry-run]"""
    data_dir = args.data_dir
    prompt_parts = [f"Train a detection model on the dataset at {data_dir}."]
    if args.model:
        prompt_parts.append(f"Use model {args.model}.")
    if args.epochs:
        prompt_parts.append(f"Train for {args.epochs} epochs.")
    if args.batch:
        prompt_parts.append(f"Batch size {args.batch}.")
    if args.dry_run:
        prompt_parts.append("Show me the training plan first (dry run), don't start yet.")
    else:
        prompt_parts.append("Start training now.")

    prompt = " ".join(prompt_parts)
    _single_shot(agent, prompt)


def _shortcut_eval(agent: Any, args: argparse.Namespace) -> None:
    """Shortcut: pico-agent eval <model_path> [--data data.yaml]"""
    prompt = f"Evaluate the model at {args.model_path}."
    if args.data:
        prompt += f" Use data config {args.data}."
    prompt += " Show mAP, precision, recall, and any bad cases."
    _single_shot(agent, prompt)


def _shortcut_search(agent: Any, args: argparse.Namespace) -> None:
    """Shortcut: pico-agent search <query> [--source github|huggingface|all]"""
    source = args.source or "all"
    if source == "all":
        prompt = f"Search for '{args.query}' across GitHub repos and HuggingFace datasets. Show me the top results."
    elif source == "github":
        prompt = f"Search GitHub for '{args.query}' repositories. Show top results with stars and clone URLs."
    else:
        prompt = f"Search {source} for '{args.query}' datasets. Show me the results."
    _single_shot(agent, prompt)


def _shortcut_download(agent: Any, args: argparse.Namespace) -> None:
    """Shortcut: pico-agent download <query_or_url> [--source huggingface|kaggle|url]"""
    source = args.source or "huggingface"
    if args.query_or_url.startswith("http"):
        prompt = f"Download this dataset: {args.query_or_url}"
    else:
        prompt = (
            f"Find and download a dataset matching '{args.query_or_url}' from {source}. "
            "Search first, then download the best match."
        )
    _single_shot(agent, prompt)


def _shortcut_clone(agent: Any, args: argparse.Namespace) -> None:
    """Shortcut: pico-agent clone <url> [--install]"""
    prompt = f"Clone the repository at {args.url}."
    if args.install:
        prompt += " After cloning, auto-detect and install its dependencies."
    if args.browse:
        prompt += " Then show me the directory structure."
    _single_shot(agent, prompt)


def _shortcut_infer(agent: Any, args: argparse.Namespace) -> None:
    """Shortcut: pico-agent infer <model_path> <image_path>"""
    prompt = f"Run inference with model {args.model_path} on image {args.image_path}. Show the results."
    _single_shot(agent, prompt)


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------

def _single_shot(agent: Any, message: str) -> None:
    """Run a single message and print the response."""
    try:
        response = agent.run(message)
        _rich_print(response)
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------

def _repl(agent: Any) -> None:
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
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

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
                break
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

    Supports both interactive REPL and shortcut commands.

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
    parser = argparse.ArgumentParser(
        prog="pico-agent",
        description="Pico Agent — AI agent with tool use, dataset/code download, and detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Shortcut commands:\n"
            "  train    Auto-detect dataset and start training\n"
            "  eval     Evaluate a trained model\n"
            "  search   Search GitHub / HuggingFace\n"
            "  download Search and download a dataset\n"
            "  clone    Clone a Git repository\n"
            "  infer    Run inference on an image\n"
        ),
    )
    parser.add_argument("--session", "-s", help="Resume a specific session by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", "-c", help="Path to config YAML file")

    subparsers = parser.add_subparsers(dest="command")

    # train
    p_train = subparsers.add_parser("train", help="Auto-detect dataset and start training")
    p_train.add_argument("data_dir", help="Path to dataset directory")
    p_train.add_argument("--model", "-m", default="", help="YOLO model (e.g. yolov8n, yolov8s)")
    p_train.add_argument("--epochs", "-e", type=int, default=0, help="Training epochs (0=auto)")
    p_train.add_argument("--batch", "-b", type=int, default=0, help="Batch size (0=auto)")
    p_train.add_argument("--dry-run", "-n", action="store_true", help="Show plan without training")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("model_path", help="Path to .pt model weights")
    p_eval.add_argument("--data", "-d", default="", help="Path to data.yaml")

    # search
    p_search = subparsers.add_parser("search", help="Search GitHub / HuggingFace")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--source", choices=["github", "huggingface", "kaggle", "all"], default="all")

    # download
    p_download = subparsers.add_parser("download", help="Search and download a dataset")
    p_download.add_argument("query_or_url", help="Search query or direct URL")
    p_download.add_argument("--source", choices=["huggingface", "kaggle", "roboflow", "url"], default="huggingface")

    # clone
    p_clone = subparsers.add_parser("clone", help="Clone a Git repository")
    p_clone.add_argument("url", help="Git URL")
    p_clone.add_argument("--install", "-i", action="store_true", help="Auto-install dependencies")
    p_clone.add_argument("--browse", "-b", action="store_true", help="Show directory structure after clone")

    # infer
    p_infer = subparsers.add_parser("infer", help="Run inference on an image")
    p_infer.add_argument("model_path", help="Path to .pt model weights")
    p_infer.add_argument("image_path", help="Path to image file")

    # "message" for bare positional (backward compat)
    parser.add_argument("message", nargs="?", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    # Lazy imports to keep startup fast
    from pico.agent import AIAgent
    from pico.config import CONFIG_DIR, load_config
    from pico.memory import Memory
    from pico.session import SessionStore
    from pico.tools import discover_and_register
    from pico.tools.registry import ToolRegistry

    config = load_config(args.config)

    # Set up history file
    history_path = str(CONFIG_DIR / "history")
    config.data["_history_path"] = history_path

    session_store = SessionStore()
    memory = Memory()
    registry = ToolRegistry()
    discover_and_register(registry)

    session_id = args.session or ""

    agent = AIAgent(
        config=config,
        tool_registry=registry,
        session=session_store,
        memory=memory,
        session_id=session_id,
    )

    # Handle shortcut commands
    if args.command == "train":
        _shortcut_train(agent, args)
    elif args.command == "eval":
        _shortcut_eval(agent, args)
    elif args.command == "search":
        _shortcut_search(agent, args)
    elif args.command == "download":
        _shortcut_download(agent, args)
    elif args.command == "clone":
        _shortcut_clone(agent, args)
    elif args.command == "infer":
        _shortcut_infer(agent, args)
    elif args.message:
        _single_shot(agent, args.message)
    else:
        _repl(agent)

    session_store.close()


if __name__ == "__main__":
    main()
