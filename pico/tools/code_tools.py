"""Code tools — search, clone, and manage code repositories from GitHub and the web.

Supports GitHub API search, git clone, dependency auto-install, and code snippet search.
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


def _get_code_dir() -> Path:
    """Get the default code download directory."""
    from pico.config import get_config
    cfg = get_config()
    base = Path(cfg.base_dir) if cfg.base_dir else Path.home() / ".pico-agent"
    code_dir = base / "repos"
    code_dir.mkdir(parents=True, exist_ok=True)
    return code_dir


def _run_git(args: list[str], cwd: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a git command with error handling."""
    cmd = ["git"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)


def _detect_install_command(repo_path: Path) -> list[str] | None:
    """Detect the best install command for a repository."""
    # Priority order for dependency files
    checks = [
        (repo_path / "pyproject.toml", ["pip", "install", "-e", "."]),
        (repo_path / "setup.py", ["pip", "install", "-e", "."]),
        (repo_path / "setup.cfg", ["pip", "install", "-e", "."]),
        (repo_path / "requirements.txt", ["pip", "install", "-r", "requirements.txt"]),
        (repo_path / "Pipfile", ["pipenv", "install"]),
        (repo_path / "poetry.lock", ["poetry", "install"]),
        (repo_path / "package.json", ["npm", "install"]),
        (repo_path / "Cargo.toml", ["cargo", "build", "--release"]),
        (repo_path / "go.mod", ["go", "build", "./..."]),
        (repo_path / "Makefile", None),  # Need to check manually
        (repo_path / "CMakeLists.txt", None),
    ]

    for dep_file, cmd in checks:
        if dep_file.exists():
            if cmd:
                return cmd
            return None

    # Check for requirements*.txt files
    for f in repo_path.glob("requirements*.txt"):
        return ["pip", "install", "-r", f.name]

    return None


def _detect_language(repo_path: Path) -> str:
    """Detect the primary language of a repository."""
    indicators = {
        "Python": ["*.py", "pyproject.toml", "setup.py", "requirements.txt"],
        "JavaScript/TypeScript": ["*.js", "*.ts", "package.json"],
        "Rust": ["Cargo.toml", "*.rs"],
        "Go": ["go.mod", "*.go"],
        "C/C++": ["*.c", "*.cpp", "*.h", "CMakeLists.txt"],
        "Java": ["*.java", "pom.xml", "build.gradle"],
        "Julia": ["*.jl", "Project.toml"],
    }

    scores: dict[str, int] = {}
    for lang, patterns in indicators.items():
        for pat in patterns:
            count = len(list(repo_path.glob(pat)))
            if count > 0:
                scores[lang] = scores.get(lang, 0) + count

    if not scores:
        return "unknown"
    return max(scores, key=scores.get)  # type: ignore


def code_search(
    query: str,
    language: str = "",
    sort: str = "stars",
    max_results: int = 10,
) -> str:
    """Search for code repositories on GitHub.

    Args:
        query: Search query (e.g., "YOLO detection", "SAM segmentation").
        language: Filter by language (e.g., "python", "c++", "rust").
        sort: Sort by "stars", "forks", "updated" (default "stars").
        max_results: Max results (default 10, max 30).

    Returns:
        JSON with repository listings including stars, description, and clone URL.
    """
    import httpx

    try:
        gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        headers = {"Accept": "application/vnd.github+json"}
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"

        params: dict[str, Any] = {
            "q": query + (f" language:{language}" if language else ""),
            "sort": sort,
            "order": "desc",
            "per_page": min(max_results, 30),
        }

        resp = httpx.get(
            "https://api.github.com/search/repositories",
            params=params,
            headers=headers,
            timeout=15,
        )

        if resp.status_code == 403:
            return _error("GitHub API rate limited. Set GITHUB_TOKEN env var for higher limits.")
        if resp.status_code != 200:
            return _error(f"GitHub API returned {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        repos = []
        for item in data.get("items", []):
            repos.append({
                "full_name": item.get("full_name", ""),
                "description": (item.get("description") or "")[:200],
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language", ""),
                "topics": item.get("topics", [])[:5],
                "updated": item.get("updated_at", ""),
                "clone_url": item.get("clone_url", ""),
                "ssh_url": item.get("ssh_url", ""),
                "html_url": item.get("html_url", ""),
                "default_branch": item.get("default_branch", "main"),
                "size_kb": item.get("size", 0),
            })

        logger.info("code_search(%s): %d results", query, len(repos))
        return _success({
            "query": query,
            "total_count": data.get("total_count", 0),
            "results": repos,
            "count": len(repos),
        })

    except Exception as e:
        logger.error("code_search failed: %s", e)
        return _error(f"Code search failed: {e}")


def code_clone(
    url: str,
    output_dir: str = "",
    branch: str = "",
    depth: int = 0,
    include_submodules: bool = False,
) -> str:
    """Clone a Git repository.

    Args:
        url: Git URL (HTTPS or SSH), e.g. "https://github.com/user/repo.git".
        output_dir: Where to clone (default: ~/.pico-agent/repos/<repo-name>).
        branch: Specific branch to clone (default: repo's default branch).
        depth: Shallow clone depth (0 = full clone, 1 = latest only).
        include_submodules: Also clone git submodules.

    Returns:
        JSON with clone path, branch, and repo info.
    """
    try:
        # Derive repo name from URL
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        if not output_dir:
            output_dir = str(_get_code_dir() / repo_name)

        output_path = Path(output_dir)
        if output_path.exists():
            # Check if it's already the same repo
            existing_remote = _run_git(["config", "--get", "remote.origin.url"], cwd=str(output_path))
            if existing_remote.returncode == 0 and existing_remote.stdout.strip() == url:
                # Already cloned, do a pull
                pull_result = _run_git(["pull", "--ff-only"], cwd=str(output_path), timeout=120)
                return _success({
                    "action": "updated",
                    "path": output_dir,
                    "remote": url,
                    "pull_output": pull_result.stdout.strip(),
                })
            else:
                return _error(f"Directory already exists and has a different remote: {existing_remote.stdout.strip()}")

        # Build clone command
        cmd_args = ["clone"]
        if branch:
            cmd_args.extend(["--branch", branch])
        if depth > 0:
            cmd_args.extend(["--depth", str(depth)])
        if include_submodules:
            cmd_args.append("--recurse-submodules")
        cmd_args.extend([url, str(output_path)])

        result = _run_git(cmd_args, timeout=600)

        if result.returncode != 0:
            return _error(f"Git clone failed: {result.stderr.strip()}")

        # Gather repo info
        actual_branch = _run_git(["branch", "--show-current"], cwd=str(output_path))
        log_result = _run_git(["log", "--oneline", "-5"], cwd=str(output_path))

        # Detect language
        language = _detect_language(output_path)

        # List top-level files
        files = [str(f.relative_to(output_path)) for f in sorted(output_path.iterdir()) if not f.name.startswith(".git")]

        return _success({
            "action": "cloned",
            "path": str(output_path),
            "remote": url,
            "branch": actual_branch.stdout.strip() or branch or "unknown",
            "language": language,
            "recent_commits": log_result.stdout.strip().split("\n") if log_result.returncode == 0 else [],
            "top_level_files": files[:30],
        })

    except subprocess.TimeoutExpired:
        return _error("Git clone timed out (>600s). Try with depth=1 for a shallow clone.")
    except Exception as e:
        logger.error("code_clone failed: %s", e)
        return _error(f"Clone failed: {e}")


def code_install(repo_path: str, auto_detect: bool = True, extra_command: str = "") -> str:
    """Install dependencies for a cloned repository.

    Args:
        repo_path: Path to the repository.
        auto_detect: Auto-detect and run the correct install command (default True).
        extra_command: Additional shell command to run after install (optional).

    Returns:
        JSON with install status and output.
    """
    try:
        p = Path(repo_path).expanduser()
        if not p.exists():
            return _error(f"Path not found: {repo_path}")

        results: list[dict] = []

        if auto_detect:
            cmd = _detect_install_command(p)
            if cmd:
                logger.info("Auto-detected install command: %s", " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(p))
                results.append({
                    "command": " ".join(cmd),
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout[-2000:] if proc.stdout else "",
                    "stderr": proc.stderr[-1000:] if proc.stderr else "",
                    "success": proc.returncode == 0,
                })
            else:
                results.append({
                    "command": "(auto-detect)",
                    "success": False,
                    "note": "No recognized dependency file found. Specify extra_command manually.",
                })

        if extra_command:
            logger.info("Running extra command: %s", extra_command)
            proc = subprocess.run(extra_command, shell=True, capture_output=True, text=True, timeout=600, cwd=str(p))
            results.append({
                "command": extra_command,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-2000:] if proc.stdout else "",
                "stderr": proc.stderr[-1000:] if proc.stderr else "",
                "success": proc.returncode == 0,
            })

        all_ok = all(r.get("success") for r in results)
        return _success({
            "path": str(p),
            "language": _detect_language(p),
            "steps": results,
            "all_success": all_ok,
        })

    except subprocess.TimeoutExpired:
        return _error("Install command timed out (>600s)")
    except Exception as e:
        logger.error("code_install failed: %s", e)
        return _error(f"Install failed: {e}")


def code_browse(repo_path: str, target: str = ".", max_depth: int = 2) -> str:
    """Browse a repository's structure and optionally read specific files.

    Args:
        repo_path: Path to the repository.
        target: File or directory to inspect. "." for tree view, "README.md" to read a file.
        max_depth: Directory tree depth (default 2).

    Returns:
        JSON with directory tree or file content.
    """
    try:
        p = Path(repo_path).expanduser()
        if not p.exists():
            return _error(f"Path not found: {repo_path}")

        target_path = p / target if target != "." else p

        if not target_path.exists():
            return _error(f"Target not found: {target_path}")

        if target_path.is_file():
            # Read file content
            try:
                content = target_path.read_text(encoding="utf-8", errors="replace")
                if len(content) > 30000:
                    content = content[:30000] + "\n... [truncated]"
                return _success({
                    "type": "file",
                    "path": str(target_path.relative_to(p)),
                    "content": content,
                    "size": target_path.stat().st_size,
                    "lines": content.count("\n") + 1,
                })
            except Exception as e:
                return _error(f"Cannot read file: {e}")

        # Build directory tree
        tree_lines: list[str] = []
        _count = 0

        def _build_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            nonlocal _count
            if depth > max_depth or _count > 200:
                return

            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            # Skip hidden dirs and common large dirs
            skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}
            entries = [e for e in entries if e.name not in skip and not e.name.startswith(".")]

            for i, entry in enumerate(entries):
                _count += 1
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                ext = "/" if entry.is_dir() else ""
                tree_lines.append(f"{prefix}{connector}{entry.name}{ext}")
                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    _build_tree(entry, prefix + extension, depth + 1)

        tree_lines.append(p.name + "/")
        _build_tree(p)

        return _success({
            "type": "directory",
            "path": str(p),
            "language": _detect_language(p),
            "tree": "\n".join(tree_lines),
            "entries": _count,
        })

    except Exception as e:
        logger.error("code_browse failed: %s", e)
        return _error(f"Browse failed: {e}")


def code_exec(repo_path: str, command: str) -> str:
    """Execute a shell command inside a repository directory.

    Args:
        repo_path: Path to the repository.
        command: Shell command to run.

    Returns:
        JSON with stdout, stderr, and exit code.
    """
    try:
        p = Path(repo_path).expanduser()
        if not p.exists():
            return _error(f"Path not found: {repo_path}")

        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300, cwd=str(p))

        return _success({
            "command": command,
            "cwd": str(p),
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-10000:] if proc.stdout else "",
            "stderr": proc.stderr[-5000:] if proc.stderr else "",
            "success": proc.returncode == 0,
        })

    except subprocess.TimeoutExpired:
        return _error(f"Command timed out: {command}")
    except Exception as e:
        return _error(f"Exec failed: {e}")


def code_list() -> str:
    """List all locally cloned repositories.

    Returns:
        JSON with list of repos including path, language, and size.
    """
    try:
        code_dir = _get_code_dir()
        repos = []

        for item in sorted(code_dir.iterdir()):
            if item.is_dir() and (item / ".git").exists():
                # Get last commit
                log = _run_git(["log", "--oneline", "-1"], cwd=str(item))
                branch = _run_git(["branch", "--show-current"], cwd=str(item))

                repos.append({
                    "name": item.name,
                    "path": str(item),
                    "branch": branch.stdout.strip() if branch.returncode == 0 else "?",
                    "last_commit": log.stdout.strip() if log.returncode == 0 else "?",
                    "language": _detect_language(item),
                })

        return _success({
            "repo_dir": str(code_dir),
            "count": len(repos),
            "repos": repos,
        })

    except Exception as e:
        return _error(f"Failed to list repos: {e}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry: ToolRegistry) -> None:
    """Register code tools."""
    registry.register(
        name="code_search",
        toolset="web",
        description=(
            "Search GitHub for code repositories by topic/keyword. "
            "Returns repos with stars, description, and clone URLs. "
            "Set GITHUB_TOKEN env var for higher API rate limits."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'YOLO detection', 'SAM segmentation', 'diffusion model'."},
                "language": {"type": "string", "description": "Filter by language, e.g. 'python', 'c++'. Default: all.", "default": ""},
                "sort": {"type": "string", "description": "Sort by: 'stars', 'forks', 'updated'. Default 'stars'.", "default": "stars"},
                "max_results": {"type": "integer", "description": "Max results (default 10, max 30).", "default": 10},
            },
            "required": ["query"],
        },
        handler=code_search,
    )

    registry.register(
        name="code_clone",
        toolset="web",
        description=(
            "Clone a Git repository to local disk. "
            "Supports HTTPS/SSH URLs, branch selection, shallow clone, and submodules. "
            "If already cloned, does a git pull instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Git URL, e.g. 'https://github.com/user/repo.git'."},
                "output_dir": {"type": "string", "description": "Clone destination (default: ~/.pico-agent/repos/<name>).", "default": ""},
                "branch": {"type": "string", "description": "Specific branch (default: repo's default).", "default": ""},
                "depth": {"type": "integer", "description": "Shallow clone depth (0=full, 1=latest only). Default 0.", "default": 0},
                "include_submodules": {"type": "boolean", "description": "Clone submodules too. Default false.", "default": False},
            },
            "required": ["url"],
        },
        handler=code_clone,
    )

    registry.register(
        name="code_install",
        toolset="web",
        description=(
            "Auto-detect and install dependencies for a cloned repository. "
            "Detects: pyproject.toml, setup.py, requirements.txt, package.json, Cargo.toml, go.mod, etc. "
            "Use after code_clone to set up the project."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the repository."},
                "auto_detect": {"type": "boolean", "description": "Auto-detect install command (default true).", "default": True},
                "extra_command": {"type": "string", "description": "Additional command to run after auto-install.", "default": ""},
            },
            "required": ["repo_path"],
        },
        handler=code_install,
    )

    registry.register(
        name="code_browse",
        toolset="web",
        description=(
            "Browse a repository's directory tree or read a specific file. "
            "Use target='.' for tree view, target='README.md' to read a file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the repository."},
                "target": {"type": "string", "description": "'.' for tree, or a file path like 'README.md'.", "default": "."},
                "max_depth": {"type": "integer", "description": "Tree depth (default 2).", "default": 2},
            },
            "required": ["repo_path"],
        },
        handler=code_browse,
    )

    registry.register(
        name="code_exec",
        toolset="web",
        description=(
            "Execute a shell command inside a repository directory. "
            "Use for running tests, builds, training scripts, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the repository."},
                "command": {"type": "string", "description": "Shell command to run."},
            },
            "required": ["repo_path", "command"],
        },
        handler=code_exec,
    )

    registry.register(
        name="code_list",
        toolset="web",
        description="List all locally cloned repositories with their branch and last commit info.",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=code_list,
    )
