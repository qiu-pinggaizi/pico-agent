"""TokenJuice — rule-based tool output compression.

Inspired by OpenHuman's TokenJuice: before any tool result reaches the LLM,
it passes through a rule overlay that strips noise and keeps signal.
HTML→Markdown, verbose logs truncated, duplicate lines removed, etc.

Rules are JSON files that merge in order:
  1. Builtin (shipped with pico-agent)
  2. User (~/.pico-agent/tokenjuice/rules/)
  3. Project (.pico-agent/tokenjuice/rules/)

Each rule matches a tool name pattern and applies reduction strategies.
"""

from __future__ import annotations

import json
import logging
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Builtin rules — sensible defaults for common tool outputs
# ---------------------------------------------------------------------------

BUILTIN_RULES: list[dict[str, Any]] = [
    {
        "name": "yolo_train_log",
        "match": ["remote_terminal"],
        "description": "Compress verbose YOLO training logs",
        "strategies": [
            {"type": "truncate_lines", "max_lines": 80},
            {"type": "drop_pattern", "patterns": [
                r"^\s*$",           # empty lines
                r"^\s*[\|=─┐└│├]",  # box-drawing / progress bars
                r"^Epoch\s+\d+/\d+.*$",  # per-epoch lines (keep summary only)
            ]},
            {"type": "dedup_lines"},
            {"type": "tail", "lines": 30},
        ],
    },
    {
        "name": "pip_install",
        "match": ["remote_terminal"],
        "description": "Compress pip install output",
        "strategies": [
            {"type": "keep_pattern", "patterns": [
                r"(Successfully|ERROR|WARNING|Requirement already|Installing|Collecting)",
                r"(Successfully installed|Failed to build)",
            ]},
            {"type": "truncate_lines", "max_lines": 30},
        ],
    },
    {
        "name": "git_status",
        "match": ["terminal"],
        "description": "Compress git status output",
        "strategies": [
            {"type": "truncate_lines", "max_lines": 50},
            {"type": "dedup_lines"},
        ],
    },
    {
        "name": "file_listing",
        "match": ["terminal"],
        "description": "Compress long file listings",
        "strategies": [
            {"type": "truncate_lines", "max_lines": 60},
        ],
    },
    {
        "name": "generic_long_output",
        "match": ["*"],
        "description": "Catch-all for very long outputs",
        "strategies": [
            {"type": "truncate_chars", "max_chars": 8000},
        ],
    },
]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _apply_truncate_lines(text: str, max_lines: int) -> str:
    """Keep first N and last N//3 lines, with a marker in between."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = lines[:max_lines * 2 // 3]
    tail = lines[-(max_lines // 3):]
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join(head) + f"\n... [{omitted} lines omitted] ...\n" + "\n".join(tail)


def _apply_truncate_chars(text: str, max_chars: int) -> str:
    """Truncate to max_chars with a marker."""
    if len(text) <= max_chars:
        return text
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars // 3
    return text[:head_chars] + f"\n... [{len(text) - max_chars} chars omitted] ...\n" + text[-tail_chars:]


def _apply_drop_pattern(text: str, patterns: list[str]) -> str:
    """Remove lines matching any of the regex patterns."""
    compiled = [re.compile(p) for p in patterns]
    lines = text.splitlines()
    kept = [line for line in lines if not any(c.search(line) for c in compiled)]
    return "\n".join(kept)


def _apply_keep_pattern(text: str, patterns: list[str]) -> str:
    """Keep ONLY lines matching at least one of the regex patterns (plus context)."""
    compiled = [re.compile(p) for p in patterns]
    lines = text.splitlines()
    kept = [line for line in lines if any(c.search(line) for c in compiled)]
    if not kept:
        return text  # fallback: if nothing matches, keep original
    return "\n".join(kept)


def _apply_dedup_lines(text: str) -> str:
    """Remove consecutive duplicate lines."""
    lines = text.splitlines()
    if not lines:
        return text
    result = [lines[0]]
    for line in lines[1:]:
        if line != result[-1]:
            result.append(line)
    return "\n".join(result)


def _apply_tail(text: str, lines: int) -> str:
    """Keep only the last N lines."""
    all_lines = text.splitlines()
    if len(all_lines) <= lines:
        return text
    return "\n".join(all_lines[-lines:])


def _apply_head(text: str, lines: int) -> str:
    """Keep only the first N lines."""
    all_lines = text.splitlines()
    if len(all_lines) <= lines:
        return text
    return "\n".join(all_lines[:lines]) + f"\n... [{len(all_lines) - lines} more lines]"


_STRATEGY_MAP = {
    "truncate_lines": lambda text, cfg: _apply_truncate_lines(text, cfg.get("max_lines", 80)),
    "truncate_chars": lambda text, cfg: _apply_truncate_chars(text, cfg.get("max_chars", 8000)),
    "drop_pattern": lambda text, cfg: _apply_drop_pattern(text, cfg.get("patterns", [])),
    "keep_pattern": lambda text, cfg: _apply_keep_pattern(text, cfg.get("patterns", [])),
    "dedup_lines": lambda text, cfg: _apply_dedup_lines(text),
    "tail": lambda text, cfg: _apply_tail(text, cfg.get("lines", 30)),
    "head": lambda text, cfg: _apply_head(text, cfg.get("lines", 30)),
}


# ---------------------------------------------------------------------------
# Rule loading and matching
# ---------------------------------------------------------------------------

def _load_user_rules() -> list[dict[str, Any]]:
    """Load user and project rules from disk."""
    rules: list[dict[str, Any]] = []
    for rule_dir in [
        Path.home() / ".pico-agent" / "tokenjuice" / "rules",
        Path(".pico-agent") / "tokenjuice" / "rules",
    ]:
        if rule_dir.exists():
            for f in sorted(rule_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        rules.extend(data)
                    elif isinstance(data, dict):
                        rules.append(data)
                except Exception as e:
                    logger.warning("Failed to load tokenjuice rule %s: %s", f, e)
    return rules


def _find_matching_rule(tool_name: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the first rule whose match patterns include the tool name."""
    for rule in reversed(rules):  # later rules override earlier
        patterns = rule.get("match", [])
        for pattern in patterns:
            if fnmatch(tool_name, pattern):
                return rule
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TokenJuice:
    """Rule-based tool output compressor.

    Call ``compress(tool_name, output)`` on every tool result before
    it enters the LLM context. Rules are loaded once on init.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        # Merge: builtin < user < project (later overrides)
        self.rules: list[dict[str, Any]] = BUILTIN_RULES + _load_user_rules()
        logger.debug("TokenJuice loaded %d rules", len(self.rules))

    def compress(self, tool_name: str, output: str) -> str:
        """Compress tool output using matching rules.

        Args:
            tool_name: Name of the tool that produced the output.
            output: Raw tool output string.

        Returns:
            Compressed output (or original if no rule matched / compression disabled).
        """
        if not self.enabled or not output:
            return output

        # Skip small outputs (no point compressing)
        if len(output) < 500:
            return output

        rule = _find_matching_rule(tool_name, self.rules)
        if not rule:
            return output

        original_len = len(output)
        for strategy in rule.get("strategies", []):
            stype = strategy.get("type", "")
            handler = _STRATEGY_MAP.get(stype)
            if handler:
                try:
                    output = handler(output, strategy)
                except Exception as e:
                    logger.warning("TokenJuice strategy %s failed: %s", stype, e)

        if len(output) < original_len:
            logger.debug(
                "TokenJuice: %s %d→%d chars (%.0f%% reduction)",
                tool_name, original_len, len(output),
                (1 - len(output) / original_len) * 100 if original_len else 0,
            )

        return output

    def add_rule(self, rule: dict[str, Any]) -> None:
        """Add a rule at runtime (highest priority)."""
        self.rules.append(rule)

    def get_rules_summary(self) -> str:
        """Return a human-readable summary of active rules."""
        lines = ["Active TokenJuice rules:"]
        for r in self.rules:
            match = ", ".join(r.get("match", ["*"]))
            desc = r.get("description", "no description")
            strats = [s.get("type", "?") for s in r.get("strategies", [])]
            lines.append(f"  [{match}] {desc} → {', '.join(strats)}")
        return "\n".join(lines)
