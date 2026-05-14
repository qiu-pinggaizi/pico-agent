"""Cross-session persistent memory stored in ~/.pico-agent/memory.md."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from pico.config import MEMORY_FILE

logger = logging.getLogger(__name__)


class Memory:
    """Persistent memory backed by a Markdown file.

    Stores user facts / preferences that persist across sessions
    and are injected into the system prompt.

    Args:
        path: Path to the memory markdown file.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else MEMORY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it doesn't exist
        if not self.path.exists():
            self.path.write_text("# Pico Agent Memory\n\n", encoding="utf-8")
            logger.debug("Created memory file at %s", self.path)

    def load(self) -> str:
        """Load the full memory content as a string.

        Returns:
            The raw markdown content, or empty string if file is missing.
        """
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8").strip()

    def add(self, content: str) -> None:
        """Append a new memory entry with a timestamp.

        Args:
            content: The memory text to add.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n- [{ts}] {content.strip()}\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Added memory entry: %s", content[:80])

    def remove(self, keyword: str) -> bool:
        """Remove all lines containing the keyword.

        Args:
            keyword: Substring to match against memory lines.

        Returns:
            True if at least one line was removed, False otherwise.
        """
        if not self.path.exists():
            return False

        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        kept = [line for line in lines if keyword.lower() not in line.lower()]

        if len(kept) == len(lines):
            return False

        self.path.write_text("".join(kept), encoding="utf-8")
        logger.info("Removed memory entries containing '%s'", keyword)
        return True


class NoMemory:
    """A no-op memory for delegation sub-agents that don't share memory."""

    def load(self) -> str:
        return ""

    def add(self, content: str) -> None:
        pass

    def remove(self, keyword: str) -> bool:
        return False
