"""Tests for Memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from pico.memory import Memory, NoMemory


class TestMemorySaveAndLoad:
    """test_save_and_load — save entries, load them back."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        mem_path = tmp_path / "memory.md"
        mem = Memory(path=mem_path)
        mem.add("first fact")
        mem.add("second fact")
        content = mem.load()
        assert "first fact" in content
        assert "second fact" in content


class TestMemoryAddEntry:
    """test_add_entry — add memory, verify it appears."""

    def test_add_entry(self, tmp_path: Path) -> None:
        mem_path = tmp_path / "memory.md"
        mem = Memory(path=mem_path)
        mem.add("user prefers Python")
        content = mem.load()
        assert "user prefers Python" in content
        # Check timestamp format present
        assert "[" in content


class TestMemoryRemoveEntry:
    """test_remove_entry — add then remove, verify gone."""

    def test_remove_entry(self, tmp_path: Path) -> None:
        mem_path = tmp_path / "memory.md"
        mem = Memory(path=mem_path)
        mem.add("remember this")
        mem.add("forget this one")
        mem.add("keep this too")
        removed = mem.remove("forget this")
        assert removed is True
        content = mem.load()
        assert "remember this" in content
        assert "keep this too" in content
        assert "forget this one" not in content

    def test_remove_entry_no_match(self, tmp_path: Path) -> None:
        mem_path = tmp_path / "memory.md"
        mem = Memory(path=mem_path)
        mem.add("hello")
        removed = mem.remove("nonexistent")
        assert removed is False


class TestNoMemory:
    """test_no_memory — NoMemory always returns empty."""

    def test_no_memory(self) -> None:
        nm = NoMemory()
        assert nm.load() == ""
        nm.add("should be ignored")
        assert nm.load() == ""
        assert nm.remove("anything") is False
