"""Tests for file tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pico.tools.file_tools import read_file, search_files, write_file


class TestReadFile:
    """test_read_file — write a file, read it back via tool."""

    def test_read_file(self, tmp_path: Path) -> None:
        fpath = tmp_path / "hello.txt"
        fpath.write_text("Hello, World!\nLine 2\n", encoding="utf-8")

        result = json.loads(read_file(str(fpath)))
        assert result["success"] is True
        assert result["content"] == "Hello, World!\nLine 2\n"
        assert result["line_count"] == 2
        assert result["size"] > 0

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        result = json.loads(read_file(str(tmp_path / "nope.txt")))
        assert result["success"] is False
        assert "not found" in result["error"].lower() or "File not found" in result["error"]

    def test_read_file_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "subdir"
        d.mkdir()
        result = json.loads(read_file(str(d)))
        assert result["success"] is False


class TestWriteFile:
    """test_write_file — write via tool, verify on disk."""

    def test_write_file(self, tmp_path: Path) -> None:
        fpath = tmp_path / "output.txt"
        result = json.loads(write_file(str(fpath), "written content"))
        assert result["success"] is True
        assert fpath.read_text(encoding="utf-8") == "written content"

    def test_write_file_append(self, tmp_path: Path) -> None:
        fpath = tmp_path / "append.txt"
        fpath.write_text("first\n", encoding="utf-8")
        result = json.loads(write_file(str(fpath), "second\n", append=True))
        assert result["success"] is True
        assert fpath.read_text(encoding="utf-8") == "first\nsecond\n"

    def test_write_file_creates_dirs(self, tmp_path: Path) -> None:
        fpath = tmp_path / "a" / "b" / "c" / "file.txt"
        result = json.loads(write_file(str(fpath), "deep"))
        assert result["success"] is True
        assert fpath.read_text(encoding="utf-8") == "deep"


class TestSearchFilesContent:
    """test_search_files_content — write files with content, search for pattern."""

    def test_search_files_content(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("import os\nprint('hello')\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("import sys\nprint('world')\n", encoding="utf-8")
        (tmp_path / "c.txt").write_text("no python here\n", encoding="utf-8")

        result = json.loads(search_files(str(tmp_path), grep="import"))
        assert result["success"] is True
        assert result["count"] == 2
        paths = {r["path"] for r in result["results"]}
        assert str(tmp_path / "a.py") in paths
        assert str(tmp_path / "b.py") in paths

    @pytest.mark.parametrize(
        "pattern, expected_count",
        [
            ("*.py", 2),
            ("*.txt", 1),
            ("*", 3),
        ],
    )
    def test_search_files_glob(self, tmp_path: Path, pattern: str, expected_count: int) -> None:
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.py").write_text("", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")

        result = json.loads(search_files(str(tmp_path), pattern=pattern))
        assert result["success"] is True
        assert result["count"] == expected_count
