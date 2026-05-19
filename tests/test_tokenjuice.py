"""Tests for TokenJuice — rule-based tool output compression."""

import pytest

from pico.tokenjuice import (
    TokenJuice,
    _apply_dedup_lines,
    _apply_drop_pattern,
    _apply_keep_pattern,
    _apply_tail,
    _apply_truncate_chars,
    _apply_truncate_lines,
    _find_matching_rule,
    BUILTIN_RULES,
)


class TestTruncateLines:
    def test_short_text_unchanged(self):
        text = "line1\nline2\nline3"
        assert _apply_truncate_lines(text, 10) == text

    def test_long_text_truncated(self):
        lines = [f"line{i}" for i in range(100)]
        text = "\n".join(lines)
        result = _apply_truncate_lines(text, 20)
        assert "lines omitted" in result
        # Should have fewer lines than original
        assert len(result.splitlines()) < 100

    def test_preserves_head_and_tail(self):
        lines = [f"line{i}" for i in range(100)]
        text = "\n".join(lines)
        result = _apply_truncate_lines(text, 20)
        assert "line0" in result  # head preserved
        assert "line99" in result  # tail preserved


class TestTruncateChars:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert _apply_truncate_chars(text, 100) == text

    def test_long_text_truncated(self):
        text = "a" * 10000
        result = _apply_truncate_chars(text, 5000)
        assert "chars omitted" in result
        assert len(result) < 10000


class TestDropPattern:
    def test_drops_matching_lines(self):
        text = "keep this\n  \nalso keep\n\nkeep too"
        result = _apply_drop_pattern(text, [r"^\s*$"])
        assert result == "keep this\nalso keep\nkeep too"

    def test_no_match_unchanged(self):
        text = "line1\nline2"
        assert _apply_drop_pattern(text, [r"^xyz$"]) == text


class TestKeepPattern:
    def test_keeps_matching_lines(self):
        text = "ERROR: bad\nnoise\nWARNING: caution\nmore noise"
        result = _apply_keep_pattern(text, [r"ERROR|WARNING"])
        assert "ERROR" in result
        assert "WARNING" in result
        assert "noise" not in result

    def test_no_match_returns_original(self):
        text = "line1\nline2"
        assert _apply_keep_pattern(text, [r"^xyz$"]) == text


class TestDedupLines:
    def test_removes_consecutive_duplicates(self):
        text = "a\na\nb\nb\nb\nc"
        assert _apply_dedup_lines(text) == "a\nb\nc"

    def test_no_dups_unchanged(self):
        text = "a\nb\nc"
        assert _apply_dedup_lines(text) == text


class TestTail:
    def test_short_text_unchanged(self):
        text = "a\nb\nc"
        assert _apply_tail(text, 10) == text

    def test_keeps_last_n(self):
        text = "a\nb\nc\nd\ne"
        result = _apply_tail(text, 2)
        assert result == "d\ne"


class TestRuleMatching:
    def test_wildcard_matches_any(self):
        rules = [{"match": ["*"], "strategies": []}]
        assert _find_matching_rule("anything", rules) is not None

    def test_exact_match(self):
        rules = [{"match": ["terminal"], "strategies": []}]
        assert _find_matching_rule("terminal", rules) is not None
        assert _find_matching_rule("other", rules) is None

    def test_glob_pattern(self):
        rules = [{"match": ["remote_*"], "strategies": []}]
        assert _find_matching_rule("remote_terminal", rules) is not None
        assert _find_matching_rule("terminal", rules) is None


class TestTokenJuice:
    def test_compress_small_output_unchanged(self):
        tj = TokenJuice()
        assert tj.compress("any_tool", "short") == "short"

    def test_compress_disabled(self):
        tj = TokenJuice(enabled=False)
        long_text = "x" * 10000
        assert tj.compress("any_tool", long_text) == long_text

    def test_compress_empty(self):
        tj = TokenJuice()
        assert tj.compress("any_tool", "") == ""

    def test_builtin_rules_loaded(self):
        tj = TokenJuice()
        assert len(tj.rules) >= len(BUILTIN_RULES)

    def test_add_rule(self):
        tj = TokenJuice()
        before = len(tj.rules)
        tj.add_rule({"match": ["test_*"], "strategies": []})
        assert len(tj.rules) == before + 1

    def test_get_rules_summary(self):
        tj = TokenJuice()
        summary = tj.get_rules_summary()
        assert "Active TokenJuice rules" in summary
        assert "wildcard" in summary.lower() or "*" in summary

    def test_compress_with_long_output(self):
        tj = TokenJuice()
        # Simulate a very long terminal output
        lines = [f"Processing file {i}: {'x' * 100}" for i in range(500)]
        long_output = "\n".join(lines)
        result = tj.compress("terminal", long_output)
        # Should be shorter due to truncation
        assert len(result) < len(long_output)
