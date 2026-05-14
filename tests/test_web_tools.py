"""Tests for web tools (mocked HTTP calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pico.tools.web_tools import web_extract, web_search


# ---------------------------------------------------------------------------
# Mock HTML snippets
# ---------------------------------------------------------------------------

DDG_HTML = """
<html>
<body>
<div class="result">
    <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">Example <b>Page</b> One</a>
    <a class="result__snippet">This is a snippet about example page one.</a>
</div>
<div class="result">
    <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage2">Example Page Two</a>
    <a class="result__snippet">Snippet for page two content.</a>
</div>
</body>
</html>
"""

SIMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<script>var x = 1;</script>
<style>body { color: red; }</style>
<h1>Hello World</h1>
<p>This is a <b>paragraph</b> with &amp; entities.</p>
<!-- hidden comment -->
</body>
</html>
"""


def _mock_httpx_post(html: str) -> MagicMock:
    """Return a mock httpx.post response."""
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


def _mock_httpx_get(html: str) -> MagicMock:
    """Return a mock httpx.get response."""
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_httpx_module() -> MagicMock:
    """Create a mock httpx module."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSearchParsing:
    """test_web_search_parsing — mock DuckDuckGo HTML, verify result parsing."""

    def test_web_search_parsing(self) -> None:
        mock_resp = _mock_httpx_post(DDG_HTML)
        mock_httpx = _make_mock_httpx_module()
        mock_httpx.post.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result_json = web_search("test query", num_results=5)

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["query"] == "test query"
        assert result["count"] == 2

        titles = [r["title"] for r in result["results"]]
        assert "Example Page One" in titles
        assert "Example Page Two" in titles

        # Verify URL decoding
        urls = [r["url"] for r in result["results"]]
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    def test_web_search_num_results_limit(self) -> None:
        mock_resp = _mock_httpx_post(DDG_HTML)
        mock_httpx = _make_mock_httpx_module()
        mock_httpx.post.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result_json = web_search("query", num_results=1)

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["count"] == 1

    def test_web_search_error(self) -> None:
        mock_httpx = _make_mock_httpx_module()
        mock_httpx.post.side_effect = Exception("network error")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result_json = web_search("query")

        result = json.loads(result_json)
        assert result["success"] is False
        assert "network error" in result["error"]


class TestWebExtractParsing:
    """test_web_extract_parsing — mock page HTML, verify text extraction."""

    def test_web_extract_parsing(self) -> None:
        mock_resp = _mock_httpx_get(SIMPLE_HTML)
        mock_httpx = _make_mock_httpx_module()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result_json = web_extract("https://example.com")

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["title"] == "Test Page"
        assert "Hello World" in result["content"]
        # Script/style should be stripped
        assert "var x" not in result["content"]
        assert "color: red" not in result["content"]
        # HTML entities should be decoded
        assert "&amp;" not in result["content"]
        # Comment should be stripped
        assert "hidden comment" not in result["content"]
        # Bold tag stripped
        assert "<b>" not in result["content"]

    def test_web_extract_truncation(self) -> None:
        long_html = "<html><head><title>Long</title></head><body>" + "x" * 30000 + "</body></html>"
        mock_resp = _mock_httpx_get(long_html)
        mock_httpx = _make_mock_httpx_module()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result_json = web_extract("https://example.com", max_chars=1000)

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["length"] <= 1050  # truncated + marker

    def test_web_extract_error(self) -> None:
        mock_httpx = _make_mock_httpx_module()
        mock_httpx.get.side_effect = Exception("timeout")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result_json = web_extract("https://example.com")

        result = json.loads(result_json)
        assert result["success"] is False
        assert "timeout" in result["error"]
