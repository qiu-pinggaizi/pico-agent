"""Web tools — search the web and extract page content.

Uses httpx for HTTP requests and DuckDuckGo for search.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pico.tools.registry import ToolRegistry
from pico.tools.utils import _error, _success

logger = logging.getLogger(__name__)


def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return structured results.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return (default 5).

    Returns:
        JSON with keys: success, query, results (list of {title, url, snippet}).
    """
    try:
        import httpx

        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PicoAgent/0.1)",
        }

        resp = httpx.post(
            url,
            data={"q": query, "b": ""},
            headers=headers,
            timeout=15,
            follow_redirects=True,
        )
        resp.raise_for_status()
        html = resp.text

        # Parse results from DuckDuckGo HTML
        results: list[dict[str, str]] = []

        # Match result blocks: <a class="result__a" href="...">title</a>
        # and <a class="result__snippet" ...>snippet</a>
        title_pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )

        titles = title_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (href, title) in enumerate(titles[:num_results]):
            snippet = snippets[i][0] if i < len(snippets) else ""
            # Clean HTML tags from title and snippet
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()

            # Extract actual URL from DuckDuckGo redirect
            actual_url = href
            if "uddg=" in href:
                match = re.search(r"uddg=([^&]+)", href)
                if match:
                    from urllib.parse import unquote
                    actual_url = unquote(match.group(1))

            results.append({
                "title": clean_title,
                "url": actual_url,
                "snippet": clean_snippet,
            })

        logger.info("web_search(%s): %d results", query, len(results))
        return _success({"query": query, "results": results, "count": len(results)})

    except Exception as e:
        logger.error("web_search failed: %s", e)
        return _error(f"Search failed: {e}")


def web_extract(url: str, max_chars: int = 20000) -> str:
    """Fetch a web page and extract its text content.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return (default 20000).

    Returns:
        JSON with keys: success, url, title, content (text), length.
    """
    try:
        import httpx

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PicoAgent/0.1)",
        }

        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        # Strip HTML tags to get plain text
        # Remove script and style blocks first
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&nbsp;", " ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

        logger.info("web_extract(%s): %d chars", url, len(text))
        return _success({
            "url": url,
            "title": title,
            "content": text,
            "length": len(text),
        })

    except Exception as e:
        logger.error("web_extract failed for %s: %s", url, e)
        return _error(f"Failed to extract page: {e}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry: ToolRegistry) -> None:
    """Register web tools with the given registry."""
    registry.register(
        name="web_search",
        toolset="web",
        description="Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "num_results": {"type": "integer", "description": "Max results (default 5).", "default": 5},
            },
            "required": ["query"],
        },
        handler=web_search,
    )

    registry.register(
        name="web_extract",
        toolset="web",
        description="Fetch a web page URL and extract its text content as plain text.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
                "max_chars": {"type": "integer", "description": "Max characters to return (default 20000).", "default": 20000},
            },
            "required": ["url"],
        },
        handler=web_extract,
    )
