"""Lightweight HTTP server for Pico Agent dashboard.

Serves the SPA and provides REST API endpoints backed by SessionStore.
Uses only stdlib http.server — no extra dependencies.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

_DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


class _APIHandler(BaseHTTPRequestHandler):
    """Handle REST API requests and serve the dashboard SPA."""

    # Injected at server start
    db_path: str = ""

    def _get_store(self) -> "SessionStore":
        """Create a per-request SessionStore (SQLite connections are thread-bound)."""
        from pico.session import SessionStore
        return SessionStore(db_path=self.db_path)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress default stderr logging; use our logger."""
        logger.debug("HTTP %s", fmt % args)

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "" or path == "/":
            self._serve_html()
        elif path == "/api/sessions":
            self._api_list_sessions(qs)
        elif re.match(r"^/api/sessions/([^/]+)$", path):
            sid = path.split("/")[3]
            self._api_get_session(sid)
        elif re.match(r"^/api/sessions/([^/]+)/messages$", path):
            sid = path.split("/")[3]
            self._api_get_messages(sid)
        elif path == "/api/stats":
            self._api_stats()
        elif path == "/api/search":
            q = qs.get("q", [""])[0]
            self._api_search(q)
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        m = re.match(r"^/api/sessions/([^/]+)$", path)
        if m:
            self._api_delete_session(m.group(1))
        else:
            self._json_response({"error": "Not found"}, 404)

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------
    def _api_list_sessions(self, qs: dict) -> None:
        limit = int(qs.get("limit", ["50"])[0])
        store = self._get_store()
        sessions = store.list_sessions(limit=limit)
        data = [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]
        self._json_response(data)

    def _api_get_session(self, session_id: str) -> None:
        store = self._get_store()
        sess = store.get_session(session_id)
        if sess is None:
            self._json_response({"error": "Session not found"}, 404)
            return
        messages = store.get_messages(session_id)
        self._json_response({
            "session": {
                "id": sess.id,
                "title": sess.title,
                "created_at": sess.created_at,
                "updated_at": sess.updated_at,
            },
            "messages": [self._msg_dict(m) for m in messages],
        })

    def _api_get_messages(self, session_id: str) -> None:
        store = self._get_store()
        messages = store.get_messages(session_id)
        self._json_response([self._msg_dict(m) for m in messages])

    def _api_delete_session(self, session_id: str) -> None:
        store = self._get_store()
        store.delete_session(session_id)
        self._json_response({"success": True})

    def _api_stats(self) -> None:
        store = self._get_store()
        conn = store._get_conn()
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        tool_calls = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role = 'tool'"
        ).fetchone()[0]

        # Tool usage breakdown
        rows = conn.execute(
            "SELECT tool_calls_json FROM messages WHERE tool_calls_json IS NOT NULL"
        ).fetchall()
        tool_usage: dict[str, int] = {}
        for row in rows:
            try:
                tcs = json.loads(row[0])
                for tc in tcs:
                    name = tc.get("name", "unknown")
                    tool_usage[name] = tool_usage.get(name, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        # Recent activity (messages per day, last 7 days)
        daily_rows = conn.execute(
            "SELECT DATE(timestamp) as day, COUNT(*) as cnt "
            "FROM messages GROUP BY day ORDER BY day DESC LIMIT 7"
        ).fetchall()

        self._json_response({
            "session_count": session_count,
            "message_count": message_count,
            "tool_call_count": tool_calls,
            "tool_usage": dict(sorted(tool_usage.items(), key=lambda x: -x[1])),
            "daily_activity": [
                {"date": r[0], "count": r[1]} for r in daily_rows
            ],
        })

    def _api_search(self, query: str) -> None:
        if not query:
            self._json_response([])
            return
        store = self._get_store()
        results = store.search_messages(query, limit=50)
        self._json_response([self._msg_dict(m) for m in results])

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _msg_dict(m: Any) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": m.id,
            "session_id": m.session_id,
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp,
        }
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls_json:
            try:
                d["tool_calls"] = json.loads(m.tool_calls_json)
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def _serve_html(self) -> None:
        if _DASHBOARD_HTML.exists():
            content = _DASHBOARD_HTML.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content.encode())))
            self.end_headers()
            self.wfile.write(content.encode())
        else:
            self._json_response({"error": "Dashboard HTML not found"}, 500)

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def start_ui(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str | None = None,
    open_browser: bool = True,
) -> None:
    """Start the Pico Agent dashboard server.

    Args:
        host: Bind address. Use "0.0.0.0" to expose on all interfaces.
        port: Port number.
        db_path: Path to sessions.db. Defaults to ~/.pico-agent/sessions.db.
        open_browser: Auto-open browser on start.
    """
    from pico.session import DEFAULT_DB_PATH

    resolved_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    _APIHandler.db_path = str(resolved_db)

    server = HTTPServer((host, port), _APIHandler)
    url = f"http://{host}:{port}"
    print(f"🤖 Pico Agent Dashboard — {url}")
    print(f"   DB: {resolved_db}")
    print("   Press Ctrl+C to stop.\n")

    if open_browser:
        import webbrowser
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
