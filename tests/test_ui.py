"""Tests for the Pico Agent dashboard (pico.ui.server)."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import urllib.request
from http.client import HTTPConnection
from pathlib import Path

import pytest

from pico.session import SessionStore
from pico.ui.server import start_ui, _APIHandler, HTTPServer


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary sessions DB with test data."""
    db_path = tmp_path / "test_sessions.db"
    store = SessionStore(db_path=db_path)

    # Create two sessions with messages
    s1 = store.create_session("训练 YOLO 模型")
    store.add_message(s1, "user", "帮我训练一个 YOLOv8n 模型")
    store.add_message(
        s1, "assistant", "好的，开始训练",
        tool_calls=[{"id": "tc1", "name": "auto_train", "arguments": {"data_dir": "/data", "model": "yolov8n"}}],
    )
    store.add_message(s1, "tool", json.dumps({"success": True, "training": {"exit_code": 0}}), tool_call_id="tc1")
    store.add_message(s1, "assistant", "训练完成！mAP50=0.894")

    s2 = store.create_session("数据集下载")
    store.add_message(s2, "user", "下载 COCO 数据集")

    store.close()
    return db_path, s1, s2


@pytest.fixture
def server_port(tmp_db):
    """Start a test server on a random port and yield the port."""
    import socket

    db_path, _, _ = tmp_db

    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = HTTPServer(("127.0.0.1", port), _APIHandler)
    _APIHandler.db_path = str(db_path)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)  # wait for server to start

    yield port

    server.shutdown()
    server.server_close()


def _get(port: int, path: str) -> dict | list:
    """Helper to GET a JSON endpoint."""
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode())
    conn.close()
    return data


def _delete(port: int, path: str) -> dict:
    """Helper to DELETE an endpoint."""
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("DELETE", path)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode())
    conn.close()
    return data


# ------------------------------------------------------------------
# API Tests
# ------------------------------------------------------------------

class TestDashboardAPI:
    def test_list_sessions(self, server_port, tmp_db):
        _, s1, s2 = tmp_db
        data = _get(server_port, "/api/sessions")
        assert len(data) == 2
        titles = {s["title"] for s in data}
        assert "训练 YOLO 模型" in titles
        assert "数据集下载" in titles

    def test_get_session_with_messages(self, server_port, tmp_db):
        _, s1, _ = tmp_db
        data = _get(server_port, f"/api/sessions/{s1}")
        assert data["session"]["title"] == "训练 YOLO 模型"
        assert len(data["messages"]) == 4

        # Check tool_calls are preserved
        assistant_msgs = [m for m in data["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["tool_calls"][0]["name"] == "auto_train"

    def test_get_session_not_found(self, server_port):
        conn = HTTPConnection("127.0.0.1", server_port, timeout=5)
        conn.request("GET", "/api/sessions/nonexistent")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()

    def test_get_messages(self, server_port, tmp_db):
        _, s1, _ = tmp_db
        data = _get(server_port, f"/api/sessions/{s1}/messages")
        assert len(data) == 4
        roles = [m["role"] for m in data]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_stats(self, server_port):
        data = _get(server_port, "/api/stats")
        assert data["session_count"] == 2
        assert data["message_count"] == 5
        assert data["tool_call_count"] == 1
        assert "auto_train" in data["tool_usage"]

    def test_search(self, server_port):
        data = _get(server_port, "/api/search?q=YOLO")
        assert len(data) > 0
        assert any("YOLO" in m["content"] for m in data)

    def test_search_empty(self, server_port):
        data = _get(server_port, "/api/search?q=")
        assert data == []

    def test_search_no_results(self, server_port):
        data = _get(server_port, "/api/search?q=zzzznonexistent")
        assert data == []

    def test_delete_session(self, server_port, tmp_db):
        _, _, s2 = tmp_db
        result = _delete(server_port, f"/api/sessions/{s2}")
        assert result["success"] is True

        # Verify it's gone
        sessions = _get(server_port, "/api/sessions")
        assert len(sessions) == 1

    def test_dashboard_html_served(self, server_port):
        conn = HTTPConnection("127.0.0.1", server_port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        html = resp.read().decode()
        assert "Pico Agent Dashboard" in html
        conn.close()

    def test_unknown_endpoint(self, server_port):
        conn = HTTPConnection("127.0.0.1", server_port, timeout=5)
        conn.request("GET", "/api/unknown")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()
