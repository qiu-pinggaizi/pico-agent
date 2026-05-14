"""Tests for SessionStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from pico.session import SessionStore


class TestCreateSession:
    """test_create_session — create and retrieve."""

    def test_create_session(self, tmp_session_db: SessionStore) -> None:
        sid = tmp_session_db.create_session(title="Test Session")
        assert sid
        record = tmp_session_db.get_session(sid)
        assert record is not None
        assert record.title == "Test Session"


class TestAddMessages:
    """test_add_messages — add user/assistant messages, retrieve in order."""

    def test_add_messages(self, tmp_session_db: SessionStore) -> None:
        sid = tmp_session_db.create_session()
        tmp_session_db.add_message(sid, "user", "hello")
        tmp_session_db.add_message(sid, "assistant", "hi there")

        msgs = tmp_session_db.get_messages_as_dicts(sid)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def test_add_messages_with_tool_call(self, tmp_session_db: SessionStore) -> None:
        sid = tmp_session_db.create_session()
        tmp_session_db.add_message(sid, "tool", "result data", tool_call_id="tc_1")

        msgs = tmp_session_db.get_messages_as_dicts(sid)
        assert len(msgs) == 1
        assert msgs[0]["tool_call_id"] == "tc_1"


class TestListSessions:
    """test_list_sessions — create multiple, list them."""

    def test_list_sessions(self, tmp_session_db: SessionStore) -> None:
        tmp_session_db.create_session(title="Session A")
        tmp_session_db.create_session(title="Session B")
        tmp_session_db.create_session(title="Session C")

        sessions = tmp_session_db.list_sessions()
        assert len(sessions) == 3
        titles = {s.title for s in sessions}
        assert titles == {"Session A", "Session B", "Session C"}
