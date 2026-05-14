"""SQLite-backed session storage for Pico Agent.

Tables:
  - sessions: id, title, created_at, updated_at
  - messages: id, session_id, role, content, tool_call_id, timestamp
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".pico-agent" / "sessions.db"

# NOTE: string type annotation (not Path) for uniformity
_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Session',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id),
    role           TEXT NOT NULL,
    content        TEXT NOT NULL DEFAULT '',
    tool_call_id   TEXT,
    tool_calls_json TEXT,
    timestamp      TEXT NOT NULL
);
"""

# Migration: add tool_calls_json column if missing
_MIGRATION_SQL = """\
ALTER TABLE messages ADD COLUMN tool_calls_json TEXT;
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class SessionRecord:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class MessageRecord:
    id: str
    session_id: str
    role: str
    content: str
    tool_call_id: str | None
    tool_calls_json: str | None
    timestamp: str


class SessionStore:
    """SQLite-backed session and message storage.

    Args:
        db_path: Path to the SQLite database file. Defaults to ~/.pico-agent/sessions.db.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # connection helpers
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        # Migration: add tool_calls_json if missing
        try:
            conn.executescript(_MIGRATION_SQL)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        logger.debug("Session DB initialised at %s", self.db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # session operations
    # ------------------------------------------------------------------
    def create_session(self, title: str = "New Session") -> str:
        """Create a new session and return its ID."""
        sid = str(uuid.uuid4())
        now = _now()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (sid, title, now, now),
        )
        conn.commit()
        logger.info("Created session %s (%s)", sid, title)
        return sid

    def update_session_title(self, session_id: str, title: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        conn.commit()

    def get_session(self, session_id: str) -> SessionRecord | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return SessionRecord(**dict(row))

    def list_sessions(self, limit: int = 50) -> list[SessionRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [SessionRecord(**dict(r)) for r in rows]

    def delete_session(self, session_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        logger.info("Deleted session %s", session_id)

    # ------------------------------------------------------------------
    # message operations
    # ------------------------------------------------------------------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        """Add a message to a session. Returns message ID."""
        import json as _json

        mid = str(uuid.uuid4())
        now = _now()
        conn = self._get_conn()
        tc_json = _json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_calls_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, session_id, role, content, tool_call_id, tc_json, now),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        conn.commit()
        return mid

    def get_messages(self, session_id: str) -> list[MessageRecord]:
        """Get all messages for a session in chronological order."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        return [MessageRecord(**dict(r)) for r in rows]

    def get_messages_as_dicts(self, session_id: str) -> list[dict[str, Any]]:
        """Return messages in the OpenAI-style dict format."""
        import json as _json

        records = self.get_messages(session_id)
        msgs: list[dict[str, Any]] = []
        for r in records:
            d: dict[str, Any] = {"role": r.role, "content": r.content}
            if r.tool_call_id:
                d["tool_call_id"] = r.tool_call_id
            if r.tool_calls_json:
                try:
                    d["tool_calls"] = _json.loads(r.tool_calls_json)
                except (ValueError, TypeError):
                    pass
            msgs.append(d)
        return msgs

    def search_messages(self, query: str, limit: int = 20) -> list[MessageRecord]:
        """Full-text search across message content."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [MessageRecord(**dict(r)) for r in rows]


class MemorylessSession:
    """A no-op session store for delegation sub-agents that don't persist."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_message(self, session_id: str, role: str, content: str, tool_call_id: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> str:
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        return ""

    def get_messages_as_dicts(self, session_id: str) -> list[dict[str, Any]]:
        return list(self.messages)
