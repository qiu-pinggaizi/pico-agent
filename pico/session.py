"""SQLite-backed session storage for Pico Agent.

Tables:
  - sessions: id, title, created_at, updated_at, source, cost tracking
  - messages: id, session_id, role, content, tool_call_id, timestamp
  - schema_version: version tracking for migrations
  - messages_fts: FTS5 virtual table for full-text search

Design follows Hermes Agent's hermes_state.py patterns:
  - WAL mode for concurrent readers
  - Schema versioning with automatic migration
  - FTS5 full-text search across all messages
  - Per-session cost tracking
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".pico-agent" / "sessions.db"

SCHEMA_VERSION = 2

_SCHEMA_V1_SQL = """\
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

_SCHEMA_V2_ALTER = [
    "ALTER TABLE sessions ADD COLUMN source TEXT NOT NULL DEFAULT 'cli'",
    "ALTER TABLE sessions ADD COLUMN parent_session_id TEXT",
    "ALTER TABLE sessions ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN estimated_cost_usd REAL NOT NULL DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN api_call_count INTEGER NOT NULL DEFAULT 0",
]

_FTS_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content=messages, content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class SessionRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    source: str = "cli"
    parent_session_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0
    api_call_count: int = 0
    message_count: int = 0  # populated by join queries


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
            self._conn = sqlite3.connect(str(self.db_path), timeout=10)
            self._conn.row_factory = sqlite3.Row
            # WAL mode for concurrent readers + better write performance
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()

        # Create schema_version table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        conn.commit()

        # Get current version
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current_version = row["version"] if row else 0

        if current_version == 0:
            # Fresh install: create all tables
            conn.executescript(_SCHEMA_V1_SQL)
            conn.commit()
            current_version = 1

        if current_version < 2:
            # Migrate v1 → v2: add new columns to sessions
            for alter_sql in _SCHEMA_V2_ALTER:
                try:
                    conn.execute(alter_sql)
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.commit()

        # Create FTS table (idempotent) and rebuild for existing data
        conn.executescript(_FTS_SQL)
        # Rebuild FTS index for any existing messages not yet indexed
        try:
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        except sqlite3.OperationalError:
            pass  # already up to date
        conn.commit()

        # Update schema version
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()

        # Ensure message_count is populated on list queries
        logger.debug("Session DB initialised at %s (schema v%d)", self.db_path, SCHEMA_VERSION)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # session operations
    # ------------------------------------------------------------------
    def create_session(
        self,
        title: str = "New Session",
        source: str = "cli",
        parent_session_id: str | None = None,
    ) -> str:
        """Create a new session and return its ID."""
        sid = str(uuid.uuid4())
        now = _now()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, source, parent_session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, title, now, now, source, parent_session_id),
        )
        conn.commit()
        logger.info("Created session %s (%s, source=%s)", sid, title, source)
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
        row = conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
            "FROM sessions s WHERE s.id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return SessionRecord(**dict(row))

    def list_sessions(self, limit: int = 50) -> list[SessionRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
            "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
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
        mid = str(uuid.uuid4())
        now = _now()
        conn = self._get_conn()
        tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
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
        records = self.get_messages(session_id)
        msgs: list[dict[str, Any]] = []
        for r in records:
            d: dict[str, Any] = {"role": r.role, "content": r.content}
            if r.tool_call_id:
                d["tool_call_id"] = r.tool_call_id
            if r.tool_calls_json:
                try:
                    d["tool_calls"] = json.loads(r.tool_calls_json)
                except (ValueError, TypeError):
                    pass
            msgs.append(d)
        return msgs

    def search_messages(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across message content.

        Tries FTS5 MATCH first (fast, with snippets). If FTS5 is unavailable
        or returns no results (e.g. due to tokenizer not splitting compound
        words), falls back to LIKE substring search.

        Returns list of dicts with session_id, role, content, timestamp, snippet.
        """
        conn = self._get_conn()

        # Try FTS5 first (prefix match with * for partial word matching)
        try:
            fts_query = f'"{query}"*'  # prefix match for partial words like YOLOv8n
            rows = conn.execute(
                "SELECT m.session_id, m.role, m.content, m.timestamp, "
                "snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet "
                "FROM messages_fts fts "
                "JOIN messages m ON m.rowid = fts.rowid "
                "WHERE messages_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass  # FTS not available, fall through to LIKE

        # Fallback: LIKE substring search (handles Chinese text, compound words)
        rows = conn.execute(
            "SELECT session_id, role, content, timestamp, '' AS snippet "
            "FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # cost tracking
    # ------------------------------------------------------------------
    def record_usage(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Accumulate token usage and cost for a session."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET "
            "  input_tokens = input_tokens + ?, "
            "  output_tokens = output_tokens + ?, "
            "  cache_read_tokens = cache_read_tokens + ?, "
            "  cache_write_tokens = cache_write_tokens + ?, "
            "  estimated_cost_usd = estimated_cost_usd + ?, "
            "  api_call_count = api_call_count + 1 "
            "WHERE id = ?",
            (input_tokens, output_tokens, cache_read, cache_write, cost, session_id),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def prune_sessions(self, older_than_days: int = 30) -> int:
        """Delete sessions older than the given number of days. Returns count deleted."""
        conn = self._get_conn()
        cutoff = datetime.now().isoformat(timespec="seconds")
        # Use a subquery to find old sessions
        rows = conn.execute(
            "SELECT id FROM sessions WHERE updated_at < datetime(?, '-' || ? || ' days')",
            (cutoff, older_than_days),
        ).fetchall()
        session_ids = [r["id"] for r in rows]
        for sid in session_ids:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        if session_ids:
            conn.execute(
                f"DELETE FROM sessions WHERE id IN ({','.join('?' * len(session_ids))})",
                session_ids,
            )
            conn.commit()
        logger.info("Pruned %d sessions older than %d days", len(session_ids), older_than_days)
        return len(session_ids)

    def get_stats(self) -> dict[str, Any]:
        """Return overview statistics."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT "
            "  COUNT(DISTINCT s.id) AS total_sessions, "
            "  (SELECT COUNT(*) FROM messages) AS total_messages, "
            "  SUM(s.input_tokens) AS total_input_tokens, "
            "  SUM(s.output_tokens) AS total_output_tokens, "
            "  SUM(s.cache_read_tokens) AS total_cache_read_tokens, "
            "  SUM(s.estimated_cost_usd) AS total_cost, "
            "  SUM(s.api_call_count) AS total_api_calls "
            "FROM sessions s"
        ).fetchone()
        return dict(row) if row else {}


class MemorylessSession:
    """A no-op session store for delegation sub-agents that don't persist."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        return ""

    def get_messages_as_dicts(self, session_id: str) -> list[dict[str, Any]]:
        return list(self.messages)
