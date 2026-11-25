import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional


def _ensure_dir(db_path: str) -> None:
    if db_path == ":memory:":
        return
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def init_db(db_path: str) -> None:
    _ensure_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                metadata    TEXT,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at    TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def get_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def check_db_health(db_path: str) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except Exception:
        return False


def get_session_system_prompt(db_path: str, session_id: str) -> Optional[str]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["system_prompt"] if row else None


def upsert_session(
    db_path: str, session_id: str, system_prompt: Optional[str] = None
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at) VALUES (?, ?, ?)",
                (session_id, system_prompt, now),
            )
        elif system_prompt is not None:
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE session_id = ?",
                (system_prompt, session_id),
            )
        conn.commit()


def get_session_messages(db_path: str, session_id: str) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    result: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {
            "role": row["role"],
            "content": row["content"],
        }
        if row["role"] == "assistant" and row["metadata"]:
            item["metadata"] = json.loads(row["metadata"])
        result.append(item)
    return result


def add_turn_messages(
    db_path: str,
    session_id: str,
    user_content: str,
    assistant_content: str,
    assistant_metadata: Dict[str, Any],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", user_content, None, now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                "assistant",
                assistant_content,
                json.dumps(assistant_metadata),
                now,
            ),
        )
        conn.commit()
