import os
import sqlite3
import threading
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generations.db")


_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    """Return the persistent connection, creating it once (thread-safe)."""
    global _conn
    if _conn is None:
        with _conn_lock:
            if _conn is None:
                conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                _conn = conn
    return _conn


@contextmanager
def _locked_conn():
    """Yield the persistent connection while holding the module lock.

    Serializes all access on the single shared connection so sqlite3 is
    thread-safe and per-query connect/close overhead is eliminated.
    """
    conn = _connect()
    with _conn_lock:
        yield conn


def init_db() -> None:
    with _locked_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                message_id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save(message_id: int, payload: dict) -> None:
    """Persist generation params for a message (as JSON)."""
    import json
    with _locked_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO generations (message_id, payload) VALUES (?, ?)",
            (message_id, json.dumps(payload)),
        )
        conn.commit()


def get(message_id: int) -> dict | None:
    import json
    with _locked_conn() as conn:
        row = conn.execute(
            "SELECT payload FROM generations WHERE message_id = ?", (message_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def pop(message_id: int) -> None:
    with _locked_conn() as conn:
        conn.execute("DELETE FROM generations WHERE message_id = ?", (message_id,))
        conn.commit()
