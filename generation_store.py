import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generations.db")
log = logging.getLogger("bot")

# Retry params for a month-old message are pointless (the Discord message is
# almost certainly gone and users never scroll back that far), so rows older
# than this are pruned at startup.
MAX_AGE_SECONDS = 30 * 24 * 3600


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
                payload TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        # One-time migration for databases created before created_at existed.
        cols = [row[1] for row in conn.execute("PRAGMA table_info(generations)")]
        if "created_at" not in cols:
            conn.execute(
                "ALTER TABLE generations ADD COLUMN created_at REAL NOT NULL DEFAULT 0"
            )
            # Give pre-existing rows a fresh timestamp so they are not
            # immediately pruned despite possibly recent activity.
            conn.execute(
                "UPDATE generations SET created_at = ?", (time.time(),)
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generations_created ON generations (created_at)"
        )
        conn.commit()
    prune()


def save(message_id: int, payload: dict) -> None:
    """Persist generation params for a message (as JSON), timestamped for pruning."""
    import json
    with _locked_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO generations (message_id, payload, created_at) VALUES (?, ?, ?)",
            (message_id, json.dumps(payload), time.time()),
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


def prune(max_age_seconds: float = MAX_AGE_SECONDS) -> int:
    """Delete generation records older than ``max_age_seconds``.

    Keeps generations.db from growing without bound: rows whose Discord
    message has long been deleted (or the channel purged) are dropped.
    Returns the number of rows removed.
    """
    cutoff = time.time() - max_age_seconds
    with _locked_conn() as conn:
        cur = conn.execute("DELETE FROM generations WHERE created_at < ?", (cutoff,))
        removed = cur.rowcount
        conn.commit()
    if removed:
        log.info("generation_store: pruned %d expired record(s)", removed)
    return removed
