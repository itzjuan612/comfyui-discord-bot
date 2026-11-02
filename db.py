"""Shared SQLite plumbing for the bot's databases.

Each database file gets exactly one persistent connection, opened lazily and
guarded by its own lock. The connection is shared between the event loop and
worker threads (``check_same_thread=False``), so every access goes through
``Database.locked()`` which serializes it.

Queries in this bot are single-row reads/writes against indexed columns, so
the lock is held for microseconds and the event loop is not meaningfully
blocked. WAL + ``busy_timeout`` cover the case where something else (a backup
script, a second process, a checkpoint) holds the write lock.
"""

import sqlite3
import threading
from contextlib import contextmanager

# Wait for a competing writer instead of failing at once with
# "database is locked".
BUSY_TIMEOUT_MS = 5000


class Database:
    """One lazily-opened, shared connection to a SQLite file."""

    def __init__(self, path: str):
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    conn = sqlite3.connect(self.path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    # Safe with WAL and much cheaper than FULL.
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
                    self._conn = conn
        return self._conn

    @contextmanager
    def locked(self):
        """Yield the shared connection while holding this database's lock."""
        conn = self._connect()
        with self._lock:
            yield conn
