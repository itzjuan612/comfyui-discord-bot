import os
import sqlite3
import threading
import logging
from contextlib import contextmanager

log = logging.getLogger("moderation")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moderation.db")


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
        conn.execute("CREATE TABLE IF NOT EXISTS admins (discord_id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE IF NOT EXISTS bans (discord_id INTEGER PRIMARY KEY)")
        conn.commit()


def _ids(table: str) -> list[int]:
    with _locked_conn() as conn:
        return [row[0] for row in conn.execute(f"SELECT discord_id FROM {table}")]


def _has(table: str, discord_id: int) -> bool:
    with _locked_conn() as conn:
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE discord_id = ?", (discord_id,)
        ).fetchone()
    return row is not None


def _set(table: str, discord_id: int, present: bool) -> None:
    with _locked_conn() as conn:
        if present:
            conn.execute("INSERT OR IGNORE INTO {} (discord_id) VALUES (?)".format(table), (discord_id,))
        else:
            conn.execute("DELETE FROM {} WHERE discord_id = ?".format(table), (discord_id,))
        conn.commit()


# --- Admins ---
def is_admin(discord_id: int) -> bool:
    return _has("admins", discord_id)


def promote(discord_id: int) -> None:
    _set("admins", discord_id, True)


def demote(discord_id: int) -> None:
    _set("admins", discord_id, False)


def get_admins() -> list[int]:
    return _ids("admins")


# --- Bans ---
def is_banned(discord_id: int) -> bool:
    return _has("bans", discord_id)


def ban(discord_id: int) -> None:
    _set("bans", discord_id, True)


def unban(discord_id: int) -> None:
    _set("bans", discord_id, False)


def get_bans() -> list[int]:
    return _ids("bans")
