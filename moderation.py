import os
import sqlite3
import logging

log = logging.getLogger("moderation")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moderation.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS admins (discord_id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE IF NOT EXISTS bans (discord_id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def _ids(table: str) -> list[int]:
    conn = _connect()
    try:
        return [row[0] for row in conn.execute(f"SELECT discord_id FROM {table}")]
    finally:
        conn.close()


def _has(table: str, discord_id: int) -> bool:
    conn = _connect()
    try:
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE discord_id = ?", (discord_id,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _set(table: str, discord_id: int, present: bool) -> None:
    conn = _connect()
    try:
        if present:
            conn.execute("INSERT OR IGNORE INTO {} (discord_id) VALUES (?)".format(table), (discord_id,))
        else:
            conn.execute("DELETE FROM {} WHERE discord_id = ?".format(table), (discord_id,))
        conn.commit()
    finally:
        conn.close()


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
