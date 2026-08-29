import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generations.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                message_id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save(message_id: int, payload: dict) -> None:
    """Persist generation params for a message (as JSON)."""
    import json
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO generations (message_id, payload) VALUES (?, ?)",
            (message_id, json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def get(message_id: int) -> dict | None:
    import json
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT payload FROM generations WHERE message_id = ?", (message_id,)
        ).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def pop(message_id: int) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM generations WHERE message_id = ?", (message_id,))
        conn.commit()
    finally:
        conn.close()
