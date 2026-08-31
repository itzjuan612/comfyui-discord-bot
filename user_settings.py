import os
import sqlite3
import threading
import logging
from contextlib import contextmanager

log = logging.getLogger("user_settings")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_settings.db")

# Whitelisted column names. Never build SQL from raw user input.
SETTINGS_FIELDS = (
    "positive_prompt",
    "negative_prompt",
    "cfg",
    "steps",
    "ideogram_quality",
    "ideogram_megapixels",
    "ideogram_aspect_ratio",
    "img2img_cfg",
    "img2img_steps",
    "img2img_sampler",
    "img2img_megapixels",
    "stealth",
    "sdxl_checkpoint",
)

# Columns added after the original schema; applied via ALTER TABLE on upgrade.
MIGRATED_COLUMNS = {
    "ideogram_quality": "TEXT",
    "ideogram_megapixels": "INTEGER",
    "ideogram_aspect_ratio": "TEXT",
    "img2img_cfg": "REAL",
    "img2img_steps": "INTEGER",
    "img2img_sampler": "TEXT",
    "img2img_megapixels": "INTEGER",
    "stealth": "INTEGER",
    "sdxl_checkpoint": "TEXT",
}


_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()

# In-memory cache of checkpoints known to lack a bundled text encoder/VAE.
# Loaded once from the database at startup so per-generation lookups are
# O(1) in memory; a database write happens only when a new split checkpoint
# is discovered.
_split_cache: set[str] = set()


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
    """Create the schema on first run and migrate older databases."""
    with _locked_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                discord_id INTEGER PRIMARY KEY,
                positive_prompt TEXT NOT NULL DEFAULT '',
                negative_prompt TEXT NOT NULL DEFAULT '',
                cfg REAL,
                steps INTEGER
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(user_settings)")}
        for col, sql_type in MIGRATED_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE user_settings ADD COLUMN {col} {sql_type}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS split_checkpoints (
                ckpt_name TEXT PRIMARY KEY
            )
            """
        )
        conn.commit()


def load_split_cache() -> None:
    """Load the split-checkpoint table into the in-memory cache (startup)."""
    with _locked_conn() as conn:
        rows = conn.execute("SELECT ckpt_name FROM split_checkpoints").fetchall()
    _split_cache.update(row[0] for row in rows)


def is_split_checkpoint(ckpt_name: str) -> bool:
    """In-memory check: does this checkpoint lack bundled CLIP/VAE?"""
    return ckpt_name in _split_cache


def mark_split_checkpoint(ckpt_name: str) -> None:
    """Record a newly discovered split checkpoint (memory + database).

    The INSERT is a no-op if the checkpoint is already stored, so repeated
    runs for the same checkpoint incur no extra database work.
    """
    if ckpt_name in _split_cache:
        return
    _split_cache.add(ckpt_name)
    with _locked_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO split_checkpoints (ckpt_name) VALUES (?)",
            (ckpt_name,),
        )
        conn.commit()


def get_settings(discord_id: int) -> dict:
    """Return a user's stored defaults."""
    with _locked_conn() as conn:
        row = conn.execute(
            "SELECT positive_prompt, negative_prompt, cfg, steps, "
            "ideogram_quality, ideogram_megapixels, ideogram_aspect_ratio, "
            "img2img_cfg, img2img_steps, img2img_sampler, img2img_megapixels, stealth, sdxl_checkpoint "
            "FROM user_settings WHERE discord_id = ?",
            (discord_id,),
        ).fetchone()
    if row is None:
        return {
            "positive_prompt": "", "negative_prompt": "",
            "cfg": None, "steps": None,
            "ideogram_quality": None, "ideogram_megapixels": None,
            "ideogram_aspect_ratio": None,
            "img2img_cfg": None, "img2img_steps": None,
            "img2img_sampler": None, "img2img_megapixels": None,
            "stealth": False, "sdxl_checkpoint": None,
        }
    return dict(zip(SETTINGS_FIELDS, row))


def set_settings(discord_id: int, **values) -> dict:
    """Upsert one or more settings for a user and return the full record."""
    updates = {k: v for k, v in values.items() if k in SETTINGS_FIELDS}
    with _locked_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO user_settings (discord_id) VALUES (?)",
            (discord_id,),
        )
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE user_settings SET {set_clause} WHERE discord_id = ?",
                (*updates.values(), discord_id),
            )
        conn.commit()
    return get_settings(discord_id)


def reset_settings(discord_id: int) -> dict:
    """Clear all stored defaults for a user."""
    with _locked_conn() as conn:
        conn.execute(
            "UPDATE user_settings "
            "SET positive_prompt = '', negative_prompt = '', cfg = NULL, steps = NULL, "
            "ideogram_quality = NULL, ideogram_megapixels = NULL, ideogram_aspect_ratio = NULL, "
            "img2img_cfg = NULL, img2img_steps = NULL, img2img_sampler = NULL, "
            "img2img_megapixels = NULL, stealth = NULL, sdxl_checkpoint = NULL "
            "WHERE discord_id = ?",
            (discord_id,),
        )
        conn.commit()
    return get_settings(discord_id)


def format_settings(s: dict) -> str:
    """Human-readable summary of a user's defaults."""
    cfg = s["cfg"] if s["cfg"] is not None else "(none)"
    steps = s["steps"] if s["steps"] is not None else "(none)"
    iq = s.get("ideogram_quality") or "(none)"
    im = s.get("ideogram_megapixels") if s.get("ideogram_megapixels") is not None else "(none)"
    ia = s.get("ideogram_aspect_ratio") or "(none)"
    ic = s.get("img2img_cfg") if s.get("img2img_cfg") is not None else "(none)"
    is_ = s.get("img2img_steps") if s.get("img2img_steps") is not None else "(none)"
    isamp = s.get("img2img_sampler") or "(none)"
    imap = s.get("img2img_megapixels") if s.get("img2img_megapixels") is not None else "(none)"
    stealth_default = "yes" if s.get("stealth") else "no"
    sdxl_ckpt = s.get("sdxl_checkpoint") or "(none)"
    return (
        f"• Positive prompt: {s['positive_prompt'] or '(none)'}\n"
        f"• Negative prompt: {s['negative_prompt'] or '(none)'}\n"
        f"• CFG: {cfg}\n"
        f"• Steps: {steps}\n"
        f"• Ideogram quality: {iq}\n"
        f"• Ideogram megapixels: {im}\n"
        f"• Ideogram aspect ratio: {ia}\n"
        f"• img2img CFG: {ic}\n"
        f"• img2img steps: {is_}\n"
        f"• img2img sampler: {isamp}\n"
        f"• img2img megapixels: {imap}\n"
        f"• Stealth (ephemeral default): {stealth_default}\n"
        f"• SDXL checkpoint (default): {sdxl_ckpt}"
    )
