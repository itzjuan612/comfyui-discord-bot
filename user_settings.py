import os
import sqlite3
import threading
import logging
from contextlib import contextmanager

log = logging.getLogger("user_settings")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_settings.db")

# Whitelisted column names. Never build SQL from raw user input.
# Order mirrors the canonical /settings display order.
SETTINGS_FIELDS = (
    "positive_prompt",
    "negative_prompt",
    "sdxl_checkpoint",
    "zimage_model",
    "width",
    "height",
    "sdxl_steps",
    "sdxl_cfg",
    "zimage_steps",
    "zimage_cfg",
    "sdxl_sampler",
    "sdxl_scheduler",
    "zimage_sampler",
    "zimage_scheduler",
    "ideogram_quality",
    "ideogram_megapixels",
    "ideogram_aspect_ratio",
    "img2img_cfg",
    "img2img_steps",
    "img2img_sampler",
    "img2img_megapixels",
    "stealth",
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
    "sdxl_sampler": "TEXT",
    "sdxl_scheduler": "TEXT",
    "zimage_model": "TEXT",
    "width": "INTEGER",
    "height": "INTEGER",
    "zimage_steps": "INTEGER",
    "zimage_cfg": "REAL",
    "zimage_sampler": "TEXT",
    "zimage_scheduler": "TEXT",
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
        # One-time migration: the legacy cfg/steps columns represent SDXL
        # defaults; rename them to sdxl_cfg/sdxl_steps to preserve stored values.
        for old, new in (("cfg", "sdxl_cfg"), ("steps", "sdxl_steps")):
            if old in existing and new not in existing:
                try:
                    conn.execute(f"ALTER TABLE user_settings RENAME COLUMN {old} TO {new}")
                    existing.discard(old)
                    existing.add(new)
                except sqlite3.OperationalError:
                    pass
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
            "SELECT positive_prompt, negative_prompt, sdxl_checkpoint, zimage_model, "
            "width, height, sdxl_steps, sdxl_cfg, zimage_steps, zimage_cfg, "
            "sdxl_sampler, sdxl_scheduler, zimage_sampler, zimage_scheduler, "
            "ideogram_quality, ideogram_megapixels, ideogram_aspect_ratio, "
            "img2img_cfg, img2img_steps, img2img_sampler, img2img_megapixels, stealth "
            "FROM user_settings WHERE discord_id = ?",
            (discord_id,),
        ).fetchone()
    if row is None:
        return {
            "positive_prompt": "", "negative_prompt": "",
            "sdxl_checkpoint": None, "zimage_model": None,
            "width": None, "height": None,
            "sdxl_steps": None, "sdxl_cfg": None,
            "zimage_steps": None, "zimage_cfg": None,
            "sdxl_sampler": None, "sdxl_scheduler": None,
            "zimage_sampler": None, "zimage_scheduler": None,
            "ideogram_quality": None, "ideogram_megapixels": None,
            "ideogram_aspect_ratio": None,
            "img2img_cfg": None, "img2img_steps": None,
            "img2img_sampler": None, "img2img_megapixels": None,
            "stealth": False,
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
            "SET positive_prompt = '', negative_prompt = '', "
            "sdxl_checkpoint = NULL, zimage_model = NULL, width = NULL, height = NULL, "
            "sdxl_steps = NULL, sdxl_cfg = NULL, zimage_steps = NULL, zimage_cfg = NULL, "
            "sdxl_sampler = NULL, sdxl_scheduler = NULL, "
            "zimage_sampler = NULL, zimage_scheduler = NULL, "
            "ideogram_quality = NULL, ideogram_megapixels = NULL, ideogram_aspect_ratio = NULL, "
            "img2img_cfg = NULL, img2img_steps = NULL, img2img_sampler = NULL, "
            "img2img_megapixels = NULL, stealth = NULL "
            "WHERE discord_id = ?",
            (discord_id,),
        )
        conn.commit()
    return get_settings(discord_id)


def format_settings(s: dict) -> str:
    """Human-readable summary of a user's defaults, in the canonical /settings order."""
    def fmt(value):
        return value if value not in (None, "") else "(none)"
    stealth_default = "yes" if s.get("stealth") else "no"
    bullet = "\u2022"
    return (
        f"{bullet} Positive prompt: {fmt(s.get('positive_prompt'))}\n"
        f"{bullet} Negative prompt: {fmt(s.get('negative_prompt'))}\n"
        f"{bullet} SDXL checkpoint: {fmt(s.get('sdxl_checkpoint'))}\n"
        f"{bullet} Z-Image model: {fmt(s.get('zimage_model'))}\n"
        f"{bullet} Width: {fmt(s.get('width'))}\n"
        f"{bullet} Height: {fmt(s.get('height'))}\n"
        f"{bullet} SDXL steps: {fmt(s.get('sdxl_steps'))}\n"
        f"{bullet} SDXL CFG: {fmt(s.get('sdxl_cfg'))}\n"
        f"{bullet} Z-Image steps: {fmt(s.get('zimage_steps'))}\n"
        f"{bullet} Z-Image CFG: {fmt(s.get('zimage_cfg'))}\n"
        f"{bullet} SDXL sampler: {fmt(s.get('sdxl_sampler'))}\n"
        f"{bullet} SDXL scheduler: {fmt(s.get('sdxl_scheduler'))}\n"
        f"{bullet} Z-Image sampler: {fmt(s.get('zimage_sampler'))}\n"
        f"{bullet} Z-Image scheduler: {fmt(s.get('zimage_scheduler'))}\n"
        f"{bullet} Ideogram quality: {fmt(s.get('ideogram_quality'))}\n"
        f"{bullet} Ideogram megapixels: {fmt(s.get('ideogram_megapixels'))}\n"
        f"{bullet} Ideogram aspect ratio: {fmt(s.get('ideogram_aspect_ratio'))}\n"
        f"{bullet} img2img CFG: {fmt(s.get('img2img_cfg'))}\n"
        f"{bullet} img2img steps: {fmt(s.get('img2img_steps'))}\n"
        f"{bullet} img2img sampler: {fmt(s.get('img2img_sampler'))}\n"
        f"{bullet} img2img megapixels: {fmt(s.get('img2img_megapixels'))}\n"
        f"{bullet} Stealth (ephemeral default): {stealth_default}\n"
    )
