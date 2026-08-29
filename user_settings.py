import os
import sqlite3
import logging

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


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the schema on first run and migrate older databases."""
    conn = _connect()
    try:
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
        conn.commit()
    finally:
        conn.close()


def get_settings(discord_id: int) -> dict:
    """Return a user's stored defaults."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT positive_prompt, negative_prompt, cfg, steps, "
            "ideogram_quality, ideogram_megapixels, ideogram_aspect_ratio, "
            "img2img_cfg, img2img_steps, img2img_sampler, img2img_megapixels, stealth, sdxl_checkpoint "
            "FROM user_settings WHERE discord_id = ?",
            (discord_id,),
        ).fetchone()
    finally:
        conn.close()
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
    conn = _connect()
    try:
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
    finally:
        conn.close()
    return get_settings(discord_id)


def reset_settings(discord_id: int) -> dict:
    """Clear all stored defaults for a user."""
    conn = _connect()
    try:
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
    finally:
        conn.close()
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
