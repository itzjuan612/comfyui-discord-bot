import os

from bot import bot, log, config
from core import user_settings, generation_store, moderation, BOT_OWNER_ID


def main():
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        token = config.get("discord", {}).get("token")
    if not token:
        raise SystemExit(
            "No Discord token found. Set the DISCORD_TOKEN env var or edit "
            "config.yaml (the bot auto-generated it with empty values)."
        )
    user_settings.init_db()
    generation_store.init_db()
    moderation.init_db()
    # Seed the owner as an admin so they appear in admin lists and can
    # manage other admins. The owner check in can_manage() still works
    # independently, but this keeps the admins table consistent.
    if BOT_OWNER_ID:
        moderation.promote(BOT_OWNER_ID)
        log.info("Owner %s seeded as admin", BOT_OWNER_ID)
    # NOTE: the persistent GenerationView is registered inside on_ready()
    # (where an event loop is running), not here, so it can actually dispatch
    # button clicks. Registering it here, before bot.run(), would leave its
    # internal "stopped" future as None and drop every click.
    bot.run(token)


if __name__ == "__main__":
    main()
