import os

import discord
from discord.ext import commands

from core import (
    log, config, BOT_OWNER_ID,
    T2I_MODELS, UPSCALE_MODELS, I2I_MODELS,
    user_settings, generation_store, moderation,
)
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

# Import cogs (they import ui.views, which needs ``bot``); each registers its
# slash commands on bot.tree at import time.
import cogs.generation
import cogs.llm
import cogs.admin
import cogs.settings

from ui.views import GenerationView


@bot.event
async def on_ready():
    await bot.tree.sync()
    # Register the persistent GenerationView inside the running event loop.
    # (Calling bot.add_view() before bot.run() creates the view outside the
    # loop, so its internal "stopped" future is None and it silently drops
    # every button click -> buttons appear to time out.)
    persistent_view = GenerationView()
    bot.add_view(persistent_view)
    log.info(
        "Registered persistent GenerationView: persistent=%s, dispatchable=%s",
        persistent_view.is_persistent(), persistent_view.is_dispatchable(),
    )
    print(f"Bot ready. T2I models: {T2I_MODELS}, Upscale models: {UPSCALE_MODELS}, I2I models: {I2I_MODELS}")


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
