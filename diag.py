import asyncio
import logging
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# IMPORTANT: use the SAME bot instance that bot.py and the cogs use.
# The cogs do `from bot import bot`, so registering/syncing on a separate
# Bot instance (as this file originally did) shows zero local commands and
# sync() would wipe the real commands from Discord.
from bot import bot
from core import config


async def main():
    token = os.environ.get("DISCORD_TOKEN") or config.get("discord", {}).get("token")
    if not token:
        raise SystemExit("No Discord token found.")
    await bot.login(token)

    local = sorted(bot.tree._global_commands.keys())
    print("Local commands:", local)

    await bot.tree.sync()
    cmds = await bot.tree.fetch_commands()
    print("Server-side commands:", sorted(c.name for c in cmds))


asyncio.run(main())
