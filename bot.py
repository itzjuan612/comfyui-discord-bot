import asyncio
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
# NOTE: bot.py is the shared bot module. The entry point is main.py, so bot.py
# is only ever loaded once and there is a single Bot instance for everything.
import cogs.generation
import cogs.llm
import cogs.admin
import cogs.settings

from ui.views import GenerationView
from http_session import close_session
from llm_client import refresh_llm_models, llm_model_refresh_loop


@bot.event
async def on_close():
    """Close the shared aiohttp session when the bot shuts down."""
    await close_session()


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
    # Fetch the LLM model list in the background so startup is never blocked
    # by the LLM endpoint, and keep it refreshed on a schedule.
    asyncio.create_task(refresh_llm_models(force=True))
    asyncio.create_task(llm_model_refresh_loop())
