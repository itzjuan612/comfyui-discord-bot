import discord
from discord import app_commands

from bot import bot
from core import config, log, ban_guard, check_cooldown, nsfw_blocked, schedule_message_deletion, reply_error, ASPECT_RATIO_CHOICES
from llm_client import fetch_llm_models, llm_model_load, probe_reasoning_efforts, LLM_MODEL_CHOICES
from ui.views import ThinkingView


@bot.tree.command(name="gen_prompt", description="Convert a natural-language prompt into an Ideogram 4 JSON caption prompt")
@app_commands.choices(aspect_ratio=ASPECT_RATIO_CHOICES)
@app_commands.choices(model=LLM_MODEL_CHOICES)
@app_commands.describe(prompt="Natural-language prompt to convert")
@app_commands.describe(megapixels="Target resolution in megapixels")
@app_commands.describe(model="LLM model to use (see /llm_models)")
@app_commands.describe(max_tokens="Max tokens for the LLM (optional)")
@app_commands.describe(temperature="LLM temperature, 0-1 (optional)")
async def gen_prompt(interaction: discord.Interaction, prompt: str, megapixels: int, aspect_ratio: str,
                     model: str | None = None, max_tokens: int | None = None,
                     temperature: float | None = None):
    if await ban_guard(interaction):
        return
    if not await check_cooldown(interaction):
        await interaction.response.send_message(
            content="\u23f3 Please wait before requesting another prompt.", ephemeral=True
        )
        return

    if nsfw_blocked(interaction, prompt):
        await interaction.response.send_message(
            content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    await interaction.response.defer(ephemeral=True)
    msg = await interaction.original_response()
    llm_cfg = config.get("llm", {})
    chosen_model = model or llm_cfg.get("default_model") or ""
    if not chosen_model:
        await reply_error(interaction, "\u274c No model selected and no default model configured.", target=msg)
        return
    log.info("gen_prompt: prompt=%r megapixels=%s aspect_ratio=%s model=%r", prompt, megapixels, aspect_ratio, chosen_model)
    try:
        # Load the model and probe which reasoning-effort values the endpoint
        # accepts; only those become selectable options in the view.
        await llm_model_load(chosen_model)
        supported = await probe_reasoning_efforts(chosen_model)
        view = ThinkingView(
            chosen_model, supported, prompt, megapixels, aspect_ratio,
            max_tokens, temperature, llm_cfg,
        )
        await msg.edit(
            content="\U0001f9e0 Select the reasoning effort for " + chosen_model + ":",
            view=view,
        )
    except Exception as exc:
        log.exception("gen_prompt setup failed")
        await reply_error(interaction, f"\u274c Prompt generation failed: {exc}", target=msg)
@bot.tree.command(name="llm_models", description="List the LLM models available on the configured endpoint")
async def llm_models(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    msg = await interaction.original_response()
    try:
        models = await fetch_llm_models()
        await msg.edit(
            content="\U0001f916 Available LLM models:\n\n" + "\n".join(f"\u2022 {m}" for m in models)
        )
    except Exception as exc:
        log.warning("llm_models failed: %s", exc)
        await reply_error(interaction, f"\u274c Could not list models: {exc}", target=msg)
