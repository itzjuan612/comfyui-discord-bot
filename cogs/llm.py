import asyncio
import io

import discord
from discord import app_commands

from bot import bot
from core import config, log, ban_guard, check_cooldown, nsfw_blocked, schedule_message_deletion, reply_error, ASPECT_RATIO_CHOICES
from llm_client import fetch_llm_models, llm_model_load, probe_reasoning_efforts, llm_model_unload, call_llm, resolve_reasoning_effort
from ui.autocomplete import llm_model_autocomplete
from job_queue import job_queue
from ui.views import ThinkingView
from workflow import run_text_workflow
from core import normalize_aspect_ratio


@bot.tree.command(name="gen_prompt", description="Convert a natural-language prompt into an Ideogram 4 JSON caption prompt")
@app_commands.choices(aspect_ratio=ASPECT_RATIO_CHOICES)
@app_commands.autocomplete(model=llm_model_autocomplete)
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
        # A single session job holds the LLM lane for the entire gen_prompt
        # session: load -> probe -> user selection -> generation -> unload.
        # This prevents other LLM jobs from starting while the model is loaded.
        completion = asyncio.Event()
        state = {"selected_value": None}

        async def _gen_prompt_session():
            # Load model and probe supported reasoning efforts.
            await llm_model_load(chosen_model)
            supported = await probe_reasoning_efforts(chosen_model)

            # Show the reasoning-effort picker and wait for the user to choose
            # (or for the 30-second timeout to fire).
            view = ThinkingView(
                chosen_model, supported, prompt, megapixels, aspect_ratio,
                max_tokens, temperature, llm_cfg, completion, state,
            )
            await msg.edit(
                content="\U0001f9e0 Select the reasoning effort for " + chosen_model + ":",
                view=view,
            )
            await completion.wait()

            # The user either selected a reasoning effort or the view timed out.
            if state["selected_value"] is not None:
                effort = await resolve_reasoning_effort(chosen_model, state["selected_value"], llm_cfg)
                patches = {
                    "191": {
                        "aspect_ratio": normalize_aspect_ratio(aspect_ratio),
                        "megapixels": int(megapixels),
                    },
                    "134:115": {
                        "value": prompt,
                    },
                }
                composed_prompt = await run_text_workflow(
                    "workflows/ideogram_prompt_gen/ideogram4_prompt_gen.json",
                    patches,
                    target_node="111",
                )
                result = await call_llm(
                    composed_prompt,
                    chosen_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=effort,
                )
                # Write the result into the "Generating prompt\u2026" follow-up
                # message that was posted when the effort was selected, not the
                # picker message. Fall back to the picker if the follow-up
                # couldn't be created.
                out_msg = state.get("followup_msg") or msg
                if len(result) > 2000:
                    txt_file = discord.File(
                        io.BytesIO(result.encode("utf-8")), filename="prompt.txt"
                    )
                    await out_msg.edit(
                        content=(
                            "\u26a0\ufe0f The generated prompt is longer than Discord's "
                            "2000-character limit, so it can't be shown directly. "
                            "It has been saved to the attached .txt file \u2014 copy the "
                            "text inside that file instead."
                        ),
                        attachments=[txt_file],
                    )
                else:
                    await out_msg.edit(content=result)
            else:
                # No reasoning effort was selected before the timeout; tell the
                # user on the picker message that generation was cancelled.
                await msg.edit(
                    content="\u23f3 You ran out of time to select a reasoning effort, so prompt generation was cancelled."
                )

            # Unload the model (ignored on servers without the unload endpoint).
            await llm_model_unload(chosen_model)

        await job_queue.submit(_gen_prompt_session(), lane="llm", name="gen_prompt")
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
