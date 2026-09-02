import discord
from discord import app_commands

from core import comfy, log
from llm_client import get_llm_models


async def llm_model_autocomplete(interaction: discord.Interaction, current_input: str):
    """Dynamic autocomplete for /gen_prompt's ``model`` parameter.

    Suggests LLM model ids from the cached list (fetched lazily and refreshed
    on a schedule), filtered by what the user has typed so far.
    """
    try:
        models = await get_llm_models()
    except Exception as exc:
        log.warning("Could not fetch LLM models for autocomplete: %s", exc)
        return []
    if current_input:
        matches = [m for m in models if current_input.lower() in m.lower()]
    else:
        matches = models
    # Discord allows at most 25 autocomplete choices.
    return [app_commands.Choice(name=m, value=m) for m in matches[:25]]


async def _sdxl_model_autocomplete(interaction: discord.Interaction, current_input: str):
    """Dynamic autocomplete for /sdxl's ``model`` parameter.

    Suggests checkpoint files from ComfyUI's models/checkpoints folder,
    filtered by what the user has typed so far. The list comes from the
    cached ``fetch_checkpoints()`` (60s TTL), so newly added checkpoint
    files appear without a bot restart or slash-command re-sync.
    """
    try:
        checkpoints = await comfy.fetch_checkpoints()
    except Exception as exc:
        log.warning("Could not fetch checkpoint list for autocomplete: %s", exc)
        return []
    if current_input:
        matches = [c for c in checkpoints if current_input.lower() in c.lower()]
    else:
        matches = checkpoints
    # Discord allows at most 25 autocomplete choices.
    return [app_commands.Choice(name=c, value=c) for c in matches[:25]]


async def _sdxl_lora_autocomplete(interaction: discord.Interaction, current_input: str):
    """Dynamic autocomplete for /sdxl's ``lora1``/``lora2`` parameters.

    Suggests LoRA files from ComfyUI's models/loras folder, filtered by what
    the user has typed so far.
    """
    try:
        loras = await comfy.fetch_loras()
    except Exception as exc:
        log.warning("Could not fetch LoRA list for autocomplete: %s", exc)
        return []
    if current_input:
        matches = [l for l in loras if current_input.lower() in l.lower()]
    else:
        matches = loras
    # Discord allows at most 25 autocomplete choices.
    return [app_commands.Choice(name=l, value=l) for l in matches[:25]]
