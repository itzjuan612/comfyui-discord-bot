import discord
from discord import app_commands

from bot import bot
from core import user_settings, QUALITY_CHOICES, ASPECT_RATIO_CHOICES, SAMPLER_CHOICES, SCHEDULER_CHOICES


@bot.tree.command(name="settings", description="View or set your personal generation defaults")
@app_commands.describe(positive_prompt="Default positive prompt")
@app_commands.describe(negative_prompt="Default negative prompt")
@app_commands.describe(cfg="Default CFG guidance scale (SDXL only)")
@app_commands.describe(steps="Default sampling steps (SDXL only)")
@app_commands.choices(ideogram_quality=QUALITY_CHOICES)
@app_commands.choices(ideogram_aspect_ratio=ASPECT_RATIO_CHOICES)
@app_commands.describe(ideogram_quality="Default Ideogram quality preset (Turbo / Default / Quality)")
@app_commands.describe(ideogram_megapixels="Default Ideogram resolution in megapixels")
@app_commands.describe(ideogram_aspect_ratio="Default Ideogram aspect ratio preset")
@app_commands.choices(img2img_sampler=SAMPLER_CHOICES)
@app_commands.choices(sdxl_sampler=SAMPLER_CHOICES)
@app_commands.choices(sdxl_scheduler=SCHEDULER_CHOICES)
@app_commands.describe(img2img_cfg="Default img2img CFG guidance scale")
@app_commands.describe(img2img_steps="Default img2img sampling steps")
@app_commands.describe(img2img_sampler="Default img2img sampler")
@app_commands.describe(img2img_megapixels="Default img2img resolution in megapixels")
@app_commands.describe(stealth="Default privacy (ephemeral) for all your generations")
@app_commands.describe(sdxl_checkpoint="Default SDXL checkpoint (models/checkpoints), used by /sdxl and /upscale (SDXL)")
@app_commands.describe(sdxl_sampler="Default SDXL sampler, used by /sdxl and /upscale (SDXL)")
@app_commands.describe(sdxl_scheduler="Default SDXL scheduler, used by /sdxl and /upscale (SDXL)")
@app_commands.describe(view="Set to true to only view your current settings")
async def settings(interaction: discord.Interaction,
                    positive_prompt: str | None = None,
                    negative_prompt: str | None = None,
                    sdxl_checkpoint: str | None = None,
                    cfg: float | None = None,
                    steps: int | None = None,
                    sdxl_sampler: str | None = None,
                    sdxl_scheduler: str | None = None,
                    ideogram_quality: str | None = None,
                    ideogram_megapixels: int | None = None,
                    ideogram_aspect_ratio: str | None = None,
                    img2img_cfg: float | None = None,
                    img2img_steps: int | None = None,
                    img2img_sampler: str | None = None,
                    img2img_megapixels: int | None = None,
                    stealth: bool | None = None,
                    view: bool = False):
    user_id = interaction.user.id
    await interaction.response.defer(ephemeral=True)

    if view or not (
        positive_prompt or negative_prompt or cfg is not None or steps is not None
        or ideogram_quality is not None or ideogram_megapixels is not None
        or ideogram_aspect_ratio is not None
        or img2img_cfg is not None or img2img_steps is not None
        or img2img_sampler is not None or img2img_megapixels is not None
        or stealth is not None
        or sdxl_checkpoint is not None
        or sdxl_sampler is not None or sdxl_scheduler is not None
    ):
        s = user_settings.get_settings(user_id)
        await interaction.edit_original_response(
            content=f"**Your saved defaults:**\n\n{user_settings.format_settings(s)}\n\n"
                     "Tip: run `/settings` with parameters to update them, or `/reset_settings` to clear them."
        )
        return

    updates = {
        k: v for k, v in {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "sdxl_checkpoint": sdxl_checkpoint,
            "cfg": cfg,
            "steps": steps,
            "sdxl_sampler": sdxl_sampler,
            "sdxl_scheduler": sdxl_scheduler,
            "ideogram_quality": ideogram_quality,
            "ideogram_megapixels": ideogram_megapixels,
            "ideogram_aspect_ratio": ideogram_aspect_ratio,
            "img2img_cfg": img2img_cfg,
            "img2img_steps": img2img_steps,
            "img2img_sampler": img2img_sampler,
            "img2img_megapixels": img2img_megapixels,
            "stealth": stealth,
        }.items() if v is not None
    }
    s = user_settings.set_settings(user_id, **updates)
    await interaction.edit_original_response(
        content=f"\u2705 Your defaults are now:\n\n{user_settings.format_settings(s)}"
    )


@bot.tree.command(name="reset_settings", description="Clear your personal generation defaults")
async def reset_settings(interaction: discord.Interaction):
    await interaction.response.defer()
    user_settings.reset_settings(interaction.user.id)
    await interaction.edit_original_response(content="\u2705 Your saved defaults have been cleared.")

