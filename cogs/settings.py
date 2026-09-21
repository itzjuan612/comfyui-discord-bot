import discord
from discord import app_commands
from discord.ext import commands

from core import user_settings, QUALITY_CHOICES, ASPECT_RATIO_CHOICES, SAMPLER_CHOICES, SCHEDULER_CHOICES
from ui.autocomplete import _sdxl_model_autocomplete, _zimage_model_autocomplete



class SettingsCog(commands.Cog):
    """Per-user generation defaults: /settings /reset_settings"""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="settings", description="View or set your personal generation defaults")
    @app_commands.describe(negative_prompt="Default negative prompt")
    @app_commands.describe(sdxl_checkpoint="Default SDXL checkpoint (models/checkpoints), used by /sdxl and /upscale (SDXL)")
    @app_commands.describe(zimage_model="Default Z-Image diffusion model (models/diffusion_models), used by /zimage")
    @app_commands.describe(width="Default width in pixels (shared by /sdxl and /zimage)")
    @app_commands.describe(height="Default height in pixels (shared by /sdxl and /zimage)")
    @app_commands.describe(sdxl_steps="Default SDXL sampling steps")
    @app_commands.describe(sdxl_cfg="Default SDXL CFG guidance scale")
    @app_commands.describe(zimage_steps="Default Z-Image sampling steps")
    @app_commands.describe(zimage_cfg="Default Z-Image CFG guidance scale")
    @app_commands.choices(sdxl_sampler=SAMPLER_CHOICES)
    @app_commands.choices(sdxl_scheduler=SCHEDULER_CHOICES)
    @app_commands.choices(zimage_sampler=SAMPLER_CHOICES)
    @app_commands.choices(zimage_scheduler=SCHEDULER_CHOICES)
    @app_commands.describe(sdxl_sampler="Default SDXL sampler, used by /sdxl and /upscale (SDXL)")
    @app_commands.describe(sdxl_scheduler="Default SDXL scheduler, used by /sdxl and /upscale (SDXL)")
    @app_commands.describe(zimage_sampler="Default Z-Image sampler, used by /zimage")
    @app_commands.describe(zimage_scheduler="Default Z-Image scheduler, used by /zimage")
    @app_commands.choices(ideogram_quality=QUALITY_CHOICES)
    @app_commands.choices(ideogram_aspect_ratio=ASPECT_RATIO_CHOICES)
    @app_commands.describe(ideogram_quality="Default Ideogram quality preset (Turbo / Default / Quality)")
    @app_commands.describe(ideogram_megapixels="Default Ideogram resolution in megapixels")
    @app_commands.describe(ideogram_aspect_ratio="Default Ideogram aspect ratio preset")
    @app_commands.describe(flux_edit_cfg="Default flux_edit CFG guidance scale")
    @app_commands.describe(flux_edit_steps="Default flux_edit sampling steps")
    @app_commands.choices(flux_edit_sampler=SAMPLER_CHOICES)
    @app_commands.describe(flux_edit_sampler="Default flux_edit sampler")
    @app_commands.describe(flux_edit_megapixels="Default flux_edit resolution in megapixels")
    @app_commands.describe(qwen_steps="Default Qwen sampling steps (shared by /qwen_image and /qwen_edit)")
    @app_commands.describe(qwen_cfg="Default Qwen CFG guidance scale (shared by /qwen_image and /qwen_edit)")
    @app_commands.choices(qwen_sampler=SAMPLER_CHOICES)
    @app_commands.choices(qwen_scheduler=SCHEDULER_CHOICES)
    @app_commands.describe(qwen_sampler="Default Qwen sampler (shared by /qwen_image and /qwen_edit)")
    @app_commands.describe(qwen_scheduler="Default Qwen scheduler (shared by /qwen_image and /qwen_edit)")
    @app_commands.describe(qwen_megapixels="Default Qwen resolution in megapixels (used by /qwen_image)")
    @app_commands.choices(qwen_aspect_ratio=ASPECT_RATIO_CHOICES)
    @app_commands.describe(qwen_aspect_ratio="Default Qwen aspect ratio preset (used by /qwen_image)")
    @app_commands.describe(stealth="Default privacy (ephemeral) for all your generations")
    @app_commands.autocomplete(sdxl_checkpoint=_sdxl_model_autocomplete)
    @app_commands.autocomplete(zimage_model=_zimage_model_autocomplete)
    @app_commands.describe(view="Set to true to only view your current settings")
    async def settings(self, interaction: discord.Interaction,
                        negative_prompt: str | None = None,
                        sdxl_checkpoint: str | None = None,
                        zimage_model: str | None = None,
                        width: int | None = None,
                        height: int | None = None,
                        sdxl_steps: int | None = None,
                        sdxl_cfg: float | None = None,
                        zimage_steps: int | None = None,
                        zimage_cfg: float | None = None,
                        sdxl_sampler: str | None = None,
                        sdxl_scheduler: str | None = None,
                        zimage_sampler: str | None = None,
                        zimage_scheduler: str | None = None,
                        ideogram_quality: str | None = None,
                        ideogram_megapixels: int | None = None,
                        ideogram_aspect_ratio: str | None = None,
                        flux_edit_cfg: float | None = None,
                        flux_edit_steps: int | None = None,
                        flux_edit_sampler: str | None = None,
                        flux_edit_megapixels: int | None = None,
                        qwen_steps: int | None = None,
                        qwen_cfg: float | None = None,
                        qwen_sampler: str | None = None,
                        qwen_scheduler: str | None = None,
                        qwen_megapixels: int | None = None,
                        qwen_aspect_ratio: str | None = None,
                        stealth: bool | None = None,
                        view: bool = False):
        user_id = interaction.user.id
        await interaction.response.defer(ephemeral=True)

        if view or not (
            negative_prompt or sdxl_checkpoint is not None or zimage_model is not None
            or width is not None or height is not None
            or sdxl_steps is not None or sdxl_cfg is not None
            or zimage_steps is not None or zimage_cfg is not None
            or sdxl_sampler is not None or sdxl_scheduler is not None
            or zimage_sampler is not None or zimage_scheduler is not None
            or ideogram_quality is not None or ideogram_megapixels is not None
            or ideogram_aspect_ratio is not None
            or flux_edit_cfg is not None or flux_edit_steps is not None
            or flux_edit_sampler is not None or flux_edit_megapixels is not None
            or qwen_steps is not None or qwen_cfg is not None
            or qwen_sampler is not None or qwen_scheduler is not None
            or qwen_megapixels is not None or qwen_aspect_ratio is not None
            or stealth is not None
        ):
            s = user_settings.get_settings(user_id)
            await interaction.edit_original_response(
                content=f"**Your saved defaults:**\n\n{user_settings.format_settings(s)}\n\n"
                          "Tip: run `/settings` with parameters to update them, or `/reset_settings` to clear them."
            )
            return

        updates = {
            k: v for k, v in {
                "negative_prompt": negative_prompt,
                "sdxl_checkpoint": sdxl_checkpoint,
                "zimage_model": zimage_model,
                "width": width,
                "height": height,
                "sdxl_steps": sdxl_steps,
                "sdxl_cfg": sdxl_cfg,
                "zimage_steps": zimage_steps,
                "zimage_cfg": zimage_cfg,
                "sdxl_sampler": sdxl_sampler,
                "sdxl_scheduler": sdxl_scheduler,
                "zimage_sampler": zimage_sampler,
                "zimage_scheduler": zimage_scheduler,
                "ideogram_quality": ideogram_quality,
                "ideogram_megapixels": ideogram_megapixels,
                "ideogram_aspect_ratio": ideogram_aspect_ratio,
                "flux_edit_cfg": flux_edit_cfg,
                "flux_edit_steps": flux_edit_steps,
                "flux_edit_sampler": flux_edit_sampler,
                "flux_edit_megapixels": flux_edit_megapixels,
                "qwen_steps": qwen_steps,
                "qwen_cfg": qwen_cfg,
                "qwen_sampler": qwen_sampler,
                "qwen_scheduler": qwen_scheduler,
                "qwen_megapixels": qwen_megapixels,
                "qwen_aspect_ratio": qwen_aspect_ratio,
                "stealth": stealth,
            }.items() if v is not None
        }
        s = user_settings.set_settings(user_id, **updates)
        await interaction.edit_original_response(
            content=f"\u2705 Your defaults are now:\n\n{user_settings.format_settings(s)}"
        )


    @app_commands.command(name="reset_settings", description="Clear your personal generation defaults")
    async def reset_settings(self, interaction: discord.Interaction):
        await interaction.response.defer()
        user_settings.reset_settings(interaction.user.id)
        await interaction.edit_original_response(content="\u2705 Your saved defaults have been cleared.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SettingsCog(bot))
