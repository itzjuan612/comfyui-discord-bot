import discord
from discord import app_commands
from discord.ext import commands

from core import user_settings, QUALITY_CHOICES, ASPECT_RATIO_CHOICES, SAMPLER_CHOICES, SCHEDULER_CHOICES
from ui.autocomplete import _sdxl_model_autocomplete, _zimage_model_autocomplete


class SettingsCog(commands.Cog):
    """Per-user generation defaults: /settings <section> and /reset_settings.

    Discord allows at most 25 options per command, and the per-model defaults
    exceed that, so /settings is a subcommand group: one subcommand per model
    (plus view/general). Running a subcommand with no parameters shows the
    current defaults instead of changing anything, and every subcommand stays
    comfortably below the 25-option limit as new models are added.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    settings = app_commands.Group(
        name="settings",
        description="View or set your personal generation defaults",
    )

    async def _show(self, interaction: discord.Interaction) -> None:
        s = user_settings.get_settings(interaction.user.id)
        await interaction.edit_original_response(
            content=f"**Your saved defaults:**\n\n{user_settings.format_settings(s)}\n\n"
                      "Tip: run a `/settings` subcommand with parameters to update them, or `/reset_settings` to clear them."
        )

    async def _apply(self, interaction: discord.Interaction, **values) -> None:
        updates = {k: v for k, v in values.items() if v is not None}
        if not updates:
            await self._show(interaction)
            return
        s = user_settings.set_settings(interaction.user.id, **updates)
        await interaction.edit_original_response(
            content=f"\u2705 Your defaults are now:\n\n{user_settings.format_settings(s)}"
        )

    @settings.command(name="view", description="View all your saved generation defaults")
    async def settings_view(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        await self._show(interaction)

    @settings.command(name="general", description="Default negative prompt, image size and privacy")
    @app_commands.describe(negative_prompt="Default negative prompt")
    @app_commands.describe(width="Default width in pixels (shared by /sdxl and /zimage)")
    @app_commands.describe(height="Default height in pixels (shared by /sdxl and /zimage)")
    @app_commands.describe(stealth="Default privacy (ephemeral) for all your generations")
    async def settings_general(self, interaction: discord.Interaction,
                               negative_prompt: str | None = None,
                               width: int | None = None,
                               height: int | None = None,
                               stealth: bool | None = None):
        await interaction.response.defer(ephemeral=True)
        await self._apply(
            interaction,
            negative_prompt=negative_prompt, width=width, height=height, stealth=stealth,
        )

    @settings.command(name="sdxl", description="Default SDXL settings (used by /sdxl and SDXL upscales)")
    @app_commands.autocomplete(sdxl_checkpoint=_sdxl_model_autocomplete)
    @app_commands.describe(sdxl_checkpoint="Default SDXL checkpoint (models/checkpoints)")
    @app_commands.describe(sdxl_steps="Default SDXL sampling steps")
    @app_commands.describe(sdxl_cfg="Default SDXL CFG guidance scale")
    @app_commands.choices(sdxl_sampler=SAMPLER_CHOICES)
    @app_commands.choices(sdxl_scheduler=SCHEDULER_CHOICES)
    @app_commands.describe(sdxl_sampler="Default SDXL sampler")
    @app_commands.describe(sdxl_scheduler="Default SDXL scheduler")
    async def settings_sdxl(self, interaction: discord.Interaction,
                            sdxl_checkpoint: str | None = None,
                            sdxl_steps: int | None = None,
                            sdxl_cfg: float | None = None,
                            sdxl_sampler: str | None = None,
                            sdxl_scheduler: str | None = None):
        await interaction.response.defer(ephemeral=True)
        await self._apply(
            interaction,
            sdxl_checkpoint=sdxl_checkpoint, sdxl_steps=sdxl_steps, sdxl_cfg=sdxl_cfg,
            sdxl_sampler=sdxl_sampler, sdxl_scheduler=sdxl_scheduler,
        )

    @settings.command(name="zimage", description="Default Z-Image settings (used by /zimage)")
    @app_commands.autocomplete(zimage_model=_zimage_model_autocomplete)
    @app_commands.describe(zimage_model="Default Z-Image diffusion model (models/diffusion_models)")
    @app_commands.describe(zimage_steps="Default Z-Image sampling steps")
    @app_commands.describe(zimage_cfg="Default Z-Image CFG guidance scale")
    @app_commands.choices(zimage_sampler=SAMPLER_CHOICES)
    @app_commands.choices(zimage_scheduler=SCHEDULER_CHOICES)
    @app_commands.describe(zimage_sampler="Default Z-Image sampler")
    @app_commands.describe(zimage_scheduler="Default Z-Image scheduler")
    async def settings_zimage(self, interaction: discord.Interaction,
                              zimage_model: str | None = None,
                              zimage_steps: int | None = None,
                              zimage_cfg: float | None = None,
                              zimage_sampler: str | None = None,
                              zimage_scheduler: str | None = None):
        await interaction.response.defer(ephemeral=True)
        await self._apply(
            interaction,
            zimage_model=zimage_model, zimage_steps=zimage_steps, zimage_cfg=zimage_cfg,
            zimage_sampler=zimage_sampler, zimage_scheduler=zimage_scheduler,
        )

    @settings.command(name="ideogram", description="Default Ideogram settings (used by /ideogram)")
    @app_commands.choices(ideogram_quality=QUALITY_CHOICES)
    @app_commands.describe(ideogram_quality="Default quality preset (Turbo / Default / Quality)")
    @app_commands.describe(ideogram_megapixels="Default resolution in megapixels")
    @app_commands.choices(ideogram_aspect_ratio=ASPECT_RATIO_CHOICES)
    @app_commands.describe(ideogram_aspect_ratio="Default aspect ratio preset")
    async def settings_ideogram(self, interaction: discord.Interaction,
                                ideogram_quality: str | None = None,
                                ideogram_megapixels: int | None = None,
                                ideogram_aspect_ratio: str | None = None):
        await interaction.response.defer(ephemeral=True)
        await self._apply(
            interaction,
            ideogram_quality=ideogram_quality, ideogram_megapixels=ideogram_megapixels,
            ideogram_aspect_ratio=ideogram_aspect_ratio,
        )

    @settings.command(name="flux_edit", description="Default Flux Edit settings (used by /flux_edit and the Flux Edit button)")
    @app_commands.describe(flux_edit_cfg="Default CFG guidance scale")
    @app_commands.describe(flux_edit_steps="Default sampling steps")
    @app_commands.choices(flux_edit_sampler=SAMPLER_CHOICES)
    @app_commands.describe(flux_edit_sampler="Default sampler")
    @app_commands.describe(flux_edit_megapixels="Default resolution in megapixels")
    async def settings_flux_edit(self, interaction: discord.Interaction,
                                 flux_edit_cfg: float | None = None,
                                 flux_edit_steps: int | None = None,
                                 flux_edit_sampler: str | None = None,
                                 flux_edit_megapixels: int | None = None):
        await interaction.response.defer(ephemeral=True)
        await self._apply(
            interaction,
            flux_edit_cfg=flux_edit_cfg, flux_edit_steps=flux_edit_steps,
            flux_edit_sampler=flux_edit_sampler, flux_edit_megapixels=flux_edit_megapixels,
        )

    @settings.command(name="qwen", description="Default Qwen Image 2.1 settings (used by /qwen_image, /qwen_edit and the Qwen Edit button)")
    @app_commands.describe(qwen_steps="Default sampling steps")
    @app_commands.describe(qwen_cfg="Default CFG guidance scale")
    @app_commands.choices(qwen_sampler=SAMPLER_CHOICES)
    @app_commands.choices(qwen_scheduler=SCHEDULER_CHOICES)
    @app_commands.describe(qwen_sampler="Default sampler")
    @app_commands.describe(qwen_scheduler="Default scheduler")
    @app_commands.describe(qwen_megapixels="Default resolution in megapixels (used by /qwen_image)")
    @app_commands.choices(qwen_aspect_ratio=ASPECT_RATIO_CHOICES)
    @app_commands.describe(qwen_aspect_ratio="Default aspect ratio preset (used by /qwen_image)")
    async def settings_qwen(self, interaction: discord.Interaction,
                            qwen_steps: int | None = None,
                            qwen_cfg: float | None = None,
                            qwen_sampler: str | None = None,
                            qwen_scheduler: str | None = None,
                            qwen_megapixels: int | None = None,
                            qwen_aspect_ratio: str | None = None):
        await interaction.response.defer(ephemeral=True)
        await self._apply(
            interaction,
            qwen_steps=qwen_steps, qwen_cfg=qwen_cfg, qwen_sampler=qwen_sampler,
            qwen_scheduler=qwen_scheduler, qwen_megapixels=qwen_megapixels,
            qwen_aspect_ratio=qwen_aspect_ratio,
        )

    @app_commands.command(name="reset_settings", description="Clear your personal generation defaults")
    async def reset_settings(self, interaction: discord.Interaction):
        await interaction.response.defer()
        user_settings.reset_settings(interaction.user.id)
        await interaction.edit_original_response(content="\u2705 Your saved defaults have been cleared.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SettingsCog(bot))
