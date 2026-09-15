import discord
from discord import app_commands
from discord.ext import commands

from core import log, moderation, BOT_OWNER_ID, can_manage, is_owner
from ui.views import AdminView



class AdminCog(commands.Cog):
    """Admin panel and cog reloading: /admin /reload"""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="admin", description="Admin panel: restart the bot, manage admins, ban users")
    async def admin(self, interaction: discord.Interaction):
        if not can_manage(interaction.user.id):
            await interaction.response.send_message(
                content="\u26a0\ufe0f Only the bot owner or promoted admins can use /admin.", ephemeral=True
            )
            return
        await interaction.response.send_message(
            content="\U0001f527 **Bot administration**\n\n"
                    "Pick an action. For user-based actions, enter the target's Discord user ID "
                    "(enable Developer Mode in Discord settings to see user IDs).",
            view=AdminView(interaction.user.id),
            ephemeral=True,
        )


    @app_commands.command(name="reload", description="Reload all cog extensions (owner only)")
    async def reload(self, interaction: discord.Interaction):
        if not is_owner(interaction.user.id):
            await interaction.response.send_message(
                content="\u26a0\ufe0f Only the bot owner can reload cogs.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        results = []
        for cog in list(self.bot.cogs.values()):
            extension = cog.__class__.__module__
            try:
                await self.bot.reload_extension(extension)
                results.append(f"\u2705 {extension}")
            except Exception as exc:
                results.append(f"\u274c {extension}: {exc}")
        try:
            await self.bot.tree.sync()
            results.append("\u2705 command tree synced")
        except Exception as exc:
            results.append(f"\u274c tree sync: {exc}")
        log.info("reload: %s", "; ".join(results))
        await interaction.edit_original_response(content="**Cog reload results:**\n" + "\n".join(results))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AdminCog(bot))
