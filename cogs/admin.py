import discord

from bot import bot
from core import log, moderation, BOT_OWNER_ID, can_manage, is_owner
from ui.views import AdminView


@bot.tree.command(name="admin", description="Admin panel: restart the bot, manage admins, ban users")
async def admin(interaction: discord.Interaction):
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
