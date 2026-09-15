

import asyncio
import io
import json
import logging
import os
import random
import subprocess
import sys

from PIL import Image
import discord
from discord import app_commands
from discord.ext import commands
from discord.ui import View, Button, Modal, TextInput, Select
from discord.enums import TextStyle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

from comfyui_client import ComfyUIClient, ComfyUIError
from config_loader import load_config
from http_session import get_session
import user_settings
import generation_store
import nsfw_guard
import moderation
config = load_config()
# Predefined bot owner (config.yaml: owner.id). The owner and promoted admins
# get elevated permissions (delete any output, /admin controls).
BOT_OWNER_ID = int(config.get("owner", {}).get("id", 0) or 0)

# Models available for each command (based on which spec each model defines).
T2I_MODELS = [m for m, cfg in config["models"].items() if "t2i" in cfg]
UPSCALE_MODELS = [m for m, cfg in config["models"].items() if "upscale" in cfg]
I2I_MODELS = [m for m, cfg in config["models"].items()
              if "i2i_single" in cfg or "i2i_multi" in cfg]
UPSCALE_CHOICES = [app_commands.Choice(name=m, value=m) for m in UPSCALE_MODELS]

# Human-friendly labels for the upscale model selection buttons.
UPSCALE_MODEL_LABELS = {
    "sdxl": "SDXL",
    "seedvr2": "SeedVR2",
    "flashvsr": "FlashVSR",
}
# Ideogram preset step options (map to the workflow's CustomCombo node).
IDEOGRAM_QUALITY_PRESETS = ("Turbo", "Default", "Quality")
QUALITY_CHOICES = [app_commands.Choice(name=p, value=p) for p in IDEOGRAM_QUALITY_PRESETS]

# Exact aspect ratio labels accepted by the ComfyUI ResolutionSelector node.
ASPECT_RATIOS = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]
ASPECT_RATIO_CHOICES = [app_commands.Choice(name=r, value=r) for r in ASPECT_RATIOS]
# Image-to-image workflow and sampler choices (Flux 2 Klein 4B Base).
I2I_WORKFLOW_CHOICES = [
    app_commands.Choice(name="1 image (edit)", value="single"),
    app_commands.Choice(name="2 images (combine)", value="multi"),
]
# Trimmed down to exactly 25 core options for native static dropdowns
SAMPLER_NAMES = [
    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", 
    "dpm_3", "dpm_3_ancestral", "dpmpp_2s_ancestral", "dpmpp_2m", 
    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", 
    "dpmpp_sde", "dpmpp_sde_gpu", "lcm", "ddim", "uni_pc", "uni_pc_bh2", 
    "sa_solver", "seeds_2", "gradient_estimation", "dpm_adaptive", 
    "ipndf", "deis"
]


SAMPLER_CHOICES = [
    app_commands.Choice(name=s, value=s)
    for s in SAMPLER_NAMES
]
SCHEDULER_NAMES = [
    "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"
]
SCHEDULER_CHOICES = [
    app_commands.Choice(name=s, value=s)
    for s in SCHEDULER_NAMES
]
def normalize_aspect_ratio(value: str | None) -> str | None:
    """Map user input to the exact ResolutionSelector label.

    Accepts the full label ("16:9 (Widescreen)") or a short form ("16:9").
    Unknown values are passed through unchanged.
    """
    if value is None:
        return None
    v = str(value).strip()
    if v in ASPECT_RATIOS:
        return v
    for r in ASPECT_RATIOS:
        if r.split(" ")[0] == v:
            return r
    return v
comfy = ComfyUIClient(config["comfyui"]["base_url"])

# NSFW guardrail config.
nsfw_cfg = config.get("nsfw", {})
nsfw_guard.configure(extra_terms=tuple(nsfw_cfg.get("extra_terms", [])))
nsfw_guard.configure_image_check(
    enabled=bool(nsfw_cfg.get("image_check", True)),
    threshold=float(nsfw_cfg.get("image_threshold", 0.5)),
)
# Per-user cooldown to prevent abuse of the GPU.
cooldowns: dict[str, float] = {}
MIN_INTERVAL_SECONDS = 20
# Entries older than this are pruned so the dict doesn't grow unboundedly.
COOLDOWN_TTL_SECONDS = MIN_INTERVAL_SECONDS * 2

# Track the last workflow file used per model so we can free memory
# when switching between workflows.
last_workflow_by_model: dict[str, str] = {}
# Track the last checkpoint (ckpt_name) used per model so we can free
# memory when switching to a different checkpoint within the same workflow.
last_ckpt_by_model: dict[str, str] = {}

# Progress bar settings for the generation progress message.
PROGRESS_BAR_WIDTH = 10
BAR_FULL = "\u2588"
BAR_EMPTY = "\u2591"
def progress_bar(level: int) -> str:
    """Build the bar string for a given level (0..PROGRESS_BAR_WIDTH)."""
    level = max(0, min(PROGRESS_BAR_WIDTH, level))
    return BAR_FULL * level + BAR_EMPTY * (PROGRESS_BAR_WIDTH - level)


class ProgressUpdater:
    """Edits a Discord message with ComfyUI's live generation progress.

    The message shows a progress bar plus the current percentage. Edits
    happen at most once per bar-level change so Discord rate limits are
    never approached. Setting ``done`` stops further edits once the final
    image (or an error) has been posted.

    When created with ``lane`` and ``label`` and then ``arm``ed with the job's
    coroutine, it also keeps the "You're #N in line" prefix live: as jobs
    ahead finish, the message's queue position is refreshed until our own job
    starts (after which normal progress edits take over).
    """

    def __init__(self, message, lane=None, label=None):
        # ``message`` is the original response message (an InteractionMessage)
        # returned by interaction.original_response(); it is edited in place
        # to show progress.
        self.message = message
        self.level = 0
        self.pct = 0.0
        self.done = False
        self.lane = lane
        self.label = label or "\U0001f3a8 Generating image\u2026"
        self.job = None
        self.started = False
        self._pos = None

    def arm(self, job):
        """Attach the job this updater is waiting for and start tracking the queue.

        ``arm`` must be called *after* the job is submitted to the lane so that
        ``live_position`` includes our job. Our waiting-line position equals the
        number of queued jobs (jobs ahead of us + us), matching the initial
        "You're #N in line" prefix shown on the progress message. We then
        decrement ``_pos`` by one for every job ahead that finishes, so the
        displayed position drops smoothly (#4 -> #3 -> #2 -> #1).
        """
        self.job = job
        if self.lane:
            from job_queue import job_queue
            self._pos = job_queue.live_position(self.lane) or 1
            job_queue.register(self.lane, self._on_queue_event)

    def _unregister(self):
        """Remove this updater from the lane's listeners."""
        if self.lane:
            from job_queue import job_queue
            job_queue.unregister(self.lane, self._on_queue_event)

    def disarm(self):
        """Stop tracking the queue (call when the job is abandoned before it runs)."""
        self._unregister()

    def update(self, progress: float) -> None:
        if self.done:
            return
        pct = max(0.0, min(100.0, progress * 100))
        self.pct = pct
        level = max(0, min(PROGRESS_BAR_WIDTH, round(pct / 100 * PROGRESS_BAR_WIDTH)))
        if level <= self.level:
            return
        self.level = level
        asyncio.create_task(self._edit(level, pct))

    async def _edit(self, level: int, pct: float) -> None:
        if self.done:
            return
        try:
            await self.message.edit(
                content=f"{self.label} [{progress_bar(level)}] {int(pct)}%"
            )
        except Exception as exc:
            log.warning("progress edit failed: %s", exc)

    async def _on_queue_event(self, event, coro):
        # Stop tracking once our job has started running or the session ended.
        if self.done or self.started:
            self._unregister()
            return
        if event == "start":
            if coro is self.job:
                self.started = True
                self._unregister()
                # Our job has begun: clear the "in line" prefix immediately so
                # the message shows the progress bar (no queue prefix) right
                # away instead of waiting for the first progress tick.
                asyncio.create_task(self._edit(self.level, self.pct))
            return
        # A job ahead of us finished: move up one slot in line. We track our
        # position with a counter (decrementing once per job ahead that
        # finishes) rather than re-reading queue stats, so the displayed number
        # drops smoothly (#4 -> #3 -> #2 -> #1) even if this callback runs
        # after the next job has already started. When we are already #1 we
        # simply stop showing a position (our job will start next).
        if self._pos is None or self._pos <= 1:
            return
        self._pos -= 1
        prefix = f"\u23f3 You're #{self._pos} in line.\n\n"
        try:
            await self.message.edit(
                content=prefix + f"{self.label} [{progress_bar(self.level)}] {int(self.pct)}%"
            )
        except Exception as exc:
            log.warning("queue position edit failed: %s", exc)


class QueueWaitUpdater:
    """Keeps a message's "You're #N in line" prefix live while its job waits
    in a queue lane (used by /gen_prompt's LLM session).

    Unlike ``ProgressUpdater`` there is no progress bar: the message shows
    only the queue position above a static label. ``arm`` must be called
    *after* the job is submitted. When the lane is idle (the job will start
    immediately) the updater stays silent so the interaction keeps Discord's
    native "thinking" state until the job edits the message itself.
    """

    def __init__(self, message, lane, label):
        # ``message`` is the interaction's original response message; it is
        # edited in place to show the waiting-line status.
        self.message = message
        self.lane = lane
        self.label = label
        self.job = None
        self.started = False
        self._stopped = False
        self._pos = None
        self._edit_task = None

    def arm(self, job):
        """Attach the queued job and start tracking the line.

        Must be called after the job is submitted so the lane stats include
        it: ``pending`` then counts our own job, so the waiting-line position
        equals ``pending`` (jobs ahead + 1). When the lane is idle nothing is
        shown and no listener is registered.
        """
        self.job = job
        from job_queue import job_queue
        active, pending = job_queue.lane_stats(self.lane)
        if active == 0 and pending <= 1:
            # Starts immediately; leave the "thinking" state untouched.
            return
        self._pos = pending
        job_queue.register(self.lane, self._on_queue_event)
        self._schedule_edit(self._pos)

    def _unregister(self):
        if self.lane:
            from job_queue import job_queue
            job_queue.unregister(self.lane, self._on_queue_event)

    def disarm(self):
        """Stop tracking the queue; no further status edits may be scheduled.

        Called when the job is abandoned or has already failed: a "start"
        event can still be fired afterwards (the worker fires it before the
        awaiting caller observes the failure), so ``_stopped`` guards against
        a late clear-prefix edit overwriting an error message.
        """
        self._stopped = True
        if self._edit_task is not None and not self._edit_task.done():
            self._edit_task.cancel()
        self._unregister()

    def _schedule_edit(self, pos: int | None) -> None:
        if self._stopped:
            return
        # Track the latest edit task so ``disarm`` can cancel a still-pending
        # status edit before an error message is written over it.
        self._edit_task = asyncio.create_task(self._edit(pos))

    async def _edit(self, pos: int | None) -> None:
        try:
            if pos is None:
                # Our job started: drop the prefix; the job's own edits
                # (picker, results, errors) take over from here.
                await self.message.edit(content=self.label)
            else:
                await self.message.edit(
                    content=f"\u23f3 You're #{pos} in line.\n\n{self.label}"
                )
        except Exception as exc:
            log.warning("queue status edit failed: %s", exc)

    async def _on_queue_event(self, event, coro):
        # Stop tracking once our job has started running.
        if self.started:
            self._unregister()
            return
        if event == "start":
            if coro is self.job:
                self.started = True
                self._unregister()
                # Clear the "in line" prefix right away so the message shows
                # only the label while the job prepares.
                self._schedule_edit(None)
            return
        # A job ahead of us finished: move up one slot in line. The position
        # is tracked with a decrementing counter (mirroring ProgressUpdater)
        # so it drops smoothly (#3 -> #2 -> #1); at #1 we stop editing, as our
        # job will start next.
        if self._pos is None or self._pos <= 1:
            return
        self._pos -= 1
        self._schedule_edit(self._pos)


def _parse_opt_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_opt_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_opt_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None
ERROR_LIFETIME_SECONDS = 5.0


async def _delete_message_after(message, delay: float):
    await asyncio.sleep(delay)
    try:
        await message.delete()
    except Exception as exc:
        log.warning("Failed to delete transient error message: %s", exc)


async def _delete_original_response_after(interaction, delay: float):
    await asyncio.sleep(delay)
    try:
        await interaction.delete_original_response()
    except Exception as exc:
        log.warning("Failed to delete transient error response: %s", exc)


def schedule_message_deletion(message, delay: float = ERROR_LIFETIME_SECONDS):
    """Schedule deletion of a transient error message after a short delay."""
    asyncio.create_task(_delete_message_after(message, delay))


def schedule_original_response_deletion(interaction, delay: float = ERROR_LIFETIME_SECONDS):
    """Schedule deletion of an interaction's original response message."""
    asyncio.create_task(_delete_original_response_after(interaction, delay))


async def reply_error(interaction: discord.Interaction, message: str, target=None):
    """Post an error message and schedule its deletion.

    If ``target`` is given, the error is written into that message (typically
    the progress message) instead of the interaction's original response.
    """
    try:
        if target is not None:
            msg = await target.edit(content=message)
            schedule_message_deletion(msg)
        else:
            # edit_original_response() returns an InteractionCallbackResponse,
            # which has no .delete(); use the interaction's own delete method.
            await interaction.edit_original_response(content=message)
            asyncio.create_task(_delete_original_response_after(interaction, ERROR_LIFETIME_SECONDS))
    except Exception:
        try:
            msg = await interaction.channel.send(content=message)
            schedule_message_deletion(msg)
        except Exception:
            # Interaction expired or followup failed; log so it's not lost silently.
            logging.getLogger("bot").exception("Could not deliver error reply: %s", message)
            return


async def ban_guard(interaction: discord.Interaction) -> bool:
    """True if the user is banned; sends the ban notice.

    Returns True when the action must be aborted, False when it may proceed.
    """
    if moderation.is_banned(interaction.user.id):
        await interaction.response.send_message(
            content="\U0001f6ab You are banned from using this bot. Please contact an admin.", ephemeral=True
        )
        return True
    return False


def nsfw_blocked(interaction, prompt: str | None) -> bool:
    """Return True if the request must be refused under the NSFW guardrail.

    A request is refused only when its prompt is NSFW *and* the channel is
    not Discord-marked NSFW. NSFW prompts inside an NSFW-marked channel are
    allowed. Negative prompts are not checked, since "nsfw" often appears
    there meaning "exclude nsfw".
    """
    nsfw_prompt = bool(prompt) and nsfw_guard.is_nsfw(prompt)
    nsfw_channel = nsfw_guard.is_nsfw_channel(interaction.channel)
    log.info(
        "nsfw_check: prompt_nsfw=%s channel_nsfw=%s channel=%r",
        nsfw_prompt, nsfw_channel, getattr(interaction.channel, "name", "?"),
    )
    return nsfw_prompt and not nsfw_channel


def can_manage(user_id: int) -> bool:
    """True for the predefined owner or any promoted admin.

    Grants: deleting any generation and banning/unbanning users.
    """
    return user_id == BOT_OWNER_ID or moderation.is_admin(user_id)


def is_owner(user_id: int) -> bool:
    """True only for the predefined bot owner.

    Owner-only powers: restarting the bot and promoting/demoting admins.
    """
    return BOT_OWNER_ID != 0 and user_id == BOT_OWNER_ID


async def check_cooldown(interaction) -> bool:
    import asyncio

    uid = str(interaction.user.id)
    now = asyncio.get_running_loop().time()
    # Prune stale entries so the cooldown dict does not grow unboundedly.
    cutoff = now - COOLDOWN_TTL_SECONDS
    if len(cooldowns) > 100:
        stale = [u for u, t in cooldowns.items() if t < cutoff]
        for u in stale:
            cooldowns.pop(u, None)
    last = cooldowns.get(uid, 0)
    if now - last < MIN_INTERVAL_SECONDS:
        return False
    cooldowns[uid] = now
    return True
def uuid_hex() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


DISCORD_IMAGE_LIMIT = 18 * 1024 * 1024  # Stay comfortably under Discord's ~20 MB upload limit
ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024  # Hard cap for inbound attachments


async def download_image(url) -> bytes:
    """Download a Discord attachment, checking it really is a reasonably-sized image."""
    session = get_session()
    async with session.get(str(url)) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if ctype and not ctype.startswith("image/"):
            raise ValueError(f"attachment is not an image (Content-Type: {ctype})")
        data = await resp.read()
    if len(data) > ATTACHMENT_MAX_BYTES:
        raise ValueError(f"attachment exceeds the {ATTACHMENT_MAX_BYTES // (1024 * 1024)} MB limit")
    return data


def compress_image(data: bytes) -> tuple[bytes, str]:
    """Return ``(bytes, extension)`` safe for a Discord upload.

    Images already at or below ``DISCORD_IMAGE_LIMIT`` pass through unchanged
    (no quality loss). Larger images are re-encoded as JPEG at progressively
    lower quality until they fit, preserving the original resolution and as
    much quality as possible.
    """
    if len(data) <= DISCORD_IMAGE_LIMIT:
        return data, ".png"
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    for quality in (90, 85, 80, 75, 70):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if len(buf.getvalue()) <= DISCORD_IMAGE_LIMIT:
            return buf.getvalue(), ".jpg"
    # Even at quality 70 it fits in memory; send it (far smaller than the original).
    return buf.getvalue(), ".jpg"


def image_resolution(data: bytes) -> str:
    """Return the "WxH" resolution string for an image's bytes."""
    img = Image.open(io.BytesIO(data))
    return f"{img.size[0]}x{img.size[1]}"


def meta_lines(meta: dict) -> list[str]:
    """Build embed lines for the seed, steps and CFG actually used.

    Only keys that are present are rendered, so SDXL outputs show steps
    and CFG while other models show just the seed.
    """
    lines = []
    if meta.get("seed") is not None:
        lines.append(f"**Seed:** {meta['seed']}")
    if meta.get("steps") is not None:
        lines.append(f"**Steps:** {meta['steps']}")
    if meta.get("cfg") is not None:
        lines.append(f"**CFG:** {meta['cfg']:g}")
    if meta.get("sampler") is not None:
        lines.append(f"**Sampler:** {meta['sampler']}")
    if meta.get("scheduler") is not None:
        lines.append(f"**Scheduler:** {meta['scheduler']}")
    return lines


async def deliver_generation(interaction, *, images, meta, base_desc, color,
                             spec, model, suffix, stealth, save_kwargs,
                             target=None):
    """Shared post-generation pipeline used by every output path.

    Runs the NSFW image gate, compresses images to Discord-safe uploads,
    replaces the progress message with the result embed + buttons, and
    persists the retry parameters in generation_store under the new
    message id. ``target`` is the follow-up message to edit (button/picker
    flows); when None, the interaction's original response is edited.
    """
    from ui.views import GenerationView  # lazy: ui.views imports this module
    blocked = "\u26a0\ufe0f Image blocked: NSFW content is only allowed in NSFW channels."
    for img in images:
        if await nsfw_guard.check_image_nsfw(img, interaction):
            if target is None:
                await interaction.edit_original_response(content=blocked)
                schedule_original_response_deletion(interaction)
            else:
                msg = await target.edit(content=blocked)
                schedule_message_deletion(msg)
            return
    files = []
    for i, img in enumerate(images):
        img_bytes, ext = compress_image(img)
        files.append(discord.File(io.BytesIO(img_bytes), filename=f"{model}_{suffix}_{i}{ext}"))
    embed = discord.Embed(
        description=base_desc + "\n" + "\n".join(meta_lines(meta)),
        color=discord.Color(color),
    )
    edit = interaction.edit_original_response if target is None else target.edit
    response_msg = await edit(content="", embed=embed, attachments=files,
                              view=GenerationView(stealth=stealth))
    generation_store.save(response_msg.id, {
        "spec": spec, "model": model, "suffix": suffix, "stealth": stealth,
        "embed_desc": base_desc, "embed_color": int(embed.color),
        "user_id": interaction.user.id,
        "kwargs": save_kwargs,
    })
