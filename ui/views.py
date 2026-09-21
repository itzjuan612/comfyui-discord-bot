import asyncio
import io
import os
import subprocess
import sys
import logging

from PIL import Image
import discord
from discord.ui import View, Button, Modal, TextInput, Select
from discord.enums import TextStyle

from bot import bot
from core import (
    config, comfy, log, generation_store, user_settings, moderation,
    BOT_OWNER_ID, SAMPLER_NAMES, SCHEDULER_NAMES, UPSCALE_MODELS, UPSCALE_MODEL_LABELS,
    image_resolution, uuid_hex,
    reply_error, ban_guard, check_cooldown, can_manage, is_owner, download_image,
    nsfw_blocked, deliver_generation,
    schedule_message_deletion, schedule_original_response_deletion,
    ProgressUpdater,
    _parse_opt_float, _parse_opt_int, _parse_opt_str,
)
from workflow import run_image, run_text_workflow
from llm_client import call_llm, resolve_reasoning_effort, llm_model_unload
from job_queue import job_queue


class RetryButton(Button):
    def __init__(self):
        super().__init__(label="Retry", emoji="\U0001f504", custom_id="retry_generation")

    async def callback(self, interaction: discord.Interaction):
        if await ban_guard(interaction):
            return
        log.info("Retry clicked for message %s", interaction.message.id)
        params = generation_store.get(interaction.message.id)
        if params is None:
            await interaction.response.send_message(
                content="\u26a0\ufe0f This generation can no longer be retried.", ephemeral=True
            )
            return
        if not await check_cooldown(interaction):
            await interaction.response.send_message(
                content="\u23f3 Please wait before retrying.", ephemeral=True
            )
            return
        # The stored prompt was gated when originally generated, but the
        # channel may not be NSFW-marked anymore; re-check before GPU work.
        saved_prompt = params.get("kwargs", {}).get("prompt")
        if saved_prompt and nsfw_blocked(interaction, saved_prompt):
            await interaction.response.send_message(
                content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.",
                ephemeral=True,
            )
            msg = await interaction.original_response()
            schedule_message_deletion(msg)
            return
        stealth = bool(params.get("stealth", False))
        await interaction.response.send_message(
            content=job_queue.waiting_prefix("comfyui") + "\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress_msg = await interaction.original_response()
        progress = ProgressUpdater(progress_msg, lane="comfyui")
        job = run_image(
            params["spec"], on_progress=progress.update,
            model_key=params["model"], **params["kwargs"]
        )
        fut = job_queue.submit(job, lane="comfyui", name=f"retry_{params['model']}")
        progress.arm(job)
        try:
            images, meta = await fut
            progress.done = True
            # Edit the progress message in place so the output replaces it.
            # deliver_generation persists the params under the new message id,
            # tagged with the clicking user so only they can delete the output.
            await deliver_generation(
                interaction, images=images, meta=meta,
                base_desc=params["embed_desc"], color=params["embed_color"],
                spec=params["spec"], model=params["model"],
                suffix=params["suffix"], stealth=stealth,
                save_kwargs=params["kwargs"],
            )
        except Exception as exc:
            progress.done = True
            log.exception("retry failed")
            await interaction.edit_original_response(content=f"\u274c Retry failed: {exc}")
            schedule_original_response_deletion(interaction)


class DeleteButton(Button):
    def __init__(self):
        super().__init__(label="Delete", emoji="\U0001f5d1\ufe0f", custom_id="delete_generation")

    async def callback(self, interaction: discord.Interaction):
        if await ban_guard(interaction):
            return
        log.info("Delete clicked for message %s by %s", interaction.message.id, interaction.user.id)
        params = generation_store.get(interaction.message.id)
        if params is not None:
            owner = params.get("user_id")
            if owner is not None and owner != interaction.user.id and not can_manage(interaction.user.id):
                await interaction.response.send_message(
                    content="\u26a0\ufe0f You can only delete images you generated yourself.",
                    ephemeral=True,
                )
                return
        await interaction.response.defer()
        deleted = True
        try:
            await interaction.message.delete()
        except discord.Forbidden:
            deleted = False
            log.warning(
                "Delete failed for message %s: missing Manage Messages permission",
                interaction.message.id,
            )
            await interaction.followup.send(
                content="\u26a0\ufe0f I don't have permission to delete messages in this channel.",
                ephemeral=True,
            )
        except discord.NotFound:
            log.info(
                "Message %s was already deleted",
                interaction.message.id,
            )
        # Only drop the stored retry params once the message is actually gone
        # (NotFound counts: the entry is then orphaned). On Forbidden, keep
        # them so the retry buttons keep working.
        if deleted:
            generation_store.pop(interaction.message.id)


class UpscaleModelButton(Button):
    """One of the model pickers shown after clicking 'Upscale 2x'."""

    def __init__(self, model: str, label: str):
        super().__init__(label=label, emoji="\U0001f505", custom_id=f"upscale_model_{model}")
        self.model = model

    async def callback(self, interaction: discord.Interaction):
        if await ban_guard(interaction):
            return
        view: UpscaleModelView = self.view
        model = self.model
        stealth = view.stealth
        log.info("Upscale model %s clicked", model)
        if not await check_cooldown(interaction):
            await interaction.response.send_message(
                content="\u23f3 Please wait before requesting another image.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=stealth)
        progress_msg = await interaction.followup.send(
            content=job_queue.waiting_prefix("comfyui") + "\U0001f3a8 Upscaling image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress = ProgressUpdater(progress_msg, lane="comfyui",
                                   label="\U0001f3a8 Upscaling image\u2026")
        spec = config["models"].get(model, {}).get("upscale")
        if spec is None:
            progress.done = True
            await reply_error(interaction, f"\u274c Model {model!r} has no upscaling workflow.", target=progress_msg)
            return
        settings = user_settings.get_settings(interaction.user.id)
        negative = settings["negative_prompt"] or None
        # For SDXL, reuse the checkpoint the source image was created with
        # so the upscale matches the original model, and reuse the sampler
        # and scheduler the original image was generated with.
        ckpt_name = view.ckpt_name if model == "sdxl" else None
        sampler = view.sampler if model == "sdxl" else None
        scheduler = view.scheduler if model == "sdxl" else None
        job = run_image(
            spec, on_progress=progress.update, model_key=model, prompt=None,
            negative=negative, strength=None,
            image_filename=view.uploaded_name, scale=2,
            input_longest_side=view.input_longest_side,
            ckpt_name=ckpt_name, sampler=sampler, scheduler=scheduler,
        )
        fut = job_queue.submit(job, lane="comfyui", name=f"{model}_upscale")
        progress.arm(job)
        try:
            images, meta = await fut
            progress.done = True
            display_ckpt = meta.get("ckpt_name") or ckpt_name
            base_desc = f"**Model:** {model}\n**Scale:** 2x"
            if model == "sdxl" and display_ckpt:
                base_desc += f"\n**Checkpoint:** {display_ckpt}"
            base_desc += f"\n**Resolution:** {image_resolution(images[0])}"
            # Edit the progress (follow-up) message, NOT the ephemeral
            # model-picker message, so the output is visible to everyone.
            await deliver_generation(
                interaction, images=images, meta=meta, base_desc=base_desc,
                color=int(discord.Color.green()), spec=spec, model=model,
                suffix="upscale", stealth=stealth, target=progress_msg,
                # Retries reuse the uploaded input image and roll a fresh seed.
                save_kwargs={"prompt": None, "negative": negative, "strength": None,
                             "image_filename": view.uploaded_name, "scale": 2,
                             "input_longest_side": view.input_longest_side,
                             "ckpt_name": ckpt_name, "sampler": sampler,
                             "scheduler": scheduler},
            )
        except Exception as exc:
            progress.done = True
            logging.getLogger("bot").exception("upscale failed")
            await reply_error(interaction, f"\u274c Upscaling failed: {exc}", target=progress_msg)


class UpscaleModelView(View):
    """Ephemeral model picker shown after clicking 'Upscale 2x'.

    Carries the uploaded input image name and input resolution so the chosen
    model button can run the upscale without re-downloading the image.
    """

    def __init__(self, uploaded_name: str, input_longest_side: int, stealth: bool = False,
                 source_model: str | None = None, ckpt_name: str | None = None,
                 sampler: str | None = None, scheduler: str | None = None):
        super().__init__(timeout=300)
        self.uploaded_name = uploaded_name
        self.input_longest_side = input_longest_side
        self.stealth = stealth
        self.ckpt_name = ckpt_name
        self.sampler = sampler
        self.scheduler = scheduler
        # SDXL upscale only works well with SDXL checkpoints, so hide the SDXL
        # option when the source image was not generated with SDXL.
        for model in UPSCALE_MODELS:
            if model == "sdxl" and source_model != "sdxl":
                continue
            label = UPSCALE_MODEL_LABELS.get(model, model)
            self.add_item(UpscaleModelButton(model, label))

    async def on_timeout(self):
        try:
            await self.message.edit(content="\u23f3 Upscale model selection expired; upscale cancelled.")
        except Exception:
            pass

    async def on_error(self, interaction: discord.Interaction, error: Exception, item):
        log.warning("UpscaleModelView error: %s", error)


class CheckpointPickerView(View):
    """Ephemeral select-menu picker shown when /upscale uses the sdxl workflow.

    Modeled on ``ThinkingView``: a single ``Select`` listing every checkpoint
    in models/checkpoints (plus "Default", which uses the workflow's own
    checkpoint with automatic fallback). Carries the uploaded input image name,
    resolution, and generation params so the selected option can run the
    upscale without re-downloading the image.
    """

    def __init__(self, spec: dict, model_key: str, uploaded_name: str,
                 input_longest_side: int | None, stealth: bool,
                 prompt: str | None, negative: str | None,
                 strength: float | None, scale: float | None,
                 checkpoints: list[str],
                 sampler: str | None = None, scheduler: str | None = None):
        super().__init__(timeout=300)
        self.spec = spec
        self.model_key = model_key
        self.uploaded_name = uploaded_name
        self.input_longest_side = input_longest_side
        self.stealth = stealth
        self.prompt = prompt
        self.negative = negative
        self.strength = strength
        self.scale = scale
        self.sampler = sampler
        self.scheduler = scheduler
        # "default" maps to the workflow's own checkpoint (with automatic
        # fallback to an available SDXL checkpoint in run_image).
        options = [discord.SelectOption(label="Default (workflow checkpoint)", value="default")]
        options += [discord.SelectOption(label=c, value=c) for c in checkpoints[:24]]
        self.add_item(CheckpointSelect(
            placeholder="Select a checkpoint",
            options=options,
            min_values=1, max_values=1,
            custom_id="upscale_checkpoint",
        ))

    async def handle_select(self, interaction: discord.Interaction, value: str):
        if await ban_guard(interaction):
            return
        # The /upscale prompt was gated at command time, but the channel's
        # NSFW marking may have changed while the picker was open.
        if self.prompt and nsfw_blocked(interaction, self.prompt):
            await interaction.response.send_message(
                content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.",
                ephemeral=True,
            )
            msg = await interaction.original_response()
            schedule_message_deletion(msg)
            return
        self.stop()
        ckpt_name = None if value == "default" else value
        log.info("Upscale checkpoint %s selected", ckpt_name or "default")
        # The /upscale command already consumed the cooldown when invoked;
        # the select click is the continuation of that same request, so we do
        # not re-check (re-checking would always fail and block the run).
        await interaction.response.defer(ephemeral=self.stealth)
        progress_msg = await interaction.followup.send(
            content=job_queue.waiting_prefix("comfyui") + "\U0001f3a8 Upscaling image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=self.stealth,
        )
        progress = ProgressUpdater(progress_msg, lane="comfyui",
                                   label="\U0001f3a8 Upscaling image\u2026")
        job = run_image(
            self.spec, on_progress=progress.update, model_key=self.model_key,
            prompt=self.prompt, negative=self.negative, strength=self.strength,
            image_filename=self.uploaded_name, scale=self.scale,
            input_longest_side=self.input_longest_side,
            ckpt_name=ckpt_name, sampler=self.sampler, scheduler=self.scheduler,
        )
        fut = job_queue.submit(job, lane="comfyui", name=f"{self.model_key}_upscale")
        progress.arm(job)
        try:
            images, meta = await fut
            progress.done = True
            display_ckpt = meta.get("ckpt_name") or ckpt_name
            base_desc = (
                f"**Model:** {self.model_key}"
                + (f"\n**Checkpoint:** {display_ckpt}" if display_ckpt else "")
                + (f"\n**Scale:** {self.scale:g}x" if self.scale is not None else "")
                + f"\n**Resolution:** {image_resolution(images[0])}"
            )
            await deliver_generation(
                interaction, images=images, meta=meta, base_desc=base_desc,
                color=int(discord.Color.green()), spec=self.spec,
                model=self.model_key, suffix="upscale", stealth=self.stealth,
                target=progress_msg,
                save_kwargs={"prompt": self.prompt, "negative": self.negative,
                             "strength": self.strength,
                             "image_filename": self.uploaded_name,
                             "scale": self.scale,
                             "input_longest_side": self.input_longest_side,
                             "ckpt_name": ckpt_name,
                             "sampler": self.sampler,
                             "scheduler": self.scheduler},
            )
        except Exception as exc:
            progress.done = True
            logging.getLogger("bot").exception("checkpoint upscale failed")
            await reply_error(interaction, f"\u274c Upscaling failed: {exc}", target=progress_msg)

    async def on_timeout(self):
        try:
            await self.message.edit(content="\u23f3 Checkpoint selection expired; upscale cancelled.")
        except Exception:
            pass

    async def on_error(self, interaction: discord.Interaction, error: Exception, item):
        log.warning("CheckpointPickerView error: %s", error)


class CheckpointSelect(Select):
    """The checkpoint select item shown by the /upscale sdxl picker.

    discord.py dispatches select-menu interactions to the component's own
    ``callback`` method, NOT to a View-level handler — so the handler lives
    on the select itself (same pattern as ``ThinkingSelect``).
    """

    async def callback(self, interaction: discord.Interaction):
        await self.view.handle_select(interaction, self.values[0])


class UpscaleButton(Button):
    """'Upscale 2x' button shown on every generated output.

    Clicking it downloads the image from the message, uploads it to ComfyUI,
    then replies with an ephemeral message asking which model to use.
    """

    def __init__(self):
        super().__init__(label="Upscale 2x", emoji="\U0001f50d", custom_id="upscale_generation")

    async def callback(self, interaction: discord.Interaction):
        if await ban_guard(interaction):
            return
        params = generation_store.get(interaction.message.id)
        stealth = bool(params.get("stealth", False)) if params else False
        # The checkpoint the source image was created with (from /sdxl's saved
        # kwargs). Used so a subsequent SDXL upscale reuses the same checkpoint.
        source_model = params.get("model") if params else None
        saved_kwargs = params.get("kwargs", {}) if params else {}
        ckpt_name = saved_kwargs.get("ckpt_name")
        sampler = saved_kwargs.get("sampler")
        scheduler = saved_kwargs.get("scheduler")
        log.info("Upscale 2x clicked for message %s", interaction.message.id)
        image_attachments = [
            a for a in interaction.message.attachments
            if a.content_type and a.content_type.startswith("image/")
        ]
        if not image_attachments:
            await interaction.response.send_message(
                content="\u26a0\ufe0f This message has no image to upscale.", ephemeral=True
            )
            return
        source = image_attachments[0]
        try:
            data = await download_image(source.url)
        except Exception as exc:
            log.warning("Could not download source image: %s", exc)
            await interaction.response.send_message(
                content=f"\u274c Could not download the image: {exc}", ephemeral=True
            )
            return
        img = Image.open(io.BytesIO(data))
        input_longest_side = max(img.size)
        uploaded_name = await comfy.upload_image(data, f"discord_{uuid_hex()}.png")
        await interaction.response.send_message(
            content="Which model should upscale this image?",
            ephemeral=True,
            view=UpscaleModelView(
                uploaded_name=uploaded_name, input_longest_side=input_longest_side,
                stealth=stealth, source_model=source_model, ckpt_name=ckpt_name,
                sampler=sampler, scheduler=scheduler,
            ),
        )


class FluxEditModal(Modal):
    """Modal form for editing a generated image via the single-image
    (1 image / edit) flux_edit workflow using Flux 2 Klein 4B Base.

    The first image of the clicked message is used as the source. The form
    takes prompt, steps, sampler, and privacy (stealth/public); CFG is not
    a field and always falls back to the user's saved flux_edit default.
    Left-empty fields fall back to the user's saved flux_edit defaults.
    """

    def __init__(self, default_stealth: bool = False):
        super().__init__(title="Flux Edit (1-image workflow)")
        self.default_stealth = default_stealth
        self.prompt_input = TextInput(
            label="Prompt", style=TextStyle.paragraph,
            placeholder="Describe the edit", required=True,
        )
        self.steps_input = TextInput(
            label="Steps (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. 30",
        )
        self.sampler_input = TextInput(
            label="Sampler (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. euler",
        )
        self.privacy_input = TextInput(
            label="Privacy (optional)", style=TextStyle.short, required=False,
            placeholder="stealth = only you see it; public = visible to everyone",
        )
        self.add_item(self.prompt_input)
        self.add_item(self.steps_input)
        self.add_item(self.sampler_input)
        self.add_item(self.privacy_input)

    async def on_submit(self, interaction: discord.Interaction):
        prompt = self.prompt_input.value.strip()
        privacy_raw = (self.privacy_input.value or "").strip().lower()
        if privacy_raw == "stealth":
            stealth = True
        elif privacy_raw in ("public", "visible", ""):
            stealth = self.default_stealth
        else:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Privacy must be \u201cstealth\u201d or \u201cpublic\u201d.", ephemeral=True
            )
            return
        # CFG is not a modal field; it always falls back to the saved flux_edit default.
        cfg = None
        steps = _parse_opt_int(self.steps_input.value)
        sampler = _parse_opt_str(self.sampler_input.value)
        # Megapixels is no longer a modal field; fall back to the saved flux_edit default.
        megapixels = None

        if sampler is not None and sampler not in SAMPLER_NAMES:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Unknown sampler \u201c" + sampler + "\u201d.", ephemeral=True
            )
            return

        if nsfw_blocked(interaction, prompt):
            await interaction.response.send_message(
                content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.",
                ephemeral=True,
            )
            msg = await interaction.original_response()
            schedule_message_deletion(msg)
            return

        if not await check_cooldown(interaction):
            await interaction.response.send_message(
                content="\u23f3 Please wait before requesting another image.", ephemeral=True
            )
            return

        image_attachments = [
            a for a in interaction.message.attachments
            if a.content_type and a.content_type.startswith("image/")
        ]
        if not image_attachments:
            await interaction.response.send_message(
                content="\u26a0\ufe0f This message has no image to edit.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=stealth)
        progress_msg = await interaction.followup.send(
            content=job_queue.waiting_prefix("comfyui") + "\U0001f3a8 Editing image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress = ProgressUpdater(progress_msg, lane="comfyui",
                                   label="\U0001f3a8 Editing image\u2026")
        log.info("Flux Edit clicked for message %s", interaction.message.id)

        model = "flux2_klein"
        spec = config["models"].get(model, {}).get("i2i_single")
        if spec is None:
            progress.done = True
            await reply_error(interaction, "\u274c The single-image edit workflow is not configured.", target=progress_msg)
            return

        # Fall back to the user's saved flux_edit defaults for unset fields.
        saved = user_settings.get_settings(interaction.user.id)
        if cfg is None:
            cfg = saved.get("flux_edit_cfg")
        if steps is None:
            steps = saved.get("flux_edit_steps")
        if sampler is None:
            sampler = saved.get("flux_edit_sampler")
        if megapixels is None:
            megapixels = saved.get("flux_edit_megapixels")

        try:
            data1 = await download_image(image_attachments[0].url)
            uploaded1 = await comfy.upload_image(data1, f"discord_{uuid_hex()}.png")
        except Exception as exc:
            progress.done = True
            log.exception("Flux Edit download/upload failed")
            await reply_error(interaction, f"\u274c Could not process the source image: {exc}", target=progress_msg)
            return

        gen_kwargs = {
            "prompt": prompt,
            "seed": None,
            "cfg": cfg,
            "steps": steps,
            "sampler": sampler,
            "megapixels": megapixels,
            "image_filename": uploaded1,
        }

        try:
            job = run_image(spec, on_progress=progress.update, model_key=model, **gen_kwargs)
            fut = job_queue.submit(job, lane="comfyui", name="flux2_klein_i2i")
            progress.arm(job)
            images, meta = await fut
            progress.done = True
            base_lines = [
                f"**Model:** {model}",
                "**Workflow:** 1 image (edit)",
                f"**Prompt:** {prompt[:3800]}",
                f"**Resolution:** {image_resolution(images[0])}",
            ]
            base_desc = "\n".join(base_lines)
            await deliver_generation(
                interaction, images=images, meta=meta, base_desc=base_desc,
                color=int(discord.Color.green()), spec=spec, model=model,
                suffix="i2i", stealth=stealth, target=progress_msg,
                save_kwargs={**gen_kwargs, "seed": None},
            )
        except Exception as exc:
            progress.done = True
            log.exception("Flux Edit failed")
            await reply_error(interaction, f"\u274c Image edit failed: {exc}", target=progress_msg)
class QwenEditModal(Modal):
    """Modal form for editing a generated image via the single-image
    Qwen Image 2.1 edit workflow.

    The first image of the clicked message is used as the source. The form
    takes prompt, steps, sampler, scheduler, and privacy (stealth/public);
    CFG is not a field and always falls back to the user's saved Qwen
    default. Left-empty fields fall back to the user's saved Qwen defaults.
    Output resolution follows the input image.
    """

    def __init__(self, default_stealth: bool = False):
        super().__init__(title="Qwen Edit (1-image workflow)")
        self.default_stealth = default_stealth
        self.prompt_input = TextInput(
            label="Prompt", style=TextStyle.paragraph,
            placeholder="Describe the edit", required=True,
        )
        self.steps_input = TextInput(
            label="Steps (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. 25",
        )
        self.sampler_input = TextInput(
            label="Sampler (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. euler",
        )
        self.scheduler_input = TextInput(
            label="Scheduler (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. simple",
        )
        self.privacy_input = TextInput(
            label="Privacy (optional)", style=TextStyle.short, required=False,
            placeholder="stealth = only you see it; public = visible to everyone",
        )
        self.add_item(self.prompt_input)
        self.add_item(self.steps_input)
        self.add_item(self.sampler_input)
        self.add_item(self.scheduler_input)
        self.add_item(self.privacy_input)

    async def on_submit(self, interaction: discord.Interaction):
        prompt = self.prompt_input.value.strip()
        privacy_raw = (self.privacy_input.value or "").strip().lower()
        if privacy_raw == "stealth":
            stealth = True
        elif privacy_raw in ("public", "visible", ""):
            stealth = self.default_stealth
        else:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Privacy must be \u201cstealth\u201d or \u201cpublic\u201d.", ephemeral=True
            )
            return
        # CFG is not a modal field; it always falls back to the saved Qwen default.
        cfg = None
        steps = _parse_opt_int(self.steps_input.value)
        sampler = _parse_opt_str(self.sampler_input.value)
        scheduler = _parse_opt_str(self.scheduler_input.value)

        if sampler is not None and sampler not in SAMPLER_NAMES:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Unknown sampler \u201c" + sampler + "\u201d.", ephemeral=True
            )
            return
        if scheduler is not None and scheduler not in SCHEDULER_NAMES:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Unknown scheduler \u201c" + scheduler + "\u201d.", ephemeral=True
            )
            return

        if nsfw_blocked(interaction, prompt):
            await interaction.response.send_message(
                content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.",
                ephemeral=True,
            )
            msg = await interaction.original_response()
            schedule_message_deletion(msg)
            return

        if not await check_cooldown(interaction):
            await interaction.response.send_message(
                content="\u23f3 Please wait before retrying.", ephemeral=True
            )
            return

        image_attachments = [
            a for a in interaction.message.attachments
            if a.content_type and a.content_type.startswith("image/")
        ]
        if not image_attachments:
            await interaction.response.send_message(
                content="\u26a0\ufe0f This message has no image to edit.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=stealth)
        progress_msg = await interaction.followup.send(
            content=job_queue.waiting_prefix("comfyui") + "\U0001f3a8 Editing image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress = ProgressUpdater(progress_msg, lane="comfyui",
                                   label="\U0001f3a8 Editing image\u2026")
        log.info("Qwen Edit clicked for message %s", interaction.message.id)

        model = "qwen_image"
        spec = config["models"].get(model, {}).get("i2i")
        if spec is None:
            progress.done = True
            await reply_error(interaction, "\u274c The Qwen single-image edit workflow is not configured.", target=progress_msg)
            return

        # Fall back to the user's shared Qwen defaults for unset fields.
        saved = user_settings.get_settings(interaction.user.id)
        if cfg is None:
            cfg = saved.get("qwen_cfg")
        if steps is None:
            steps = saved.get("qwen_steps")
        if sampler is None:
            sampler = saved.get("qwen_sampler")
        if scheduler is None:
            scheduler = saved.get("qwen_scheduler")

        try:
            data1 = await download_image(image_attachments[0].url)
            uploaded1 = await comfy.upload_image(data1, f"discord_{uuid_hex()}.png")
        except Exception as exc:
            progress.done = True
            log.exception("Qwen Edit download/upload failed")
            await reply_error(interaction, f"\u274c Could not process the source image: {exc}", target=progress_msg)
            return

        gen_kwargs = {
            "prompt": prompt,
            "seed": None,
            "cfg": cfg,
            "steps": steps,
            "sampler": sampler,
            "scheduler": scheduler,
            "image_filename": uploaded1,
            # Single-image button flow: the second-image node is omitted.
            "image2_filename": None,
        }

        try:
            job = run_image(spec, on_progress=progress.update, model_key=model, **gen_kwargs)
            fut = job_queue.submit(job, lane="comfyui", name="qwen_image_i2i")
            progress.arm(job)
            images, meta = await fut
            progress.done = True
            display_model = meta.get("ckpt_name") or model
            base_lines = [
                f"**Model:** {display_model}",
                "**Workflow:** 1 image (edit)",
                f"**Prompt:** {prompt[:3800]}",
                f"**Resolution:** {image_resolution(images[0])}",
            ]
            base_desc = "\n".join(base_lines)
            await deliver_generation(
                interaction, images=images, meta=meta, base_desc=base_desc,
                color=int(discord.Color.purple()), spec=spec, model=model,
                suffix="i2i", stealth=stealth, target=progress_msg,
                save_kwargs={**gen_kwargs, "seed": None},
            )
        except Exception as exc:
            progress.done = True
            log.exception("Qwen Edit failed")
            await reply_error(interaction, f"\u274c Image edit failed: {exc}", target=progress_msg)


class FluxEditButton(Button):
    """Blue 'Flux Edit' button shown on every generated output.

    Opens a modal to run the single-image (1 image / edit) flux_edit workflow
    on the image(s) in the message. The custom_id is unchanged from the old
    'Edit Image' button so buttons on already-posted messages keep working.
    """

    def __init__(self):
        super().__init__(label="Flux Edit", emoji="\U0001f973", style=discord.ButtonStyle.primary, custom_id="edit_image_generation")

    async def callback(self, interaction: discord.Interaction):
        if await ban_guard(interaction):
            return
        log.info("Flux Edit clicked for message %s", interaction.message.id)
        params = generation_store.get(interaction.message.id)
        default_stealth = bool(params.get("stealth", False)) if params else False
        await interaction.response.send_modal(FluxEditModal(default_stealth=default_stealth))


class QwenEditButton(Button):
    """Orange 'Qwen Edit' button shown on every generated output.

    Opens a modal to run the single-image Qwen Image 2.1 edit workflow
    on the image(s) in the message.
    """

    def __init__(self):
        super().__init__(label="Qwen Edit", emoji="\u2728", style=discord.ButtonStyle.green, custom_id="qwen_edit_image_generation")

    async def callback(self, interaction: discord.Interaction):
        if await ban_guard(interaction):
            return
        log.info("Qwen Edit clicked for message %s", interaction.message.id)
        params = generation_store.get(interaction.message.id)
        default_stealth = bool(params.get("stealth", False)) if params else False
        await interaction.response.send_modal(QwenEditModal(default_stealth=default_stealth))


class GenerationView(View):
    """Persistent view: buttons work indefinitely and survive bot restarts.

    Because the view has timeout=None and stable custom_ids, it is registered
    globally via bot.add_view(). After a restart, clicks on old generation
    messages are dispatched to this registered view, which reads the
    generation params from the SQLite store by message id.
    """

    def __init__(self, stealth: bool = False):
        super().__init__(timeout=None)
        self.add_item(RetryButton())
        if not stealth:
            self.add_item(DeleteButton())
        self.add_item(UpscaleButton())
        self.add_item(FluxEditButton())
        self.add_item(QwenEditButton())

    async def on_error(self, interaction: discord.Interaction, error: Exception, item):
        log.warning("GenerationView error: %s", error)
class ThinkingView(View):
    """Select menu shown by /gen_prompt once the model is chosen.

    The select lists only the reasoning-effort values the endpoint accepted
    during probing (HTTP 200), plus "API default", which sends no
    reasoning_effort tag at all.

    The view does NOT run the generation itself. Instead it stores the chosen
    value and signals ``completion`` so the session job (which holds the LLM
    lane for the whole session) can proceed with generation, or skip it on
    timeout. This keeps the LLM lane occupied from model load through unload
    so no other LLM job can start while the model is loaded.
    """

    def __init__(self, chosen_model: str, supported: list[str], prompt: str,
                 megapixels: int, aspect_ratio: str,
                 max_tokens: int | None, temperature: float | None, llm_cfg: dict,
                 completion, state: dict):
        super().__init__(timeout=30)
        self.chosen_model = chosen_model
        self.prompt = prompt
        self.megapixels = megapixels
        self.aspect_ratio = aspect_ratio
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_cfg = llm_cfg
        self.completion = completion
        self.state = state
        options = [discord.SelectOption(label="API default", value="api_default")]
        options += [discord.SelectOption(label=e, value=e) for e in supported]
        self.add_item(ThinkingSelect(
            placeholder="Reasoning effort",
            options=options,
            min_values=1, max_values=1,
            custom_id="gen_prompt_thinking",
        ))

    async def handle_select(self, interaction: discord.Interaction, value: str):
        # Record the user's choice and release the session job, which performs
        # the actual generation and unload. Respond so Discord acknowledges
        # the click. The follow-up "Generating prompt\u2026" message is stored
        # in state so the session job can write the final result there instead
        # of the picker message.
        self.stop()
        self.state["selected_value"] = value
        self.state["followup_msg"] = None
        try:
            await interaction.response.defer(ephemeral=True)
            self.state["followup_msg"] = await interaction.followup.send(
                content="\U0001f9e0 Generating prompt\u2026", ephemeral=True
            )
        except Exception:
            self.state["followup_msg"] = None
        self.completion.set()

    async def on_timeout(self):
        # No reasoning effort was chosen; signal the session job so it can post
        # the "ran out of time" message on the picker and unload the model to
        # free the LLM lane. (Messaging is done by the session job because it
        # holds a reliable reference to the picker message.)
        self.stop()
        self.state["selected_value"] = None
        self.state["followup_msg"] = None
        self.completion.set()


class ThinkingSelect(Select):
    """The reasoning-effort select item shown by /gen_prompt.

    discord.py dispatches select-menu interactions to the component's own
    ``callback`` method (via ``Component.interaction``), NOT to a View-level
    ``interaction`` method — so the handler has to live on the select itself.
    """

    def __init__(self, **kwargs):
        # ``view`` is a read-only property set by ``View.add_item()``
        # (component.view = self), so by callback time ``self.view``
        # already points back to the ThinkingView.
        super().__init__(**kwargs)

    async def callback(self, interaction: discord.Interaction):
        await self.view.handle_select(interaction, self.values[0])
class AdminView(View):
    """Ephemeral admin panel: Restart, Promote/Demote, Ban/Unban.

    Owner-only buttons (Restart, Promote/Demote) are hidden for promoted
    admins; ban/view buttons are hidden for the owner unless they are also
    a promoted admin.
    """

    def __init__(self, user_id: int):
        super().__init__(timeout=None)
        owner = is_owner(user_id)
        is_admin = moderation.is_admin(user_id)
        if owner:
            self.add_item(RestartButton())
            self.add_item(ManageAdminsButton())
        if owner or is_admin:
            self.add_item(ManageBansButton())
            self.add_item(ViewBansButton())


class RestartButton(Button):
    def __init__(self):
        super().__init__(label="Restart Bot", emoji="\U0001f504", custom_id="admin_restart")

    async def callback(self, interaction: discord.Interaction):
        if not is_owner(interaction.user.id):
            await interaction.response.send_message(
                content="\u26a0\ufe0f Only the bot owner can restart the bot.", ephemeral=True
            )
            return
        await interaction.response.send_message(
            content="\U0001f504 Restarting the bot\u2026", ephemeral=True
        )
        log.info("Restart requested by %s", interaction.user.id)
        # Spawn a fresh process with the same argv, from the project root
        # (workflow JSON paths and the *.db files are resolved relative to it),
        # then close the bot gracefully so websockets/aiohttp shut down cleanly;
        # main.py's finally-block runs the remaining session cleanup.
        script = os.path.abspath(sys.argv[0])
        command = [sys.executable, script] + sys.argv[1:]
        subprocess.Popen(command, cwd=os.path.dirname(script))
        asyncio.create_task(bot.close())


class ManageAdminsButton(Button):
    def __init__(self):
        super().__init__(label="Promote / Demote Admins", emoji="\U0001f465", custom_id="admin_manage_admins")

    async def callback(self, interaction: discord.Interaction):
        if not is_owner(interaction.user.id):
            await interaction.response.send_message(
                content="\u26a0\ufe0f Only the bot owner can promote or demote admins.", ephemeral=True
            )
            return
        await interaction.response.send_message(
            content="Promote or demote an admin:", view=AdminActionView(), ephemeral=True
        )


class ManageBansButton(Button):
    def __init__(self):
        super().__init__(label="Ban / Unban Users", emoji="\U0001f6ab", custom_id="admin_manage_bans")

    async def callback(self, interaction: discord.Interaction):
        if not can_manage(interaction.user.id):
            await interaction.response.send_message(
                content="\u26a0\ufe0f Only the owner or admins can manage bans.", ephemeral=True
            )
            return
        await interaction.response.send_message(
            content="Ban or unban a user:", view=BanActionView(), ephemeral=True
        )


class ViewBansButton(Button):
    """List the users currently banned (and the current admins)."""

    def __init__(self):
        super().__init__(label="View User List", emoji="\U0001f4dc", custom_id="admin_view_bans")

    async def callback(self, interaction: discord.Interaction):
        if not can_manage(interaction.user.id):
            await interaction.response.send_message(
                content="\u26a0\ufe0f Only the owner or admins can view the user list.", ephemeral=True
            )
            return
        bans = moderation.get_bans()
        admins = moderation.get_admins()
        ban_lines = [f"\u2022 {await _user_label(b)}" for b in sorted(bans)]
        admin_labels = [
            (await _user_label(a)) + (" (owner)" if a == BOT_OWNER_ID else "")
            for a in sorted(admins)
        ]
        if not ban_lines:
            text = "\U0001f6ab **Ban list:** (empty)\n\n\U0001f465 **Admins:** " + ", ".join(admin_labels)
        else:
            text = (
                "\U0001f6ab **Ban list:**\n"
                + "\n".join(ban_lines)
                + "\n\n\U0001f465 **Admins:** " + ", ".join(admin_labels)
            )
        await interaction.response.send_message(content=text, ephemeral=True)


async def _user_label(discord_id: int) -> str:
    """Return a ``name (id)`` label, falling back to the raw id if the
    user can't be fetched (e.g. deleted account)."""
    try:
        user = await bot.fetch_user(discord_id)
        name = user.name
        tag = user.discriminator
        if tag and tag != "0":
            name = f"{name}#{tag}"
        return f"{name} ({discord_id})"
    except Exception:
        return str(discord_id)


class AdminActionView(View):
    """Buttons to pick promote or demote, then open the ID modal."""

    def __init__(self):
        super().__init__()
        self.add_item(AdminPromoteButton())
        self.add_item(AdminDemoteButton())


class AdminPromoteButton(Button):
    def __init__(self):
        super().__init__(label="Promote to admin", custom_id="admin_promote")

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(AdminsModal("promote"))


class AdminDemoteButton(Button):
    def __init__(self):
        super().__init__(label="Demote from admin", custom_id="admin_demote")

    async def callback(self, interaction: discord.Interaction):
        # The owner can never be demoted, so exclude them from the list.
        admins = [a for a in moderation.get_admins() if a != BOT_OWNER_ID]
        if not admins:
            await interaction.response.send_message(
                content="\U0001f465 There are no demotable admins.", ephemeral=True
            )
            return
        shown = sorted(admins)[:25]  # Discord caps select options at 25
        options = [(await _user_label(a), str(a)) for a in shown]
        listing = "\n".join(f"\u2022 {label}" for label, _ in options)
        content = "Current admins (demotable):\n" + listing
        if len(admins) > 25:
            content += "\n\n\u26a0\ufe0f Only the first 25 are listed; use the manual ID button for the rest."
        content += "\n\nPick an admin above, or enter an ID manually."
        await interaction.response.send_message(
            content=content,
            view=AdminDemoteView(options),
            ephemeral=True,
        )


class AdminDemoteView(View):
    """Dropdown of admins (click to demote) plus a manual ID button."""

    def __init__(self, options: list[tuple[str, str]]):
        super().__init__()
        self.add_item(DemoteSelect(options))
        self.add_item(AdminDemoteManualButton())


class DemoteSelect(Select):
    """Dropdown of admins; selecting one demotes them."""

    def __init__(self, options: list[tuple[str, str]]):
        super().__init__(placeholder="Select an admin to demote")
        for label, value in options:
            self.add_option(label=label, value=value)

    async def callback(self, interaction: discord.Interaction):
        target = int(self.values[0])
        if target == BOT_OWNER_ID:
            await interaction.response.edit_message(
                content="\u26a0\ufe0f The owner cannot be demoted.", view=None
            )
            return
        if not moderation.is_admin(target):
            await interaction.response.edit_message(
                content=f"\u26a0\ufe0f User {target} is not an admin.", view=None
            )
            return
        moderation.demote(target)
        await interaction.response.edit_message(
            content=f"\u2705 Demoted user {target} from admin.", view=None
        )


class AdminDemoteManualButton(Button):
    """Opens the manual user-ID modal for demotion."""

    def __init__(self):
        super().__init__(label="Enter user ID manually", custom_id="admin_demote_manual")

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(AdminsModal("demote"))


class AdminsModal(Modal):
    """Promote or demote a user via their Discord ID (TextInput only)."""

    def __init__(self, action: str):
        super().__init__(title="Promote / Demote Admin")
        self.action = action
        self.id_input = TextInput(
            label="Discord user ID", style=TextStyle.short,
            placeholder="e.g. 123456789012345678",
        )
        self.add_item(self.id_input)

    async def on_submit(self, interaction: discord.Interaction):
        raw = (self.id_input.value or "").strip()
        try:
            target = int(raw)
        except ValueError:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Please enter a numeric Discord user ID.", ephemeral=True
            )
            return
        action = self.action
        if action == "promote":
            if target == BOT_OWNER_ID:
                await interaction.response.send_message(
                    content="\u26a0\ufe0f The owner is already an admin.", ephemeral=True
                )
                return
            moderation.promote(target)
            await interaction.response.send_message(
                content=f"\u2705 Promoted user {target} to admin.", ephemeral=True
            )
        else:
            if target == BOT_OWNER_ID:
                await interaction.response.send_message(
                    content="\u26a0\ufe0f The owner cannot be demoted.", ephemeral=True
                )
                return
            if not moderation.is_admin(target):
                await interaction.response.send_message(
                    content=f"\u26a0\ufe0f User {target} is not an admin.", ephemeral=True
                )
                return
            if target == interaction.user.id:
                await interaction.response.send_message(
                    content="\u26a0\ufe0f You cannot demote yourself.", ephemeral=True
                )
                return
            moderation.demote(target)
            await interaction.response.send_message(
                content=f"\u2705 Demoted user {target} from admin.", ephemeral=True
            )


class BanActionView(View):
    """Buttons to pick ban or unban, then open the ID modal."""

    def __init__(self):
        super().__init__()
        self.add_item(BanActionButton("ban", "Ban user", "\U0001f6ab", "ban_action_ban"))
        self.add_item(BanActionButton("unban", "Unban user", "\u2705", "ban_action_unban"))


class BanActionButton(Button):
    def __init__(self, action: str, label: str, emoji: str, custom_id: str):
        super().__init__(label=label, emoji=emoji, custom_id=custom_id)
        self.action = action

    async def callback(self, interaction: discord.Interaction):
        if self.action == "ban":
            await interaction.response.send_modal(BansModal("ban"))
            return
        # Unban: show the ban list as a clickable dropdown plus a manual
        # ID entry button.
        bans = moderation.get_bans()
        if not bans:
            await interaction.response.send_message(
                content="\U0001f6ab No users are currently banned.", ephemeral=True
            )
            return
        shown = sorted(bans)[:25]  # Discord caps select options at 25
        options = [(await _user_label(b), str(b)) for b in shown]
        listing = "\n".join(f"\u2022 {label}" for label, _ in options)
        content = "Currently banned:\n" + listing
        if len(bans) > 25:
            content += "\n\n\u26a0\ufe0f Only the first 25 are listed; use the manual ID button for the rest."
        content += "\n\nPick a user above, or enter an ID manually."
        await interaction.response.send_message(
            content=content,
            view=UnbanSelectView(options),
            ephemeral=True,
        )


class UnbanSelect(Select):
    """Dropdown of banned users; selecting one unbans them."""

    def __init__(self, options: list[tuple[str, str]]):
        super().__init__(placeholder="Select a user to unban")
        for label, value in options:
            self.add_option(label=label, value=value)

    async def callback(self, interaction: discord.Interaction):
        target = int(self.values[0])
        moderation.unban(target)
        await interaction.response.edit_message(
            content=f"\u2705 Unbanned user {target}.", view=None
        )


class UnbanSelectView(View):
    """Dropdown of banned users (click to unban) plus a manual ID button."""

    def __init__(self, options: list[tuple[str, str]]):
        super().__init__()
        self.add_item(UnbanSelect(options))
        self.add_item(UnbanManualButton())


class UnbanManualButton(Button):
    """Opens the manual user-ID modal for unbanning."""

    def __init__(self):
        super().__init__(label="Enter user ID manually", custom_id="unban_manual")

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(BansModal("unban"))


class BansModal(Modal):
    """Ban or unban a user via their Discord ID (TextInput only)."""

    def __init__(self, action: str):
        super().__init__(title="Ban / Unban User")
        self.action = action
        self.id_input = TextInput(
            label="Discord user ID", style=TextStyle.short,
            placeholder="e.g. 123456789012345678",
        )
        self.add_item(self.id_input)

    async def on_submit(self, interaction: discord.Interaction):
        raw = (self.id_input.value or "").strip()
        try:
            target = int(raw)
        except ValueError:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Please enter a numeric Discord user ID.", ephemeral=True
            )
            return
        action = self.action
        if action == "ban":
            if target == BOT_OWNER_ID or moderation.is_admin(target):
                await interaction.response.send_message(
                    content="\u26a0\ufe0f The owner and admins cannot be banned.", ephemeral=True
                )
                return
            moderation.ban(target)
            await interaction.response.send_message(
                content=f"\U0001f6ab Banned user {target} from using the bot.", ephemeral=True
            )
        else:
            if not moderation.is_banned(target):
                await interaction.response.send_message(
                    content=f"\u26a0\ufe0f User {target} is not banned.", ephemeral=True
                )
                return
            moderation.unban(target)
            await interaction.response.send_message(
                content=f"\u2705 Unbanned user {target}.", ephemeral=True
            )
