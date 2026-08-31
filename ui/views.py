import io
import os
import subprocess
import sys

from http_session import get_session
from PIL import Image
import discord
from discord.ui import View, Button, Modal, TextInput, Select
from discord.enums import TextStyle

from bot import bot
from core import (
    config, comfy, log, nsfw_guard, generation_store, user_settings, moderation,
    BOT_OWNER_ID, SAMPLER_NAMES, UPSCALE_MODELS, UPSCALE_MODEL_LABELS,
    compress_image, image_resolution, meta_lines, uuid_hex,
    reply_error, ban_guard, check_cooldown, can_manage, is_owner,
    schedule_message_deletion, schedule_original_response_deletion,
    ProgressUpdater, progress_bar, normalize_aspect_ratio,
    _parse_opt_float, _parse_opt_int, _parse_opt_str,
)
from workflow import run_image, run_text_workflow
from llm_client import call_llm, resolve_reasoning_effort, llm_model_unload


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
        stealth = bool(params.get("stealth", False))
        await interaction.response.send_message(
            content="\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress_msg = await interaction.original_response()
        progress = ProgressUpdater(progress_msg)
        try:
            images, meta = await run_image(
                params["spec"], on_progress=progress.update,
                model_key=params["model"], **params["kwargs"]
            )
            progress.done = True
            for img in images:
                if await nsfw_guard.check_image_nsfw(img, interaction):
                    await interaction.edit_original_response(
                        content="\u26a0\ufe0f Image blocked: NSFW content is only allowed in NSFW channels."
                    )
                    schedule_original_response_deletion(interaction)
                    return
            files = []
            for i, img in enumerate(images):
                img_bytes, ext = compress_image(img)
                files.append(discord.File(io.BytesIO(img_bytes), filename=f"{params['model']}_{params['suffix']}_{i}{ext}"))
            embed = discord.Embed(
                description=params["embed_desc"] + "\n" + "\n".join(meta_lines(meta)),
                color=discord.Color(params["embed_color"]),
            )
            # Edit the progress message in place so the output replaces it.
            response_msg = await interaction.edit_original_response(
                content="", embed=embed, attachments=files, view=GenerationView(stealth=stealth)
            )
            # Persist the params under the message id so it can itself be retried/deleted.
            # Tag with the generating user so only that user can delete the output.
            params["user_id"] = interaction.user.id
            generation_store.save(response_msg.id, params)
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
        generation_store.pop(interaction.message.id)
        try:
            await interaction.message.delete()
        except discord.Forbidden:
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
            content="\U0001f3a8 Upscaling image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress = ProgressUpdater(progress_msg)
        spec = config["models"].get(model, {}).get("upscale")
        if spec is None:
            progress.done = True
            await reply_error(interaction, f"\u274c Model {model!r} has no upscaling workflow.", target=progress_msg)
            return
        settings = user_settings.get_settings(interaction.user.id)
        negative = settings["negative_prompt"] or None
        # For SDXL, reuse the checkpoint the source image was created with
        # so the upscale matches the original model.
        ckpt_name = view.ckpt_name if model == "sdxl" else None
        try:
            images, meta = await run_image(
                spec, on_progress=progress.update, model_key=model, prompt=None,
                negative=negative, strength=None,
                image_filename=view.uploaded_name, scale=2,
                input_longest_side=view.input_longest_side,
                ckpt_name=ckpt_name,
            )
            progress.done = True
            for img in images:
                if await nsfw_guard.check_image_nsfw(img, interaction):
                    msg = await progress_msg.edit(
                        content="\u26a0\ufe0f Image blocked: NSFW content is only allowed in NSFW channels."
                    )
                    schedule_message_deletion(msg)
                    return
            files = []
            for i, img in enumerate(images):
                img_bytes, ext = compress_image(img)
                files.append(discord.File(io.BytesIO(img_bytes), filename=f"{model}_upscale_{i}{ext}"))
            display_ckpt = meta.get("ckpt_name") or ckpt_name
            base_desc = f"**Model:** {model}\n**Scale:** 2x"
            if model == "sdxl" and display_ckpt:
                base_desc += f"\n**Checkpoint:** {display_ckpt}"
            base_desc += f"\n**Resolution:** {image_resolution(images[0])}"
            embed = discord.Embed(
                description=base_desc + "\n" + "\n".join(meta_lines(meta)),
                color=discord.Color.green(),
            )
            # Edit the progress (follow-up) message, NOT the ephemeral
            # model-picker message, so the output is visible to everyone.
            response_msg = await progress_msg.edit(
                content="", embed=embed, attachments=files, view=GenerationView(stealth=stealth)
            )
            generation_store.save(response_msg.id, {
                "spec": spec, "model": model, "suffix": "upscale", "stealth": stealth,
                "embed_desc": base_desc, "embed_color": int(embed.color),
                "user_id": interaction.user.id,
                # Retries reuse the uploaded input image and roll a fresh seed.
                "kwargs": {"prompt": None, "negative": negative, "strength": None,
                            "image_filename": view.uploaded_name, "scale": 2,
                            "input_longest_side": view.input_longest_side,
                            "ckpt_name": ckpt_name},
            })
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
                 source_model: str | None = None, ckpt_name: str | None = None):
        super().__init__(timeout=300)
        self.uploaded_name = uploaded_name
        self.input_longest_side = input_longest_side
        self.stealth = stealth
        self.ckpt_name = ckpt_name
        # SDXL upscale only works well with SDXL checkpoints, so hide the SDXL
        # option when the source image was not generated with SDXL.
        for model in UPSCALE_MODELS:
            if model == "sdxl" and source_model != "sdxl":
                continue
            label = UPSCALE_MODEL_LABELS.get(model, model)
            self.add_item(UpscaleModelButton(model, label))

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
                 checkpoints: list[str]):
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
        # "default" maps to the workflow's own checkpoint (with automatic
        # fallback to an available SDXL checkpoint in run_image).
        options = [discord.SelectOption(label="Default (workflow checkpoint)", value="default")]
        options += [discord.SelectOption(label=c, value=c) for c in checkpoints[:100]]
        self.add_item(CheckpointSelect(
            placeholder="Select a checkpoint",
            options=options,
            min_values=1, max_values=1,
            custom_id="upscale_checkpoint",
        ))

    async def handle_select(self, interaction: discord.Interaction, value: str):
        if await ban_guard(interaction):
            return
        self.stop()
        ckpt_name = None if value == "default" else value
        log.info("Upscale checkpoint %s selected", ckpt_name or "default")
        # The /upscale command already consumed the cooldown when invoked;
        # the select click is the continuation of that same request, so we do
        # not re-check (re-checking would always fail and block the run).
        await interaction.response.defer(ephemeral=self.stealth)
        progress_msg = await interaction.followup.send(
            content="\U0001f3a8 Upscaling image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=self.stealth,
        )
        progress = ProgressUpdater(progress_msg)
        try:
            images, meta = await run_image(
                self.spec, on_progress=progress.update, model_key=self.model_key,
                prompt=self.prompt, negative=self.negative, strength=self.strength,
                image_filename=self.uploaded_name, scale=self.scale,
                input_longest_side=self.input_longest_side,
                ckpt_name=ckpt_name,
            )
            progress.done = True
            for img in images:
                if await nsfw_guard.check_image_nsfw(img, interaction):
                    msg = await progress_msg.edit(
                        content="\u26a0\ufe0f Image blocked: NSFW content is only allowed in NSFW channels."
                    )
                    schedule_message_deletion(msg)
                    return
            files = []
            for i, img in enumerate(images):
                img_bytes, ext = compress_image(img)
                files.append(discord.File(io.BytesIO(img_bytes), filename=f"{self.model_key}_upscale_{i}{ext}"))
            display_ckpt = meta.get("ckpt_name") or ckpt_name
            base_desc = (
                f"**Model:** {self.model_key}"
                + (f"\n**Checkpoint:** {display_ckpt}" if display_ckpt else "")
                + (f"\n**Scale:** {self.scale:g}x" if self.scale is not None else "")
                + f"\n**Resolution:** {image_resolution(images[0])}"
            )
            embed = discord.Embed(
                description=base_desc + "\n" + "\n".join(meta_lines(meta)),
                color=discord.Color.green(),
            )
            response_msg = await progress_msg.edit(
                content="", embed=embed, attachments=files, view=GenerationView(stealth=self.stealth)
            )
            generation_store.save(response_msg.id, {
                "spec": self.spec, "model": self.model_key, "suffix": "upscale", "stealth": self.stealth,
                "embed_desc": base_desc, "embed_color": int(embed.color),
                "user_id": interaction.user.id,
                "kwargs": {"prompt": self.prompt, "negative": self.negative, "strength": self.strength,
                            "image_filename": self.uploaded_name, "scale": self.scale,
                            "input_longest_side": self.input_longest_side, "ckpt_name": ckpt_name},
            })
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
        ckpt_name = (params.get("kwargs", {}).get("ckpt_name") if params else None)
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
            session = get_session()
            async with session.get(source.url) as resp:
                resp.raise_for_status()
                data = await resp.read()
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
            ),
        )


class EditImageModal(Modal):
    """Modal form for editing a generated image via the single-image
    (1 image / edit) img2img workflow using Flux 2 Klein 4B Base.

    The first image of the clicked message is used as the source. The form
    takes prompt, privacy (stealth/public), cfg, steps, and sampler;
    left-empty fields fall back to the user's saved img2img defaults.
    """

    def __init__(self, default_stealth: bool = False):
        super().__init__(title="Edit Image (1-image workflow)")
        self.default_stealth = default_stealth
        self.prompt_input = TextInput(
            label="Prompt", style=TextStyle.paragraph,
            placeholder="Describe the edit", required=True,
        )
        self.privacy_input = TextInput(
            label="Privacy (optional)", style=TextStyle.short, required=False,
            placeholder="stealth = only you see it; public = visible to everyone",
        )
        self.cfg_input = TextInput(
            label="CFG (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. 7",
        )
        self.steps_input = TextInput(
            label="Steps (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. 30",
        )
        self.sampler_input = TextInput(
            label="Sampler (optional)", style=TextStyle.short, required=False,
            placeholder="e.g. euler",
        )
        self.add_item(self.prompt_input)
        self.add_item(self.privacy_input)
        self.add_item(self.cfg_input)
        self.add_item(self.steps_input)
        self.add_item(self.sampler_input)

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
        cfg = _parse_opt_float(self.cfg_input.value)
        steps = _parse_opt_int(self.steps_input.value)
        sampler = _parse_opt_str(self.sampler_input.value)
        # Megapixels is no longer a modal field; fall back to the saved img2img default.
        megapixels = None

        if sampler is not None and sampler not in SAMPLER_NAMES:
            await interaction.response.send_message(
                content="\u26a0\ufe0f Unknown sampler \u201c" + sampler + "\u201d.", ephemeral=True
            )
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
            content="\U0001f3a8 Editing image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress = ProgressUpdater(progress_msg)
        log.info("Edit Image clicked for message %s", interaction.message.id)

        model = "flux2_klein"
        spec = config["models"].get(model, {}).get("i2i_single")
        if spec is None:
            progress.done = True
            await reply_error(interaction, "\u274c The single-image edit workflow is not configured.", target=progress_msg)
            return

        # Fall back to the user's saved img2img defaults for unset fields.
        saved = user_settings.get_settings(interaction.user.id)
        if cfg is None:
            cfg = saved.get("img2img_cfg")
        if steps is None:
            steps = saved.get("img2img_steps")
        if sampler is None:
            sampler = saved.get("img2img_sampler")
        if megapixels is None:
            megapixels = saved.get("img2img_megapixels")

        try:
            session = get_session()
            async with session.get(image_attachments[0].url) as resp:
                resp.raise_for_status()
                data1 = await resp.read()
            uploaded1 = await comfy.upload_image(data1, f"discord_{uuid_hex()}.png")
        except Exception as exc:
            progress.done = True
            log.exception("Edit Image download/upload failed")
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
            images, meta = await run_image(spec, on_progress=progress.update, model_key=model, **gen_kwargs)
            progress.done = True
            for img in images:
                if await nsfw_guard.check_image_nsfw(img, interaction):
                    msg = await progress_msg.edit(
                        content="\u26a0\ufe0f Image blocked: NSFW content is only allowed in NSFW channels."
                    )
                    schedule_message_deletion(msg)
                    return
            files = []
            for i, img in enumerate(images):
                img_bytes, ext = compress_image(img)
                files.append(discord.File(io.BytesIO(img_bytes), filename=f"flux2_klein_i2i_{i}{ext}"))
            base_lines = [
                f"**Model:** {model}",
                f"**Workflow:** 1 image (edit)",
                f"**Prompt:** {prompt}",
                f"**Resolution:** {image_resolution(images[0])}",
            ]
            base_desc = "\n".join(base_lines)
            embed = discord.Embed(
                description=base_desc + "\n" + "\n".join(meta_lines(meta)),
                color=discord.Color.green(),
            )
            response_msg = await progress_msg.edit(
                content="", embed=embed, attachments=files, view=GenerationView(stealth=stealth)
            )
            generation_store.save(response_msg.id, {
                "spec": spec, "model": model, "suffix": "i2i", "stealth": stealth,
                "embed_desc": base_desc, "embed_color": int(embed.color),
                "user_id": interaction.user.id,
                "kwargs": {**gen_kwargs, "seed": None},
            })
        except Exception as exc:
            progress.done = True
            log.exception("Edit Image failed")
            await reply_error(interaction, f"\u274c Image edit failed: {exc}", target=progress_msg)
class EditButton(Button):
    """Green 'Edit Image' button shown on every generated output.

    Opens a modal to run the single-image (1 image / edit) img2img workflow
    on the image(s) in the message.
    """

    def __init__(self):
        super().__init__(label="Edit Image", emoji="\U0001f973", style=discord.ButtonStyle.green, custom_id="edit_image_generation")

    async def callback(self, interaction: discord.Interaction):
        if await ban_guard(interaction):
            return
        log.info("Edit Image clicked for message %s", interaction.message.id)
        params = generation_store.get(interaction.message.id)
        default_stealth = bool(params.get("stealth", False)) if params else False
        await interaction.response.send_modal(EditImageModal(default_stealth=default_stealth))


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
        self.add_item(EditButton())

    async def on_error(self, interaction: discord.Interaction, error: Exception, item):
        log.warning("GenerationView error: %s", error)
class ThinkingView(View):
    """Select menu shown by /gen_prompt once the model is chosen.

    The select lists only the reasoning-effort values the endpoint accepted
    during probing (HTTP 200), plus "API default", which sends no
    reasoning_effort tag at all. Selecting an option runs the prompt
    workflow and the LLM call.
    """

    def __init__(self, chosen_model: str, supported: list[str], prompt: str,
                 megapixels: int, aspect_ratio: str,
                 max_tokens: int | None, temperature: float | None, llm_cfg: dict):
        super().__init__(timeout=120)
        self.chosen_model = chosen_model
        self.prompt = prompt
        self.megapixels = megapixels
        self.aspect_ratio = aspect_ratio
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_cfg = llm_cfg
        options = [discord.SelectOption(label="API default", value="api_default")]
        options += [discord.SelectOption(label=e, value=e) for e in supported]
        self.add_item(ThinkingSelect(
            placeholder="Reasoning effort",
            options=options,
            min_values=1, max_values=1,
            custom_id="gen_prompt_thinking",
        ))

    async def handle_select(self, interaction: discord.Interaction, value: str):
        self.stop()
        await interaction.response.defer(ephemeral=True)
        msg = await interaction.original_response()
        try:
            effort = await resolve_reasoning_effort(self.chosen_model, value, self.llm_cfg)
            # The model was already loaded when the select menu was shown;
            # don't load it again (each load needs a matching unload).
            # v2 workflow composes the full prompt (system + aspect ratio +
            # user idea) via string nodes and outputs it through PreviewAny
            # node 111. The bot sends that composed prompt to the LLM itself,
            # so the timeout is controlled by llm.timeout in config.yaml
            # (300s) instead of the old hardcoded 30s.
            patches = {
                "191": {
                    "aspect_ratio": normalize_aspect_ratio(self.aspect_ratio),
                    "megapixels": int(self.megapixels),
                },
                "134:115": {
                    "value": self.prompt,
                },
            }
            composed_prompt = await run_text_workflow(
                "workflows/ideogram_prompt_gen/ideogram4_prompt_gen.json",
                patches,
                target_node="111",
            )
            result = await call_llm(
                composed_prompt,
                self.chosen_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=effort,
            )
            await msg.edit(content=result)
        except Exception as exc:
            log.exception("gen_prompt failed")
            await reply_error(interaction, f"\u274c Prompt generation failed: {exc}", target=msg)
        finally:
            # Free memory once the prompt is done (ignored on servers without
            # the unload endpoint).
            await llm_model_unload(self.chosen_model)

    async def on_timeout(self):
        # The model was loaded when this view was shown; free it since no
        # prompt will be generated.
        await llm_model_unload(self.chosen_model)
        try:
            await self.message.edit(content="\u23f3 Reasoning-effort selection expired; prompt generation cancelled.")
        except Exception:
            pass


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
        # Spawn a fresh process with the same argv, then exit immediately.
        command = [sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]
        subprocess.Popen(command, cwd=os.path.dirname(os.path.abspath(__file__)))
        os._exit(0)


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
