import io

from PIL import Image
import discord
import logging
from discord import app_commands

from bot import bot
from http_session import get_session
from core import (
    config, comfy, log, user_settings, generation_store, nsfw_guard, moderation,
    compress_image, meta_lines, image_resolution, uuid_hex,
    reply_error, ban_guard, check_cooldown, nsfw_blocked,
    schedule_message_deletion, schedule_original_response_deletion,
    ProgressUpdater, progress_bar, normalize_aspect_ratio,
    UPSCALE_CHOICES, ASPECT_RATIO_CHOICES, QUALITY_CHOICES,
    SAMPLER_CHOICES, SCHEDULER_CHOICES, I2I_WORKFLOW_CHOICES,
    ComfyUIError,
)
from ui.autocomplete import _sdxl_model_autocomplete, _sdxl_lora_autocomplete, _zimage_model_autocomplete
from workflow import run_image
from ui.views import GenerationView, CheckpointPickerView
from job_queue import job_queue


async def run_t2i_generation(interaction: discord.Interaction, model: str,
                             prompt: str, stealth: bool, gen_kwargs: dict):
    """Shared generation pipeline for the /ideogram and /sdxl commands.

    ``gen_kwargs`` carries the model-specific parameters (plus prompt/seed).
    The embed description and saved retry parameters are built from
    ``gen_kwargs`` so each command exposes only its own settings.
    """
    spec = config["models"][model]["t2i"]
    wait_prefix = job_queue.waiting_prefix("comfyui")
    await interaction.response.send_message(
        content=wait_prefix + "\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
        ephemeral=stealth,
    )
    progress_msg = await interaction.original_response()
    progress = ProgressUpdater(progress_msg)
    log.info("%s: prompt=%r", model, prompt)
    try:
        images, meta = await job_queue.submit(
            run_image(
                spec, on_progress=progress.update,
                model_key=model, **gen_kwargs,
            ),
            lane="comfyui",
            name=f"{model}_t2i",
        )
        progress.done = True
        log.info("%s: got %d images", model, len(images))
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
            files.append(discord.File(io.BytesIO(img_bytes), filename=f"{model}_t2i_{i}{ext}"))
        display_model = meta.get("ckpt_name") or gen_kwargs.get("ckpt_name") or model
        base_lines = [f"**Model:** {display_model}", f"**Prompt:** {prompt}"]
        if model == "ideogram":
            quality = gen_kwargs.get("quality")
            if quality:
                base_lines.append(f"**Quality:** {quality}")
            res_parts = []
            megapixels = gen_kwargs.get("megapixels")
            if megapixels:
                res_parts.append(f"{megapixels} MP")
            aspect_ratio = gen_kwargs.get("aspect_ratio")
            if aspect_ratio:
                res_parts.append(aspect_ratio)
            if res_parts:
                base_lines.append("**Resolution:** " + ", ".join(res_parts))
            base_lines.append(f"**Output:** {image_resolution(images[0])}")
        else:
            base_lines.append(f"**Resolution:** {image_resolution(images[0])}")
        base_desc = "\n".join(base_lines)
        embed = discord.Embed(
            description=base_desc + "\n" + "\n".join(meta_lines(meta)),
            color=discord.Color.blue(),
        )
        response_msg = await interaction.edit_original_response(content="", embed=embed, attachments=files, view=GenerationView(stealth=stealth))
        generation_store.save(response_msg.id, {
            "spec": spec, "model": model, "suffix": "t2i", "stealth": stealth,
            "embed_desc": base_desc, "embed_color": int(embed.color),
            "user_id": interaction.user.id,
            # seed=None so Retry rolls a fresh seed and produces a new image.
            # sampler/scheduler recorded so an SDXL upscale can reuse them.
            "kwargs": {**gen_kwargs, "seed": None,
                        "sampler": meta.get("sampler"),
                        "scheduler": meta.get("scheduler")},
        })
    except (ComfyUIError, Exception) as exc:
        progress.done = True
        log.exception("generation failed")
        await reply_error(interaction, f"\u274c Generation failed: {exc}")


@bot.tree.command(name="ideogram", description="Generate an image with Ideogram 4")
@app_commands.choices(quality=QUALITY_CHOICES)
@app_commands.choices(aspect_ratio=ASPECT_RATIO_CHOICES)
@app_commands.describe(prompt="Text prompt")
@app_commands.describe(seed="Seed (optional)")
@app_commands.describe(quality="Quality preset: Turbo / Default / Quality")
@app_commands.describe(megapixels="Target resolution in megapixels")
@app_commands.describe(aspect_ratio="Aspect ratio preset")
@app_commands.describe(stealth="Ephemeral output, visible only to you")
async def ideogram(interaction: discord.Interaction, prompt: str,
                   seed: int | None = None,
                   quality: str | None = None,
                   megapixels: int | None = None,
                   aspect_ratio: str | None = None,
                   stealth: bool | None = None):
    if stealth is None:
        stealth = bool(user_settings.get_settings(interaction.user.id).get("stealth", False))
    if moderation.is_banned(interaction.user.id):
        await interaction.response.send_message(
            content="\U0001f6ab You are banned from using this bot. Please contact an admin.", ephemeral=True
        )
        return
    if not await check_cooldown(interaction):
        await interaction.response.send_message(
            content="\u23f3 Please wait before requesting another image.", ephemeral=True
        )
        return

    if nsfw_blocked(interaction, prompt):
        await interaction.response.send_message(
            content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    settings = user_settings.get_settings(interaction.user.id)
    if quality is None:
        quality = settings.get("ideogram_quality")
    if megapixels is None:
        megapixels = settings.get("ideogram_megapixels")
    if aspect_ratio is None:
        aspect_ratio = settings.get("ideogram_aspect_ratio")
    aspect_ratio = normalize_aspect_ratio(aspect_ratio)
    gen_kwargs = {
        "prompt": prompt,
        "seed": seed,
        "quality": quality,
        "megapixels": megapixels,
        "aspect_ratio": aspect_ratio,
    }
    await run_t2i_generation(interaction, "ideogram", prompt, stealth, gen_kwargs)


@bot.tree.command(name="sdxl", description="Generate an image with SDXL or any checkpoint in models/checkpoints")
@app_commands.autocomplete(model=_sdxl_model_autocomplete)
@app_commands.describe(prompt="Text prompt")
@app_commands.describe(model="Checkpoint in models/checkpoints (optional)")
@app_commands.describe(negative="Negative prompt")
@app_commands.describe(steps="Sampling steps")
@app_commands.describe(width="Width in pixels, multiple of 64")
@app_commands.describe(height="Height in pixels, multiple of 64")
@app_commands.describe(sampler="Sampler (optional)")
@app_commands.describe(scheduler="Scheduler (optional)")
@app_commands.describe(cfg="CFG guidance scale")
@app_commands.describe(lora1="First LoRA (optional)")
@app_commands.describe(lora2="Second LoRA (optional)")
@app_commands.describe(seed="Seed (optional)")
@app_commands.describe(lora_strength="LoRA strength (optional, default 1.0)")
@app_commands.describe(stealth="Ephemeral output, visible only to you")
@app_commands.choices(sampler=SAMPLER_CHOICES)
@app_commands.choices(scheduler=SCHEDULER_CHOICES)
@app_commands.autocomplete(lora1=_sdxl_lora_autocomplete)
@app_commands.autocomplete(lora2=_sdxl_lora_autocomplete)
async def sdxl(interaction: discord.Interaction, prompt: str,
                model: str | None = None,
                negative: str | None = None,
                steps: int | None = None,
                width: int | None = None,
                height: int | None = None,
                sampler: str | None = None,
                scheduler: str | None = None,
                cfg: float | None = None,
                lora1: str | None = None,
                lora2: str | None = None,
                seed: int | None = None,
                lora_strength: float | None = None,
                stealth: bool | None = None):
    if stealth is None:
        stealth = bool(user_settings.get_settings(interaction.user.id).get("stealth", False))
    if moderation.is_banned(interaction.user.id):
        await interaction.response.send_message(
            content="\U0001f6ab You are banned from using this bot. Please contact an admin.", ephemeral=True
        )
        return
    if not await check_cooldown(interaction):
        await interaction.response.send_message(
            content="\u23f3 Please wait before requesting another image.", ephemeral=True
        )
        return

    if nsfw_blocked(interaction, prompt):
        await interaction.response.send_message(
            content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    settings = user_settings.get_settings(interaction.user.id)
    if negative is None:
        negative = settings["negative_prompt"] or None
    if steps is None:
        steps = settings.get("sdxl_steps")
    if cfg is None:
        cfg = settings.get("sdxl_cfg")
    if width is None:
        width = settings.get("width")
    if height is None:
        height = settings.get("height")
    if sampler is None:
        sampler = settings.get("sdxl_sampler")
    if scheduler is None:
        scheduler = settings.get("sdxl_scheduler")

    # Per-user default SDXL checkpoint (set via /settings). Used only by
    # /sdxl when no explicit model is passed; silently ignored if unavailable.
    if model is None:
        saved = settings.get("sdxl_checkpoint")
        if saved:
            try:
                available = await comfy.fetch_checkpoints()
            except Exception as exc:
                log.warning("Could not list checkpoints: %s", exc)
                available = None
            if available is not None:
                if saved in available:
                    model = saved
                else:
                    log.info("Saved SDXL checkpoint %r not available; using workflow default.", saved)

    # Optional checkpoint selection: any file in ComfyUI's models/checkpoints
    # folder can be used instead of the workflow's default checkpoint.
    if model is not None:
        try:
            available = await comfy.fetch_checkpoints()
        except Exception as exc:
            log.warning("Could not list checkpoints: %s", exc)
            await interaction.response.send_message(
                content=f"\u274c Could not query ComfyUI for available checkpoints: {exc}",
                ephemeral=True,
            )
            return
        if model not in available:
            await interaction.response.send_message(
                content=f"\u274c Checkpoint \u201c{model}\u201d was not found in models/checkpoints.",
                ephemeral=True,
            )
            return

    gen_kwargs = {
        "prompt": prompt,
        "negative": negative,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "sampler": sampler,
        "scheduler": scheduler,
        "cfg": cfg,
        "ckpt_name": model,
        "lora1": lora1,
        "lora2": lora2,
        "lora_strength": lora_strength,
    }
    await run_t2i_generation(interaction, "sdxl", prompt, stealth, gen_kwargs)


@bot.tree.command(name="zimage", description="Generate an image with Z-Image")
@app_commands.autocomplete(model=_zimage_model_autocomplete)
@app_commands.choices(sampler=SAMPLER_CHOICES)
@app_commands.choices(scheduler=SCHEDULER_CHOICES)
@app_commands.autocomplete(lora1=_sdxl_lora_autocomplete)
@app_commands.autocomplete(lora2=_sdxl_lora_autocomplete)
@app_commands.describe(prompt="Text prompt")
@app_commands.describe(model="Diffusion model in models/diffusion_models (optional)")
@app_commands.describe(negative="Negative prompt")
@app_commands.describe(steps="Sampling steps")
@app_commands.describe(cfg="CFG guidance scale")
@app_commands.describe(width="Width in pixels")
@app_commands.describe(height="Height in pixels")
@app_commands.describe(sampler="Sampler (optional)")
@app_commands.describe(scheduler="Scheduler (optional)")
@app_commands.describe(lora1="First LoRA (optional)")
@app_commands.describe(lora2="Second LoRA (optional)")
@app_commands.describe(lora_strength="LoRA strength (optional, default 1.0)")
@app_commands.describe(seed="Seed (optional)")
@app_commands.describe(batch_size="Number of images to generate (optional)")
@app_commands.describe(stealth="Ephemeral output, visible only to you")
async def zimage(interaction: discord.Interaction, prompt: str,
                 model: str | None = None,
                 negative: str | None = None,
                 steps: int | None = None,
                 cfg: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 sampler: str | None = None,
                 scheduler: str | None = None,
                 lora1: str | None = None,
                 lora2: str | None = None,
                 lora_strength: float | None = None,
                 seed: int | None = None,
                 batch_size: int | None = None,
                 stealth: bool | None = None):
    if stealth is None:
        stealth = bool(user_settings.get_settings(interaction.user.id).get("stealth", False))
    if moderation.is_banned(interaction.user.id):
        await interaction.response.send_message(
            content="\U0001f6ab You are banned from using this bot. Please contact an admin.", ephemeral=True
        )
        return
    if not await check_cooldown(interaction):
        await interaction.response.send_message(
            content="\u23f3 Please wait before requesting another image.", ephemeral=True
        )
        return

    if nsfw_blocked(interaction, prompt):
        await interaction.response.send_message(
            content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    settings = user_settings.get_settings(interaction.user.id)
    if negative is None:
        negative = settings["negative_prompt"] or None
    if steps is None:
        steps = settings.get("zimage_steps")
    if cfg is None:
        cfg = settings.get("zimage_cfg")
    if width is None:
        width = settings.get("width")
    if height is None:
        height = settings.get("height")
    if sampler is None:
        sampler = settings.get("zimage_sampler")
    if scheduler is None:
        scheduler = settings.get("zimage_scheduler")

    # Per-user default Z-Image diffusion model (set via /settings). Used only
    # by /zimage when no explicit model is passed; silently ignored if
    # unavailable.
    if model is None:
        saved = settings.get("zimage_model")
        if saved:
            try:
                available = await comfy.fetch_diffusion_models()
            except Exception as exc:
                log.warning("Could not list diffusion models: %s", exc)
                available = None
            if available is not None:
                if saved in available:
                    model = saved
                else:
                    log.info("Saved Z-Image model %r not available; using workflow default.", saved)

    # Optional diffusion model selection: any file in ComfyUI's
    # models/diffusion_models folder can be used instead of the workflow's
    # default model.
    if model is not None:
        try:
            available = await comfy.fetch_diffusion_models()
        except Exception as exc:
            log.warning("Could not list diffusion models: %s", exc)
            await interaction.response.send_message(
                content=f"\u274c Could not query ComfyUI for available diffusion models: {exc}",
                ephemeral=True,
            )
            return
        if model not in available:
            await interaction.response.send_message(
                content=f"\u274c Model \u201c{model}\u201d was not found in models/diffusion_models.",
                ephemeral=True,
            )
            return

    gen_kwargs = {
        "prompt": prompt,
        "negative": negative,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "sampler": sampler,
        "scheduler": scheduler,
        "batch_size": batch_size,
        "ckpt_name": model,
        "lora1": lora1,
        "lora2": lora2,
        "lora_strength": lora_strength,
    }
    await run_t2i_generation(interaction, "zimage", prompt, stealth, gen_kwargs)


@bot.tree.command(name="upscale", description="Upscale an image from your gallery")
@app_commands.choices(model=UPSCALE_CHOICES)
@app_commands.describe(image="Image to upscale (attach from your gallery)")
@app_commands.describe(prompt="Prompt to guide the upscale (optional)")
@app_commands.describe(negative="Negative prompt (optional)")
@app_commands.describe(strength="Denoise strength, 0-1 (optional)")
@app_commands.describe(sampler="Sampler (SDXL only, optional)")
@app_commands.describe(scheduler="Scheduler (SDXL only, optional)")
@app_commands.describe(scale="Upscale factor (optional, e.g. 2 or 3)")
@app_commands.describe(stealth="Ephemeral output, visible only to you")
@app_commands.choices(sampler=SAMPLER_CHOICES)
@app_commands.choices(scheduler=SCHEDULER_CHOICES)
async def upscale(interaction: discord.Interaction, model: str, image: discord.Attachment,
                  prompt: str | None = None,
                  negative: str | None = None, strength: float | None = None,
                  sampler: str | None = None,
                  scheduler: str | None = None,
                  scale: float | None = None, stealth: bool | None = None):
    if stealth is None:
        stealth = bool(user_settings.get_settings(interaction.user.id).get("stealth", False))
    if moderation.is_banned(interaction.user.id):
        await interaction.response.send_message(
            content="\U0001f6ab You are banned from using this bot. Please contact an admin.", ephemeral=True
        )
        return
    if not await check_cooldown(interaction):
        await interaction.response.send_message(
            content="\u23f3 Please wait before requesting another image.", ephemeral=True
        )
        return

    if nsfw_blocked(interaction, prompt):
        await interaction.response.send_message(
            content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    image_url = str(image.url)
    try:
        session = get_session()
        async with session.get(image_url) as resp:
            resp.raise_for_status()
            data = await resp.read()
        input_longest_side = None
        if scale is not None:
            # SeedVR2 targets absolute pixels, so we need the input image size.
            img = Image.open(io.BytesIO(data))
            input_longest_side = max(img.size)
        uploaded_name = await comfy.upload_image(data, f"discord_{uuid_hex()}.png")
        settings = user_settings.get_settings(interaction.user.id)
        spec = config["models"].get(model, {}).get("upscale")
        if spec is None:
            await reply_error(interaction, f"\u274c Model {model!r} has no upscaling workflow.")
            return
        if negative is None:
            negative = settings["negative_prompt"] or None
        # SDXL upscale: fall back to the user's saved SDXL sampler/scheduler defaults.
        if model == "sdxl":
            if sampler is None:
                sampler = settings.get("sdxl_sampler")
            if scheduler is None:
                scheduler = settings.get("sdxl_scheduler")

        # SDXL upscale: let the user pick any checkpoint from models/checkpoints.
        # Show an ephemeral picker; the selected button runs the upscale.
        if model == "sdxl":
            # Defer first (within Discord's 3-second window) so the interaction
            # token doesn't expire while fetch_checkpoints() queries ComfyUI.
            await interaction.response.defer(ephemeral=True)
            msg = await interaction.original_response()
            try:
                checkpoints = await comfy.fetch_checkpoints()
            except Exception as exc:
                log.warning("Could not fetch checkpoints for sdxl upscale: %s", exc)
                checkpoints = []
            await msg.edit(
                content="\u2705 Image uploaded. Pick a checkpoint to upscale with:",
                view=CheckpointPickerView(
                    spec, model, uploaded_name, input_longest_side, stealth,
                    prompt, negative, strength, scale, checkpoints,
                    sampler, scheduler,
                ),
            )
            return

        await interaction.response.send_message(
            content=job_queue.waiting_prefix("comfyui") + "\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress_msg = await interaction.original_response()
        progress = ProgressUpdater(progress_msg)
        images, meta = await job_queue.submit(
            run_image(
                spec, on_progress=progress.update, model_key=model, prompt=prompt,
                negative=negative, strength=strength, image_filename=uploaded_name,
                scale=scale, input_longest_side=input_longest_side,
            ),
            lane="comfyui",
            name=f"{model}_upscale",
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
            files.append(discord.File(io.BytesIO(img_bytes), filename=f"{model}_upscale_{i}{ext}"))
        base_desc = (
            f"**Model:** {model}"
            + (f"\n**Scale:** {scale:g}x" if scale is not None else "")
            + f"\n**Resolution:** {image_resolution(images[0])}"
        )
        embed = discord.Embed(
            description=base_desc + "\n" + "\n".join(meta_lines(meta)),
            color=discord.Color.green(),
        )
        response_msg = await interaction.edit_original_response(content="", embed=embed, attachments=files, view=GenerationView(stealth=stealth))
        generation_store.save(response_msg.id, {
            "spec": spec, "model": model, "suffix": "upscale", "stealth": stealth,
            "embed_desc": base_desc, "embed_color": int(embed.color),
            "user_id": interaction.user.id,
            # Retries reuse the uploaded input image and roll a fresh seed.
            "kwargs": {"prompt": prompt, "negative": negative, "strength": strength,
                       "image_filename": uploaded_name, "scale": scale,
                       "input_longest_side": input_longest_side},
        })
    except (ComfyUIError, Exception) as exc:
        # progress is only bound on the non-sdxl path; guard so an error in the
        # sdxl branch (which returns early) doesn't raise UnboundLocalError.
        try:
            progress.done = True
        except UnboundLocalError:
            pass
        logging.getLogger("bot").exception("upscale failed")
        await reply_error(interaction, f"\u274c Upscaling failed: {exc}")


@bot.tree.command(name="img2img", description="Edit one image (or combine two) into a new image using Flux 2 Klein 4B Base")
@app_commands.choices(workflow=I2I_WORKFLOW_CHOICES)
@app_commands.choices(sampler=SAMPLER_CHOICES)
@app_commands.describe(workflow="Workflow: 1 image (edit) or 2 images (combine)")
@app_commands.describe(image="First input image (attach from your gallery)")
@app_commands.describe(prompt="Prompt describing the desired edit")
@app_commands.describe(image2="Second input image (required when workflow is 2 images)")
@app_commands.describe(cfg="CFG guidance scale (optional)")
@app_commands.describe(steps="Sampling steps (optional)")
@app_commands.describe(sampler="Sampler (optional)")
@app_commands.describe(megapixels="Target resolution in megapixels (optional)")
@app_commands.describe(stealth="Ephemeral output, visible only to you")
async def img2img(interaction: discord.Interaction, workflow: str,
                  image: discord.Attachment, prompt: str,
                  image2: discord.Attachment | None = None,
                  cfg: float | None = None, steps: int | None = None,
                  sampler: str | None = None, megapixels: int | None = None,
                  stealth: bool | None = None):
    if stealth is None:
        stealth = bool(user_settings.get_settings(interaction.user.id).get("stealth", False))
    if moderation.is_banned(interaction.user.id):
        await interaction.response.send_message(
            content="\U0001f6ab You are banned from using this bot. Please contact an admin.", ephemeral=True
        )
        return
    if not await check_cooldown(interaction):
        await interaction.response.send_message(
            content="\u23f3 Please wait before requesting another image.", ephemeral=True
        )
        return

    if nsfw_blocked(interaction, prompt):
        await interaction.response.send_message(
            content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    need_two = workflow == "multi"
    if need_two and image2 is None:
        await interaction.response.send_message(
            content="\u26a0\ufe0f The 2-image workflow needs two images. Please attach a second image.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    await interaction.response.send_message(
        content=job_queue.waiting_prefix("comfyui") + "\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
        ephemeral=stealth,
    )
    progress_msg = await interaction.original_response()
    progress = ProgressUpdater(progress_msg)
    log = logging.getLogger("bot")
    log.info("img2img: workflow=%s prompt=%r", workflow, prompt)

    model = "flux2_klein"
    spec_key = "i2i_single" if workflow == "single" else "i2i_multi"
    spec = config["models"].get(model, {}).get(spec_key)
    if spec is None:
        progress.done = True
        await reply_error(interaction, f"\u274c Workflow {spec_key!r} is not configured.", target=progress_msg)
        return

    # Download and upload the input image(s) to ComfyUI.
    try:
        session = get_session()
        async with session.get(str(image.url)) as resp:
            resp.raise_for_status()
            data1 = await resp.read()
        uploaded1 = await comfy.upload_image(data1, f"discord_{uuid_hex()}.png")
        uploaded_files = [uploaded1]
        if need_two:
            async with session.get(str(image2.url)) as resp:
                resp.raise_for_status()
                data2 = await resp.read()
            uploaded2 = await comfy.upload_image(data2, f"discord_{uuid_hex()}.png")
            uploaded_files.append(uploaded2)
    except Exception as exc:
        progress.done = True
        log.exception("img2img download/upload failed")
        await reply_error(interaction, f"\u274c Could not process the input images: {exc}", target=progress_msg)
        return

    # Fall back to the user's saved img2img defaults for any parameter left unset.
    saved = user_settings.get_settings(interaction.user.id)
    if cfg is None:
        cfg = saved.get("img2img_cfg")
    if steps is None:
        steps = saved.get("img2img_steps")
    if sampler is None:
        sampler = saved.get("img2img_sampler")
    if megapixels is None:
        megapixels = saved.get("img2img_megapixels")

    gen_kwargs = {
        "prompt": prompt,
        "seed": None,
        "cfg": cfg,
        "steps": steps,
        "sampler": sampler,
        "megapixels": megapixels,
    }
    if need_two:
        gen_kwargs["image_files"] = uploaded_files
    else:
        gen_kwargs["image_filename"] = uploaded_files[0]

    try:
        images, meta = await job_queue.submit(
            run_image(
                spec, on_progress=progress.update, model_key=model, **gen_kwargs
            ),
            lane="comfyui",
            name="flux2_klein_i2i",
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
            files.append(discord.File(io.BytesIO(img_bytes), filename=f"flux2_klein_i2i_{i}{ext}"))
        workflow_label = "1 image (edit)" if workflow == "single" else "2 images (combine)"
        base_lines = [
            f"**Model:** {model}",
            f"**Workflow:** {workflow_label}",
            f"**Prompt:** {prompt}",
            f"**Resolution:** {image_resolution(images[0])}",
        ]
        base_desc = "\n".join(base_lines)
        embed = discord.Embed(
            description=base_desc + "\n" + "\n".join(meta_lines(meta)),
            color=discord.Color.orange(),
        )
        response_msg = await interaction.edit_original_response(content="", embed=embed, attachments=files, view=GenerationView(stealth=stealth))
        generation_store.save(response_msg.id, {
            "spec": spec, "model": model, "suffix": "i2i", "stealth": stealth,
            "embed_desc": base_desc, "embed_color": int(embed.color),
            "user_id": interaction.user.id,
            "kwargs": {**gen_kwargs, "seed": None},
        })
    except Exception as exc:
        progress.done = True
        log.exception("img2img failed")
        await reply_error(interaction, f"\u274c Image-to-image failed: {exc}", target=progress_msg)
@bot.tree.command(name="flush", description="Unload all models and execution cache from ComfyUI")
async def flush(interaction: discord.Interaction):
    await interaction.response.defer()
    log = logging.getLogger("bot")
    log.info("flush: freeing ComfyUI memory")
    try:
        # Wait for every queued generation / prompt job to finish first,
        # so /flush never interrupts a running job.
        await job_queue.wait_drained()
        await comfy.free_memory()
        await interaction.edit_original_response(content="\U0001f9f9 Done. All models and execution cache have been unloaded from ComfyUI.")
    except ComfyUIError as exc:
        await reply_error(interaction, f"\u274c Flush failed: {exc}")
