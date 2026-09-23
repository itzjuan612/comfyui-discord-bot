import json
import os
import random

from core import config, comfy, log, normalize_aspect_ratio, last_workflow_by_model, last_ckpt_by_model, ComfyUIError, user_settings


# Values that disable the optional LoRA loader. Sentinels usable in the
# `lora` command parameter to override a config default for one run.
_LORA_DISABLE_VALUES = {"", "none", "off", "disable", "disabled"}


def _resolve_lora(spec: dict, kwargs: dict) -> str:
    """Effective LoRA filename: command kwarg -> config ``lora`` default.

    Sentinel values (empty string, "none", "off", ...) and a missing default
    both resolve to "" which disables the loader node (apply_spec then removes
    it and rewires the model chain around it).
    """
    lora = kwargs.get("lora")
    if lora is None:
        lora = spec.get("lora")
    if lora is None:
        return ""
    if str(lora).strip().lower() in _LORA_DISABLE_VALUES:
        return ""
    return str(lora)


def load_workflow(file_path: str) -> dict:
    import json

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    with open(path) as f:
        return json.load(f)


def graph_to_api(graph: dict) -> dict:
    """Convert a ComfyUI graph-format workflow into API/prompt format."""
    links = {l[0]: (l[1], l[2]) for l in graph.get("links", [])}
    api = {}
    for node in graph["nodes"]:
        ctype = node["type"]
        if ctype in ("Note", "MarkdownNote"):
            continue
        nid = str(node["id"])
        inputs = {}
        named = node.get("widgets_values_named") or {}
        positional = node.get("widgets_values") or []
        pos_idx = 0
        for inp in node.get("inputs", []):
            name = inp["name"]
            if name in named:
                inputs[name] = named[name]
            elif "widget" in inp:
                inputs[name] = positional[pos_idx]
                pos_idx += 1
            if inp.get("link") is not None:
                src, slot = links[inp["link"]]
                inputs[name] = [str(src), slot]
        api[nid] = {"class_type": ctype, "inputs": inputs}
    return api


def apply_spec(workflow: dict, spec: dict, **kwargs) -> None:
    """Patch node inputs in the workflow based on config node IDs."""

    def set_node(node_id, key, value):
        if node_id is not None:
            workflow[str(node_id)]["inputs"][key] = value

    ckpt_name = kwargs.get("ckpt_name")
    if ckpt_name is not None:
        # Point the model loader at the requested file. Works with SDXL
        # checkpoints (models/checkpoints) and with UNet/DiT model files.
        # Node type -> input key used to name the model file.
        loader_keys = {
            "CheckpointLoaderSimple": "ckpt_name",
            "UNETLoader": "unet_name",
            "SeedVR2LoadDiTModel": "model",
            "FlashVSRNode": "model",
        }
        # Ideogram has two UNet loaders (conditional + unconditional); the
        # spec's model_node pins which one a default_model refers to.
        model_node = spec.get("model_node")
        for nid, node in workflow.items():
            key = loader_keys.get(node.get("class_type"))
            if key is None:
                continue
            if model_node is not None and str(nid) != str(model_node):
                continue
            node["inputs"][key] = ckpt_name
            break
        else:
            raise ValueError(f"Workflow has no model-loader node; cannot select model {ckpt_name!r}.")

    # Ideogram's second UNet loader (unconditional branch) can be overridden
    # independently via the spec's default_model_unconditional / model_node_unconditional.
    default_model_unconditional = spec.get("default_model_unconditional")
    model_node_unconditional = spec.get("model_node_unconditional")
    if default_model_unconditional is not None and model_node_unconditional is not None:
        node = workflow.get(str(model_node_unconditional))
        if node is not None and node.get("class_type") == "UNETLoader":
            node["inputs"]["unet_name"] = default_model_unconditional

    # Text-encoder default (e.g. the Qwen 3 VL quant shared by Ideogram 4 and
    # Qwen Image 2.1). An explicit ``text_encoder`` kwarg wins; otherwise the
    # spec's default is applied so the workflow never depends on whatever
    # filename was saved in the JSON. ``enhancer_text_encoder_node`` (when
    # present) points at a second CLIPLoader and receives the same file.
    if spec.get("text_encoder_node") is not None or spec.get("enhancer_text_encoder_node") is not None:
        effective_encoder = kwargs.get("text_encoder") or spec.get("default_text_encoder")
        if effective_encoder is not None:
            for encoder_key in ("text_encoder_node", "enhancer_text_encoder_node"):
                node_id = spec.get(encoder_key)
                if node_id is None:
                    continue
                node = workflow.get(str(node_id))
                if node is not None and node.get("class_type") == "CLIPLoader":
                    node["inputs"]["clip_name"] = effective_encoder

    prompt = kwargs.get("prompt")
    negative = kwargs.get("negative")
    seed = kwargs.get("seed")
    steps = kwargs.get("steps")
    strength = kwargs.get("strength")
    image_filename = kwargs.get("image_filename")
    width = kwargs.get("width")
    height = kwargs.get("height")
    cfg = kwargs.get("cfg")

    if prompt is not None:
        input_node = spec.get("prompt_input_node")
        if input_node is not None:
            # Prompt enters through an intermediate node (e.g. Qwen's Google
            # Translate node feeding the optional prompt enhancer); the
            # encode node's prompt input is a link into that chain, so
            # writing to prompt_node would clobber the link and bypass it.
            set_node(input_node, spec.get("prompt_input_key", "text"), prompt)
        else:
            set_node(spec.get("prompt_node"), spec.get("prompt_key", "text"), prompt)
    # Qwen Image 2.1 T2I: optional prompt translation. The GoogleTranslateTextNode
    # input is inverted (manual_translate: True = pass through untranslated,
    # False = translate to English), so the node receives the opposite value.
    # None leaves the workflow default (translate) untouched.
    translate = kwargs.get("translate")
    if translate is not None:
        set_node(spec.get("translate_node"), spec.get("translate_key", "manual_translate"),
                 not bool(translate))
    # Qwen Image 2.1 T2I: optional prompt enhancement chain between the
    # translate node and the encoder (StringConcatenate -> TextGenerate ->
    # showAnything). enhance=False removes those nodes and rewires the
    # encoder's prompt input directly to the translate node; None/True
    # leaves the workflow's saved chain (enabled) untouched.
    enhance = kwargs.get("enhance")
    if enhance is not None and not bool(enhance):
        for node_id in spec.get("enhance_nodes") or []:
            workflow.pop(str(node_id), None)
        prompt_node = spec.get("prompt_node")
        input_node = spec.get("prompt_input_node")
        if prompt_node is not None and input_node is not None:
            workflow[str(prompt_node)]["inputs"][spec.get("prompt_key", "prompt")] = [
                str(input_node),
                0,
            ]
    if negative is not None:
        set_node(spec.get("negative_node"), spec.get("negative_key", "text"), negative)
    if seed is not None:
        set_node(spec.get("seed_node"), spec.get("seed_key", "seed"), int(seed))
    if steps is not None:
        set_node(spec.get("steps_node"), "steps", int(steps))
    if strength is not None:
        set_node(spec.get("denoise_node"), "denoise", float(strength))
    if image_filename is not None:
        set_node(spec.get("image_node"), "image", image_filename)
    image_files = kwargs.get("image_files")
    if image_files:
        image_nodes = spec.get("image_nodes")
        if image_nodes:
            for node_id, fname in zip(image_nodes, image_files):
                workflow[str(node_id)]["inputs"]["image"] = fname
    # Qwen edit's optional second image (single spec for 1- and 2-image
    # edits). When omitted, the second LoadImage node is removed and its
    # input is dropped from the encode node so ComfyUI never validates an
    # empty LoadImage.
    image2_node = spec.get("image2_node")
    if image2_node is not None:
        image2_filename = kwargs.get("image2_filename")
        if image2_filename:
            workflow[str(image2_node)]["inputs"]["image"] = image2_filename
        else:
            workflow.pop(str(image2_node), None)
            encode_node = workflow.get(str(spec.get("prompt_node")))
            if encode_node is not None:
                encode_node["inputs"].pop(spec.get("image2_key", "images.image_2"), None)
    if width is not None and height is not None:
        latent_id = spec.get("latent_node")
        if latent_id is not None:
            workflow[str(latent_id)]["inputs"]["width"] = int(width)
            workflow[str(latent_id)]["inputs"]["height"] = int(height)
    if cfg is not None:
        set_node(spec.get("cfg_node"), "cfg", float(cfg))
    sampler = kwargs.get("sampler")
    if sampler is not None:
        set_node(spec.get("sampler_node"), "sampler_name", sampler)
    scheduler = kwargs.get("scheduler")
    if scheduler is not None:
        set_node(spec.get("sampler_node"), "scheduler", scheduler)

    # Batch size: applied to the empty latent node when provided.
    batch_size = kwargs.get("batch_size")
    if batch_size is not None:
        latent_id = spec.get("latent_node")
        if latent_id is not None:
            workflow[str(latent_id)]["inputs"]["batch_size"] = int(batch_size)

    # LoRA support: an empty lora name disables the loader node. Since
    # ComfyUI validates lora_name against the available-loras list ("" is
    # not a valid value), disabled loaders are removed from the graph and
    # the model/clip chain is rewired around them.
    lora1 = kwargs.get("lora1")
    lora2 = kwargs.get("lora2")
    lora = _resolve_lora(spec, kwargs)
    lora_strength = kwargs.get("lora_strength")

    if spec.get("lora_strength_node") is not None:
        # Unified LoRA strength (e.g. Z-Image): a single PrimitiveFloat node
        # drives the strength of every LoRA loader via its strength_model
        # link. Set that node's value once; disabled loaders are removed and
        # the model-only chain is rewired around the active loaders.
        lora_strength_node = str(spec["lora_strength_node"])
        # Apply the unified strength only when the user provides it; otherwise
        # leave the workflow's PrimitiveFloat default value untouched.
        if lora_strength is not None:
            workflow[lora_strength_node]["inputs"]["value"] = float(lora_strength)
        active_nodes = []
        for lora_name, lora_node in ((lora1, spec.get("lora1_node")), (lora2, spec.get("lora2_node"))):
            if lora_node is None:
                continue
            nid = str(lora_node)
            if lora_name:
                workflow[nid]["inputs"]["lora_name"] = lora_name
                active_nodes.append(nid)
            else:
                workflow.pop(nid)  # disabled: remove so it is not validated
        chain_start = spec.get("model_chain_start")
        chain_end = spec.get("model_chain_end")
        if chain_start is not None and chain_end is not None:
            # Reconnect the model chain: source -> active LoRAs (in order) -> target.
            prev = str(chain_start)
            for nid in active_nodes:
                workflow[nid]["inputs"]["model"] = [prev, 0]
                prev = nid
            workflow[str(chain_end)]["inputs"]["model"] = [prev, 0]
    elif spec.get("model_chain_start") is not None and spec.get("lora1_node") is not None:
        # Single model-only LoRA loader in a plain chain (Qwen Image 2.1:
        # UNETLoader -> LoraLoaderModelOnly -> APG). The effective name is the
        # command's `lora` param or the config `lora` default; empty/missing
        # removes the node and rewires the chain straight from start to end.
        nid = str(spec["lora1_node"])
        if lora:
            node = workflow[nid]
            node["inputs"]["lora_name"] = lora
            if lora_strength is not None:
                node["inputs"]["strength_model"] = float(lora_strength)
        else:
            workflow.pop(nid, None)  # disabled: remove so it is not validated
            chain_start = spec.get("model_chain_start")
            chain_end = spec.get("model_chain_end")
            if chain_start is not None and chain_end is not None:
                workflow[str(chain_end)]["inputs"]["model"] = [str(chain_start), 0]
    else:
        strength_val = float(lora_strength) if lora_strength is not None else None
        active = []
        for lora_name, lora_node in ((lora1, spec.get("lora1_node")), (lora2, spec.get("lora2_node"))):
            if lora_node is None:
                continue
            nid = str(lora_node)
            if lora_name:
                node = workflow[nid]
                node["inputs"]["lora_name"] = lora_name
                if strength_val is not None:
                    node["inputs"]["strength_model"] = strength_val
                    node["inputs"]["strength_clip"] = strength_val
                active.append(nid)
            else:
                workflow.pop(nid)  # disabled: remove so it is not validated
        if active or spec.get("lora1_node") is not None or spec.get("lora2_node") is not None:
            # Rebuild the chain: checkpoint (model) / CLIP switch (clip) ->
            # active LoRA loaders -> sampler (model) and text encoders (clip).
            ckpt_nid = next((str(k) for k, v in workflow.items()
                              if v["class_type"] == "CheckpointLoaderSimple"), None)
            clip_nid = next((str(k) for k, v in workflow.items()
                              if v["class_type"] == "CLIP Input Switch"), None)
            model_src, clip_src = ckpt_nid, clip_nid
            for nid in active:
                # Clip output index: 0 from the CLIP switch, 1 from a LoraLoader.
                workflow[nid]["inputs"]["model"] = [model_src, 0]
                workflow[nid]["inputs"]["clip"] = [clip_src, 0 if clip_src == clip_nid else 1]
                model_src, clip_src = nid, nid
            clip_out = 1 if active else 0
            sampler_nid = str(spec.get("steps_node"))
            workflow[sampler_nid]["inputs"]["model"] = [model_src, 0]
            for nid in (spec.get("prompt_node"), spec.get("negative_node")):
                workflow[str(nid)]["inputs"]["clip"] = [clip_src, clip_out]

    # Ideogram: resolution selector (megapixels + aspect ratio) and quality preset.
    megapixels = kwargs.get("megapixels")
    aspect_ratio = kwargs.get("aspect_ratio")
    quality = kwargs.get("quality")
    resolution_node = spec.get("resolution_node")
    if megapixels is not None and resolution_node is not None:
        workflow[str(resolution_node)]["inputs"][spec.get("megapixels_key", "megapixels")] = int(megapixels)
    if megapixels is not None:
        mp_key = spec.get("megapixels_key", "megapixels")
        for node_id in spec.get("megapixels_nodes") or []:
            workflow[str(node_id)]["inputs"][mp_key] = int(megapixels)
    if aspect_ratio is not None and resolution_node is not None:
        aspect_ratio = normalize_aspect_ratio(aspect_ratio)
        workflow[str(resolution_node)]["inputs"][spec.get("aspect_ratio_key", "aspect_ratio")] = aspect_ratio
    if quality is not None:
        set_node(spec.get("quality_node"), spec.get("quality_key", "choice"), quality)

    scale = kwargs.get("scale")
    input_longest_side = kwargs.get("input_longest_side")
    if scale is not None:
        scale_id = spec.get("scale_node")
        if scale_id is not None:
            mode = spec.get("scale_mode", "factor")
            if mode == "resolution":
                # Target absolute pixels = longest input side * scale.
                if input_longest_side is None:
                    raise ValueError("Scale requires the input image dimensions.")
                value = int(input_longest_side * float(scale))
                keys = spec.get("scale_keys") or [spec.get("scale_key", "resolution")]
                for key in keys:
                    workflow[str(scale_id)]["inputs"][key] = value
            else:
                # Plain multiplicative scale factor (e.g. 2x, 3x).
                scale_id_str = str(scale_id)
                factor = float(scale)
                value = int(factor) if factor.is_integer() else factor
                workflow[scale_id_str]["inputs"][spec.get("scale_key", "scale_by")] = value


# Runtime GPU-safety bounds. Command decorators use app_commands.Range for
# immediate user-facing errors, but values coming from /settings, persisted
# retry params, and older messages all funnel through here.
_KW_BOUNDS = {
    "steps": (1, 150),
    "width": (64, 4096),
    "height": (64, 4096),
    "cfg": (0.5, 20.0),
    "batch_size": (1, 8),
    "megapixels": (1, 8),
    "scale": (1.0, 4.0),
}


def _clamp_kwargs(kwargs: dict) -> None:
    for key, (low, high) in _KW_BOUNDS.items():
        value = kwargs.get(key)
        if value is None:
            continue
        clamped = max(low, min(high, value))
        if clamped != value:
            log.warning("Clamping %s=%s to %s", key, value, clamped)
            kwargs[key] = clamped


async def run_image(spec: dict, on_progress=None, **kwargs):
    _clamp_kwargs(kwargs)
    if kwargs.get("seed") is None:
        kwargs["seed"] = random.randint(0, 2**32 - 1)

    # Free VRAM/RAM when switching to a different workflow, so models
    # from the previous workflow don't stay loaded.
    model_key = kwargs.get("model_key", "default")
    last = last_workflow_by_model.get(model_key)
    workflow_changed = last is not None and last != spec["file"]
    if workflow_changed:
        log.info("Switching workflow %s -> %s; freeing memory", last, spec["file"])
        await comfy.free_memory()
    last_workflow_by_model[model_key] = spec["file"]

    workflow = load_workflow(spec["file"])
    api_workflow = graph_to_api(workflow) if "nodes" in workflow else workflow

    # Determine the effective model file. Priority: explicit request
    # (ckpt_name) > config default_model > the workflow's own value. The
    # checkpoints-availability fallback applies only to SDXL, whose files live
    # in models/checkpoints (UNet/DiT models live in different folders).
    effective_ckpt = kwargs.get("ckpt_name")
    if effective_ckpt is None:
        default_ckpt = spec.get("default_model")
        if default_ckpt is None:
            for node in api_workflow.values():
                if node.get("class_type") in ("CheckpointLoaderSimple", "UNETLoader", "SeedVR2LoadDiTModel", "FlashVSRNode"):
                    key = {"CheckpointLoaderSimple": "ckpt_name", "UNETLoader": "unet_name",
                           "SeedVR2LoadDiTModel": "model", "FlashVSRNode": "model"}.get(node["class_type"])
                    if key and node["inputs"].get(key):
                        default_ckpt = node["inputs"][key]
                        break
        if default_ckpt is not None:
            if model_key == "sdxl":
                try:
                    available = await comfy.fetch_checkpoints()
                except Exception as exc:
                    log.warning("Could not list checkpoints for fallback: %s", exc)
                    available = []
                if available and default_ckpt not in available:
                    effective_ckpt = next((c for c in available if "sdxl" in c.lower()), available[0])
                    log.info("Default checkpoint %r not available; falling back to %r", default_ckpt, effective_ckpt)
                else:
                    effective_ckpt = default_ckpt
                kwargs["ckpt_name"] = effective_ckpt
            else:
                # UNet/DiT models live in different folders, so no availability
                # check — trust the config value (falling back to the workflow).
                effective_ckpt = default_ckpt
                kwargs["ckpt_name"] = effective_ckpt
    last_ckpt = last_ckpt_by_model.get(model_key)
    if not workflow_changed and last_ckpt is not None and last_ckpt != effective_ckpt:
        log.info("Switching checkpoint %s -> %s; freeing memory", last_ckpt, effective_ckpt)
        await comfy.free_memory()
    last_ckpt_by_model[model_key] = effective_ckpt

    # Qwen LoRA: resolve command param -> config default, then fall back to
    # disabled when the file is not installed in ComfyUI's models/loras
    # folder (checked against the TTL cache, refreshed once), so generation
    # still runs instead of failing LoRA validation. Only specs that declare
    # a `lora` default key (the Qwen workflows) are checked; SDXL/Z-Image
    # pass their own lora1/lora2 kwargs and keep their existing behavior.
    if spec.get("lora") is not None or kwargs.get("lora") is not None:
        lora = _resolve_lora(spec, kwargs)
        if lora and spec.get("lora1_node") is not None:
            try:
                available = await comfy.fetch_loras()
                if lora not in available:
                    available = await comfy.fetch_loras(force=True)
            except Exception as exc:
                log.warning("Could not list LoRAs; keeping %r as-is: %s", lora, exc)
                available = None
            if available is not None and lora not in available:
                log.warning("LoRA %r not found in models/loras; disabling the LoRA node.", lora)
                lora = ""
        kwargs["lora"] = lora

    apply_spec(api_workflow, spec, **kwargs)

    # Capture the exact parameters actually used, so the output embed can
    # display the seed, steps and CFG (and the checkpoint actually run).
    meta = {"seed": kwargs["seed"], "ckpt_name": effective_ckpt}
    if spec.get("steps_node") is not None:
        meta["steps"] = api_workflow[str(spec["steps_node"])]["inputs"].get("steps")
    if spec.get("cfg_node") is not None:
        meta["cfg"] = api_workflow[str(spec["cfg_node"])]["inputs"].get("cfg")
    if spec.get("sampler_node") is not None:
        meta["sampler"] = api_workflow[str(spec["sampler_node"])]["inputs"].get("sampler_name")
        meta["scheduler"] = api_workflow[str(spec["sampler_node"])]["inputs"].get("scheduler")

    # If this checkpoint was previously seen without a bundled text
    # encoder/VAE, use the separate loaders straight away (no error/retry).
    switch_node = spec.get("switch_node")
    if switch_node is not None and effective_ckpt and user_settings.is_split_checkpoint(effective_ckpt):
        api_workflow[str(switch_node)]["inputs"]["value"] = True
        log.info("Checkpoint %r cached as split; using separate CLIP/VAE loaders", effective_ckpt)

    prompt_id, client_id = await comfy.queue_prompt(api_workflow)
    try:
        filenames = await comfy.wait_for_result(prompt_id, client_id, on_progress=on_progress)
    except ComfyUIError as exc:
        # If the checkpoint has no bundled text encoder/VAE, the run fails
        # (ComfyUI reports a generic "error"). Flip the switch node so the
        # separate CLIPLoader/VAELoader are used, and retry once.
        if switch_node is not None and api_workflow[str(switch_node)]["inputs"]["value"] is False:
            log.info("Checkpoint %r likely lacks bundled CLIP/VAE; enabling split loaders and retrying", effective_ckpt)
            user_settings.mark_split_checkpoint(effective_ckpt)
            api_workflow[str(switch_node)]["inputs"]["value"] = True
            prompt_id, client_id = await comfy.queue_prompt(api_workflow)
            filenames = await comfy.wait_for_result(prompt_id, client_id, on_progress=on_progress)
        else:
            raise
    images = []
    for filename in filenames:
        images.append(await comfy.fetch_image(filename))
    return images, meta


async def run_text_workflow(file_path: str, patches: dict, target_node: str | None = None) -> str:
    """Run a workflow that outputs text (not images) and return the text.

    ``patches`` maps node id -> dict of input overrides applied before queueing.
    ``target_node`` limits the result to a specific node's output, so numeric
    outputs from intermediate nodes (e.g. ComfyMathExpression) are ignored.
    """
    workflow = load_workflow(file_path)
    if "nodes" in workflow:
        workflow = graph_to_api(workflow)
    for node_id, inputs in patches.items():
        for key, value in inputs.items():
            workflow[str(node_id)]["inputs"][key] = value
    async def _run() -> str:
        prompt_id, _client_id = await comfy.queue_prompt(workflow)
        return await comfy.wait_for_output_text(prompt_id, target_node=target_node)

    # Text workflows occupy ComfyUI's GPU. Their callers (gen_prompt) hold the
    # LLM lane, so in "separate" mode this work would run concurrently with
    # image jobs on the comfyui lane and race their free_memory calls; route it
    # through that lane so all ComfyUI execution stays serial. In unified mode
    # everything already shares one lane.
    from job_queue import job_queue
    if job_queue.mode == "separate":
        return await job_queue.submit(_run(), lane="comfyui", name="prompt_gen_workflow")
    return await _run()
