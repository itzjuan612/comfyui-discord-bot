import asyncio
import json
import os
import random

from core import (
    ComfyUIError,
    comfy,
    last_ckpt_by_model,
    last_workflow_by_model,
    log,
    normalize_aspect_ratio,
    user_settings,
)

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
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    with open(path) as f:
        return json.load(f)


# Node classes that only carry documentation in the graph editor; ComfyUI
# rejects them in API format.
_NOTE_NODES = ("Note", "MarkdownNote")


def _link_map(graph: dict) -> dict:
    """Map link id -> (origin node id, origin slot)."""
    return {link[0]: (link[1], link[2]) for link in graph.get("links", [])}


def _node_inputs(node: dict, links: dict) -> dict:
    """Resolve one graph node's inputs: widget values plus incoming links."""
    named = node.get("widgets_values_named") or {}
    positional = node.get("widgets_values") or []
    pos_idx = 0
    inputs = {}
    for inp in node.get("inputs", []):
        name = inp["name"]
        if name in named:
            inputs[name] = named[name]
        elif "widget" in inp:
            # Widget values without names are stored positionally.
            inputs[name] = positional[pos_idx]
            pos_idx += 1
        if inp.get("link") is not None:
            src, slot = links[inp["link"]]
            inputs[name] = [str(src), slot]
    return inputs


def graph_to_api(graph: dict) -> dict:
    """Convert a ComfyUI graph-format workflow into API/prompt format."""
    links = _link_map(graph)
    api = {}
    for node in graph["nodes"]:
        if node["type"] in _NOTE_NODES:
            continue
        api[str(node["id"])] = {
            "class_type": node["type"],
            "inputs": _node_inputs(node, links),
        }
    return api


# Model-loader node classes and the input that names the model file.
_MODEL_LOADER_KEYS = {
    "CheckpointLoaderSimple": "ckpt_name",
    "UNETLoader": "unet_name",
    "SeedVR2LoadDiTModel": "model",
    "FlashVSRNode": "model",
}


def _set_node(workflow: dict, node_id, key: str, value) -> None:
    """Write ``value`` into input ``key`` of node ``node_id`` (no-op if None)."""
    if node_id is not None:
        workflow[str(node_id)]["inputs"][key] = value


def _model_key_for(node: dict, spec: dict, key_name: str = "model_key") -> str | None:
    """Input that names the model file, for a loader the spec points at.

    ``model_key`` / ``model_key_unconditional`` in the spec win, so a workflow
    using a loader class the bot does not know still needs no code change; the
    class table above is only the fallback.
    """
    return spec.get(key_name) or _MODEL_LOADER_KEYS.get(node.get("class_type"))


def _apply_model(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Point the model loader(s) at the requested file."""
    ckpt_name = kwargs.get("ckpt_name")
    if ckpt_name is not None:
        # Works with SDXL checkpoints (models/checkpoints) and with UNet/DiT
        # model files. Ideogram has two UNet loaders (conditional +
        # unconditional); the spec's model_node pins which one a
        # default_model refers to.
        model_node = spec.get("model_node")
        if model_node is not None:
            node = workflow.get(str(model_node))
            if node is None:
                raise ValueError(
                    f"Workflow has no node {model_node!r}; cannot select model {ckpt_name!r}."
                )
            key = _model_key_for(node, spec)
            if key is None:
                raise ValueError(
                    f"Node {model_node!r} is not a known model loader; "
                    "set model_key in the spec to name its model input."
                )
            node["inputs"][key] = ckpt_name
        else:
            # No pinned node: the first known loader class in the graph wins.
            for node in workflow.values():
                key = _MODEL_LOADER_KEYS.get(node.get("class_type"))
                if key is None:
                    continue
                node["inputs"][key] = ckpt_name
                break
            else:
                raise ValueError(
                    f"Workflow has no model-loader node; cannot select model {ckpt_name!r}."
                )

    # Ideogram's second UNet loader (unconditional branch) can be overridden
    # independently via the spec's default_model_unconditional / model_node_unconditional.
    default_model_unconditional = spec.get("default_model_unconditional")
    model_node_unconditional = spec.get("model_node_unconditional")
    if default_model_unconditional is not None and model_node_unconditional is not None:
        node = workflow.get(str(model_node_unconditional))
        if node is not None:
            key = _model_key_for(node, spec, "model_key_unconditional")
            if key is not None:
                node["inputs"][key] = default_model_unconditional


def _current_model(workflow: dict, spec: dict) -> str | None:
    """The model file this workflow already references (spec node first)."""
    model_node = spec.get("model_node")
    if model_node is not None:
        node = workflow.get(str(model_node))
        key = _model_key_for(node, spec) if node is not None else None
        if key and node["inputs"].get(key):
            return node["inputs"][key]
    for node in workflow.values():
        key = _MODEL_LOADER_KEYS.get(node.get("class_type"))
        if key and node["inputs"].get(key):
            return node["inputs"][key]
    return None


def _apply_text_encoder(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Point the CLIPLoader(s) at the effective text-encoder file."""
    if spec.get("text_encoder_node") is None and spec.get("enhancer_text_encoder_node") is None:
        return
    # An explicit ``text_encoder`` kwarg wins; otherwise the spec's default is
    # applied so the workflow never depends on whatever filename was saved in
    # the JSON. ``enhancer_text_encoder_node`` (when present) points at a
    # second CLIPLoader and receives the same file.
    effective_encoder = kwargs.get("text_encoder") or spec.get("default_text_encoder")
    if effective_encoder is None:
        return
    for encoder_key in ("text_encoder_node", "enhancer_text_encoder_node"):
        node_id = spec.get(encoder_key)
        if node_id is None:
            continue
        node = workflow.get(str(node_id))
        if node is not None and node.get("class_type") == "CLIPLoader":
            node["inputs"]["clip_name"] = effective_encoder


def _apply_prompt(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Write the prompt, its translation switch, the enhancer and the negative."""
    prompt = kwargs.get("prompt")
    if prompt is not None:
        input_node = spec.get("prompt_input_node")
        if input_node is not None:
            # Prompt enters through an intermediate node (e.g. Qwen's Google
            # Translate node feeding the optional prompt enhancer); the
            # encode node's prompt input is a link into that chain, so
            # writing to prompt_node would clobber the link and bypass it.
            _set_node(workflow, input_node, spec.get("prompt_input_key", "text"), prompt)
        else:
            _set_node(workflow, spec.get("prompt_node"), spec.get("prompt_key", "text"), prompt)

    # Qwen Image 2.1 T2I: optional prompt translation. The GoogleTranslateTextNode
    # input is inverted (manual_translate: True = pass through untranslated,
    # False = translate to English). Translation is off unless explicitly
    # enabled: only translate=True translates; None/False pass the prompt
    # through untranslated. The raw workflow ships manual_translate: false
    # (translate on), so the input is always rewritten when the node exists.
    _set_node(
        workflow,
        spec.get("translate_node"),
        spec.get("translate_key", "manual_translate"),
        not bool(kwargs.get("translate")),
    )

    # Qwen Image 2.1 T2I: optional prompt enhancement chain between the
    # translate node and the encoder (StringConcatenate -> TextGenerate ->
    # showAnything). Enhancement is off unless explicitly enabled: only
    # enhance=True keeps the chain; None/False remove those nodes and
    # rewire the encoder's prompt input directly to the translate node.
    enhance = kwargs.get("enhance")
    if not bool(enhance):
        for node_id in spec.get("enhance_nodes") or []:
            workflow.pop(str(node_id), None)
        prompt_node = spec.get("prompt_node")
        input_node = spec.get("prompt_input_node")
        if prompt_node is not None and input_node is not None:
            # Same input the prompt was written to above, so a spec without
            # prompt_key rewires the encoder it actually patched.
            prompt_key = spec.get("prompt_key", "text")
            workflow[str(prompt_node)]["inputs"][prompt_key] = [str(input_node), 0]
    # Flux 3 Klein T2I: the Interruptor (If/Else Switch) picks between the raw
    # prompt and the upsampler's text. Its inputs are lazy, so off skips the
    # enhancer entirely. Off unless enhance=True is passed.
    enhance_node = spec.get("enhance_node")
    if enhance_node is not None:
        workflow[str(enhance_node)]["inputs"][spec.get("enhance_key", "switch")] = bool(enhance)

    negative = kwargs.get("negative")
    if negative is not None:
        _set_node(workflow, spec.get("negative_node"), spec.get("negative_key", "text"), negative)


def _apply_sampling(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Write the sampler's inputs and the img2img denoise strength."""
    seed = kwargs.get("seed")
    if seed is not None:
        _set_node(workflow, spec.get("seed_node"), spec.get("seed_key", "seed"), int(seed))
    steps = kwargs.get("steps")
    if steps is not None:
        _set_node(workflow, spec.get("steps_node"), "steps", int(steps))
    strength = kwargs.get("strength")
    if strength is not None:
        _set_node(workflow, spec.get("denoise_node"), "denoise", float(strength))
    cfg = kwargs.get("cfg")
    if cfg is not None:
        _set_node(workflow, spec.get("cfg_node"), "cfg", float(cfg))
    sampler = kwargs.get("sampler")
    if sampler is not None:
        _set_node(workflow, spec.get("sampler_node"), "sampler_name", sampler)
    scheduler = kwargs.get("scheduler")
    if scheduler is not None:
        _set_node(workflow, spec.get("sampler_node"), "scheduler", scheduler)


def _apply_images(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Write the LoadImage filenames, dropping the second loader when unused."""
    image_filename = kwargs.get("image_filename")
    if image_filename is not None:
        _set_node(workflow, spec.get("image_node"), "image", image_filename)
    image_files = kwargs.get("image_files")
    if image_files:
        image_nodes = spec.get("image_nodes")
        if image_nodes:
            for node_id, fname in zip(image_nodes, image_files):
                workflow[str(node_id)]["inputs"]["image"] = fname
    # Optional second image for edit models that take more than one reference.
    # When omitted, the second LoadImage node is removed and every input that
    # referenced it is dropped so ComfyUI never validates an empty LoadImage
    # (qwen: image2_key on prompt_node; flux3: image2_refs auto-grow inputs).
    image2_node = spec.get("image2_node")
    if image2_node is None:
        return
    image2_filename = kwargs.get("image2_filename")
    if image2_filename:
        workflow[str(image2_node)]["inputs"]["image"] = image2_filename
        return
    workflow.pop(str(image2_node), None)
    for ref_node, ref_key in spec.get("image2_refs") or []:
        node = workflow.get(str(ref_node))
        if node is not None:
            node["inputs"].pop(ref_key, None)
    encode_node = workflow.get(str(spec.get("prompt_node")))
    if encode_node is not None:
        encode_node["inputs"].pop(spec.get("image2_key", "images.image_2"), None)


def _apply_latent(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Set the empty-latent resolution and batch size."""
    latent_id = spec.get("latent_node")
    if latent_id is None:
        return
    width = kwargs.get("width")
    height = kwargs.get("height")
    if width is not None and height is not None:
        workflow[str(latent_id)]["inputs"]["width"] = int(width)
        workflow[str(latent_id)]["inputs"]["height"] = int(height)
    batch_size = kwargs.get("batch_size")
    if batch_size is not None:
        workflow[str(latent_id)]["inputs"]["batch_size"] = int(batch_size)


def _apply_lora(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Fill the LoRA loaders and rewire the model/clip chain around them.

    An empty lora name disables the loader node. Since ComfyUI validates
    lora_name against the available-loras list ("" is not a valid value),
    disabled loaders are removed from the graph and the model/clip chain is
    rewired around them.
    """
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
        for lora_name, lora_node in (
            (lora1, spec.get("lora1_node")),
            (lora2, spec.get("lora2_node")),
        ):
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
        # UNETLoader -> LoraLoaderModelOnly -> SageAttention -> APG). The
        # effective name is the command's `lora` param or the config `lora`
        # default; empty/missing removes the node and reconnects whatever
        # consumed it to the chain source, so the attention patch in between
        # is neither bypassed nor left pointing at a missing node.
        nid = str(spec["lora1_node"])
        if lora:
            node = workflow[nid]
            node["inputs"]["lora_name"] = lora
            if lora_strength is not None:
                node["inputs"]["strength_model"] = float(lora_strength)
        else:
            workflow.pop(nid, None)  # disabled: remove so it is not validated
            chain_start = spec.get("model_chain_start")
            if chain_start is not None:
                for node in workflow.values():
                    if node["inputs"].get("model") == [nid, 0]:
                        node["inputs"]["model"] = [str(chain_start), 0]
    else:
        strength_val = float(lora_strength) if lora_strength is not None else None
        active = []
        for lora_name, lora_node in (
            (lora1, spec.get("lora1_node")),
            (lora2, spec.get("lora2_node")),
        ):
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
            ckpt_nid = next(
                (
                    str(k)
                    for k, v in workflow.items()
                    if v["class_type"] == "CheckpointLoaderSimple"
                ),
                None,
            )
            clip_nid = next(
                (str(k) for k, v in workflow.items() if v["class_type"] == "CLIP Input Switch"),
                None,
            )
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


def _apply_resolution(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Ideogram: resolution selector (megapixels + aspect ratio) and quality preset."""
    megapixels = kwargs.get("megapixels")
    aspect_ratio = kwargs.get("aspect_ratio")
    quality = kwargs.get("quality")
    resolution_node = spec.get("resolution_node")
    if megapixels is not None and resolution_node is not None:
        workflow[str(resolution_node)]["inputs"][spec.get("megapixels_key", "megapixels")] = int(
            megapixels
        )
    if megapixels is not None:
        mp_key = spec.get("megapixels_key", "megapixels")
        for node_id in spec.get("megapixels_nodes") or []:
            workflow[str(node_id)]["inputs"][mp_key] = int(megapixels)
        # Dynamic-combo nodes (e.g. ReferenceMultiLatent) also need their mode
        # selected: [node, key, value] switches it to "resize".
        combo = spec.get("megapixels_combo")
        if combo:
            workflow[str(combo[0])]["inputs"][combo[1]] = combo[2]
    if aspect_ratio is not None and resolution_node is not None:
        workflow[str(resolution_node)]["inputs"][spec.get("aspect_ratio_key", "aspect_ratio")] = (
            normalize_aspect_ratio(aspect_ratio)
        )
    if quality is not None:
        _set_node(workflow, spec.get("quality_node"), spec.get("quality_key", "choice"), quality)


def _apply_scale(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Set the upscaler's factor, or its absolute pixel target, when given."""
    scale = kwargs.get("scale")
    scale_id = spec.get("scale_node")
    if scale is None or scale_id is None:
        return
    if spec.get("scale_mode", "factor") == "resolution":
        # Target absolute pixels = longest input side * scale.
        input_longest_side = kwargs.get("input_longest_side")
        if input_longest_side is None:
            raise ValueError("Scale requires the input image dimensions.")
        value = int(input_longest_side * float(scale))
        for key in spec.get("scale_keys") or [spec.get("scale_key", "resolution")]:
            workflow[str(scale_id)]["inputs"][key] = value
    else:
        # Plain multiplicative scale factor (e.g. 2x, 3x).
        factor = float(scale)
        value = int(factor) if factor.is_integer() else factor
        workflow[str(scale_id)]["inputs"][spec.get("scale_key", "scale_by")] = value


def _apply_encode_resolution(workflow: dict, spec: dict, kwargs: dict) -> None:
    """Qwen Image 2.1 I2I: encode-node resolution budget (output area ~= R^2
    pixels). The bot computes R from the input image so the output follows
    the input size on the node's 32-px grid; without the kwarg the
    workflow's fixed default (1376 ~= 1.89 MP) is kept.
    """
    encode_resolution = kwargs.get("resolution")
    encode_node = spec.get("encode_resolution_node")
    if encode_resolution is not None and encode_node is not None:
        workflow[str(encode_node)]["inputs"]["resolution"] = int(encode_resolution)


def apply_spec(workflow: dict, spec: dict, **kwargs) -> None:
    """Patch node inputs in the workflow based on config node IDs."""
    _apply_model(workflow, spec, kwargs)
    _apply_text_encoder(workflow, spec, kwargs)
    _apply_prompt(workflow, spec, kwargs)
    _apply_sampling(workflow, spec, kwargs)
    _apply_images(workflow, spec, kwargs)
    _apply_latent(workflow, spec, kwargs)
    _apply_lora(workflow, spec, kwargs)
    _apply_resolution(workflow, spec, kwargs)
    _apply_scale(workflow, spec, kwargs)
    _apply_encode_resolution(workflow, spec, kwargs)


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
    "resolution": (32, 4096),
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
            default_ckpt = _current_model(api_workflow, spec)
        if default_ckpt is not None:
            if model_key == "sdxl":
                try:
                    available = await comfy.fetch_checkpoints()
                except Exception as exc:
                    log.warning("Could not list checkpoints for fallback: %s", exc)
                    available = []
                if available and default_ckpt not in available:
                    effective_ckpt = next(
                        (c for c in available if "sdxl" in c.lower()), available[0]
                    )
                    log.info(
                        "Default checkpoint %r not available; falling back to %r",
                        default_ckpt,
                        effective_ckpt,
                    )
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

    # Identify the sampler so live progress ignores other nodes' counters
    # (the TextGenerate prompt enhancer reports one unit per generated token
    # and would otherwise race the bar ahead of actual image sampling).
    progress_node_id = str(spec["steps_node"]) if spec.get("steps_node") is not None else None
    progress_total = meta.get("steps") if progress_node_id is not None else None

    # If this checkpoint was previously seen without a bundled text
    # encoder/VAE, use the separate loaders straight away (no error/retry).
    switch_node = spec.get("switch_node")
    if (
        switch_node is not None
        and effective_ckpt
        and user_settings.is_split_checkpoint(effective_ckpt)
    ):
        api_workflow[str(switch_node)]["inputs"]["value"] = True
        log.info("Checkpoint %r cached as split; using separate CLIP/VAE loaders", effective_ckpt)

    prompt_id, client_id = await comfy.queue_prompt(api_workflow)
    try:
        filenames = await comfy.wait_for_result(
            prompt_id,
            client_id,
            on_progress=on_progress,
            progress_node_id=progress_node_id,
            progress_total=progress_total,
        )
    except ComfyUIError:
        # If the checkpoint has no bundled text encoder/VAE, the run fails
        # (ComfyUI reports a generic "error"). Flip the switch node so the
        # separate CLIPLoader/VAELoader are used, and retry once.
        if switch_node is not None and api_workflow[str(switch_node)]["inputs"]["value"] is False:
            log.info(
                "Checkpoint %r likely lacks bundled CLIP/VAE; enabling split loaders and retrying",
                effective_ckpt,
            )
            user_settings.mark_split_checkpoint(effective_ckpt)
            api_workflow[str(switch_node)]["inputs"]["value"] = True
            prompt_id, client_id = await comfy.queue_prompt(api_workflow)
            filenames = await comfy.wait_for_result(
                prompt_id,
                client_id,
                on_progress=on_progress,
                progress_node_id=progress_node_id,
                progress_total=progress_total,
            )
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
        job = job_queue.submit(_run(), lane="comfyui", name="prompt_gen_workflow")
        try:
            return await job
        except asyncio.CancelledError:
            # The caller was cancelled (e.g. /gen_prompt cancelled by the user):
            # stop the ComfyUI half too instead of composing an unused prompt.
            job.cancel()
            raise
    return await _run()
