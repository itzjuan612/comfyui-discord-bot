import json
import os
import random

from core import config, comfy, log, normalize_aspect_ratio, last_workflow_by_model, last_ckpt_by_model


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
        set_node(spec.get("prompt_node"), spec.get("prompt_key", "text"), prompt)
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


async def run_image(spec: dict, on_progress=None, **kwargs):
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

    prompt_id, client_id = await comfy.queue_prompt(api_workflow)
    filenames = await comfy.wait_for_result(prompt_id, client_id, on_progress=on_progress)
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
    prompt_id, _client_id = await comfy.queue_prompt(workflow)
    return await comfy.wait_for_output_text(prompt_id, target_node=target_node)
