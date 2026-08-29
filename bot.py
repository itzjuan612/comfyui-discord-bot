import asyncio
import io
import json
import logging
import os
import random
import subprocess
import sys

import aiohttp
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
SAMPLER_NAMES = [
    "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_heun",
    "dpm_2", "dpm_3", "lms", "heun", "ddpm", "ddim",
]
SAMPLER_CHOICES = [
    app_commands.Choice(name=s, value=s)
    for s in SAMPLER_NAMES
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

# --- LLM endpoint (configured in config.yaml: llm.address / llm.port) ---

def llm_base_url() -> str:
    """Build the OpenAI-compatible API base URL from config.yaml.

    Two ways to configure the endpoint:
    - ``url``: a full base URL used verbatim, e.g. ``https://api.openai.com``.
    - ``address`` + optional ``port``: builds ``<scheme>://address[:port]``,
      where ``scheme`` defaults to ``http`` (set it to ``https`` for endpoints
      like OpenAI that require HTTPS; port can be omitted in that case).
    """
    llm_cfg = config.get("llm", {})
    url = str(llm_cfg.get("url") or "").strip()
    if url:
        return url.rstrip("/")
    address = str(llm_cfg.get("address") or "").strip().split("://")[-1]
    if not address:
        raise ComfyUIError("LLM endpoint not configured in config.yaml (set llm.url or llm.address)")
    scheme = str(llm_cfg.get("scheme") or "http")
    port = llm_cfg.get("port")
    if port not in (None, ""):
        return f"{scheme}://{address}:{port}"
    return f"{scheme}://{address}"


async def fetch_llm_models() -> list[str]:
    """Fetch the list of model ids from the LLM endpoint.

    Handles both response shapes:
    - LM Studio ``/api/v1/models``: ``{"models": [{"type": ..., "key": ...}]}``
      (lists all models, loaded or not).
    - OpenAI-compatible ``/v1/models``: ``{"data": [{"id": ...}]}``.
    """
    llm_cfg = config.get("llm", {})
    headers = {}
    token = llm_cfg.get("api_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(llm_base_url() + "/api/v1/models", headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                else:
                    data = None
        except Exception:
            data = None
    if data is None:
        # Endpoint not available (non-LM-Studio server) or malformed response;
        # fall back to the standard OpenAI models endpoint.
        async with aiohttp.ClientSession() as session:
            async with session.get(llm_base_url() + "/v1/models", headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
    if "models" in data:
        # LM Studio shape: use "key" as the id, skip non-chat (embedding) models.
        return [
            m["key"]
            for m in data["models"]
            if m.get("type", "llm") == "llm"
        ]
    # OpenAI shape.
    return [m["id"] for m in data.get("data", [])]


async def llm_model_load(model: str) -> bool:
    """Request the LLM server to load a model (LM Studio: /api/v1/models/load).

    Returns True if the server accepted the load, False if the endpoint is
    unsupported or the request failed. Failures are logged but never raised,
    so non-LM-Studio servers keep working with plain OpenAI-compatible APIs.
    """
    llm_cfg = config.get("llm", {})
    headers = {}
    token = llm_cfg.get("api_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = llm_base_url() + "/api/v1/models/load"
    try:
        async with aiohttp.ClientSession() as session:
            # LM Studio expects {"model": <key>} (not "id").
            async with session.post(url, json={"model": model}, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status in (404, 405):
                    log.info("LLM endpoint has no /api/v1/models/load; skipping load for %r", model)
                    return False
                resp.raise_for_status()
        log.info("Loaded LLM model %r", model)
        return True
    except Exception as exc:
        log.warning("Could not load LLM model %r: %s", model, exc)
        return False


async def llm_model_unload(model: str) -> bool:
    """Request the LLM server to unload a model (LM Studio: /api/v1/models/unload).

    Like ``llm_model_load``, failures are swallowed so servers without these
    endpoints (or transient errors) don't break the request flow.
    """
    llm_cfg = config.get("llm", {})
    headers = {}
    token = llm_cfg.get("api_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = llm_base_url() + "/api/v1/models/unload"
    try:
        async with aiohttp.ClientSession() as session:
            # LM Studio expects {"instance_id": <key>} (not "id").
            async with session.post(url, json={"instance_id": model}, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status in (404, 405):
                    log.info("LLM endpoint has no /api/v1/models/unload; skipping unload for %r", model)
                    return False
                resp.raise_for_status()
        log.info("Unloaded LLM model %r", model)
        return True
    except Exception as exc:
        log.warning("Could not unload LLM model %r: %s", model, exc)
        return False


async def call_llm(prompt: str, model: str, max_tokens: int | None = None,
                   temperature: float | None = None, reasoning_effort: str | None = None) -> str:
    """Call the configured OpenAI-compatible endpoint directly from the bot.

    This bypasses the ComfyUI LLM node entirely, so the request timeout is
    controlled here (``llm.timeout`` in config.yaml) instead of whatever is
    hardcoded inside the custom node. The ``prompt`` argument is the fully
    composed message produced by the template workflow.

    ``reasoning_effort`` is one of ``low`` / ``medium`` / ``high`` / ``xhigh``
    (or ``none`` to send no reasoning effort). Endpoints that don't support
    the field simply ignore it.
    """
    llm_cfg = config.get("llm", {})
    timeout = float(llm_cfg.get("timeout", 300.0))
    headers = {"Content-Type": "application/json"}
    token = llm_cfg.get("api_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort
    url = llm_base_url() + "/v1/chat/completions"
    log.debug("call_llm payload: %s", payload)
    async with aiohttp.ClientSession() as session:
        for _attempt in range(2):
            async with session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                data = await resp.json()
                if resp.status == 400 and "reasoning_effort" in payload:
                    # Endpoint rejected the reasoning effort; retry once
                    # without it so the model falls back to its native default.
                    log.info("reasoning_effort rejected by endpoint; retrying without it")
                    payload.pop("reasoning_effort")
                    continue
                log.debug("call_llm usage: %s", data.get("usage"))
                if resp.status >= 400:
                    raise ComfyUIError(f"LLM request failed (HTTP {resp.status}): {data}")
                choices = data.get("choices")
                if not choices:
                    raise ComfyUIError("LLM endpoint returned no choices")
                return choices[0]["message"]["content"]


REASONING_EFFORT_CANDIDATES = ("low", "medium", "high", "xhigh", "on", "off")
reasoning_effort_cache: dict[str, list[str]] = {}


async def probe_reasoning_efforts(model: str) -> list[str]:
    """Discover which ``reasoning_effort`` values this model's endpoint accepts.

    For each candidate, sends a minimal request (``Say ok.`` with
    ``max_tokens=1``) and keeps the values the server accepts. Rejected
    values return instantly with a 400, so they cost no tokens. The result
    is cached per model, so probes only happen once per model per restart.
    """
    if model in reasoning_effort_cache:
        return reasoning_effort_cache[model]
    llm_cfg = config.get("llm", {})
    headers = {"Content-Type": "application/json"}
    token = llm_cfg.get("api_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = llm_base_url() + "/v1/chat/completions"
    supported: list[str] = []
    async with aiohttp.ClientSession() as session:
        for effort in REASONING_EFFORT_CANDIDATES:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Say ok."}],
                "max_tokens": 1,
                "reasoning_effort": effort,
            }
            try:
                async with session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    log.debug("Probe reasoning_effort=%r for %r: HTTP %s", effort, model, resp.status)
                    if resp.status >= 400:
                        await resp.release()
                        continue
                    await resp.release()
                    supported.append(effort)
            except Exception as exc:
                log.warning("Probe reasoning_effort=%r failed: %s", effort, exc)
                continue
    reasoning_effort_cache[model] = supported
    log.info("Reasoning effort probe: model %r supports %s", model, supported or "(none)")
    return supported


async def resolve_reasoning_effort(model: str, thinking: str | None, llm_cfg: dict) -> str | None:
    """Return the ``reasoning_effort`` value to send for this model.

    ``thinking`` is either one of the endpoint's probed values, or
    ``None`` / ``"api_default"`` to send no reasoning_effort tag at all.
    If the value isn't in the model's probed set, falls back to API default
    (``None``) so the endpoint never rejects the request with a 400.
    """
    if thinking in (None, "api_default"):
        return None
    if not model:
        return thinking
    supported = await probe_reasoning_efforts(model)
    if thinking in supported:
        return thinking
    return None


def _llm_model_choices() -> list[app_commands.Choice]:
    """Best-effort model autocomplete list; empty if the endpoint is unreachable."""
    try:
        models = asyncio.run(fetch_llm_models())
    except Exception as exc:
        log.warning("Could not fetch LLM model list at startup: %s", exc)
        models = []
    return [app_commands.Choice(name=m, value=m) for m in models]


LLM_MODEL_CHOICES = _llm_model_choices()


async def _sdxl_model_autocomplete(interaction: discord.Interaction, current_input: str):
    """Dynamic autocomplete for /sdxl's ``model`` parameter.

    Suggests checkpoint files from ComfyUI's models/checkpoints folder,
    filtered by what the user has typed so far. The list comes from the
    cached ``fetch_checkpoints()`` (60s TTL), so newly added checkpoint
    files appear without a bot restart or slash-command re-sync.
    """
    try:
        checkpoints = await comfy.fetch_checkpoints()
    except Exception as exc:
        log.warning("Could not fetch checkpoint list for autocomplete: %s", exc)
        return []
    if current_input:
        matches = [c for c in checkpoints if current_input.lower() in c.lower()]
    else:
        matches = checkpoints
    # Discord allows at most 25 autocomplete choices.
    return [app_commands.Choice(name=c, value=c) for c in matches[:25]]


bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

# Per-user cooldown to prevent abuse of the GPU.
cooldowns: dict[str, float] = {}
MIN_INTERVAL_SECONDS = 20

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
    """

    def __init__(self, message):
        # ``message`` is the original response message (an InteractionMessage)
        # returned by interaction.original_response(); it is edited in place
        # to show progress.
        self.message = message
        self.level = 0
        self.done = False

    def update(self, progress: float) -> None:
        if self.done:
            return
        pct = max(0.0, min(100.0, progress * 100))
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
                content=f"\U0001f3a8 Generating image\u2026 [{progress_bar(level)}] {int(pct)}%"
            )
        except Exception as exc:
            log.warning("progress edit failed: %s", exc)

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
            async with aiohttp.ClientSession() as session:
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
            async with aiohttp.ClientSession() as session:
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
    last = cooldowns.get(uid, 0)
    if now - last < MIN_INTERVAL_SECONDS:
        return False
    cooldowns[uid] = now
    return True


async def run_t2i_generation(interaction: discord.Interaction, model: str,
                             prompt: str, stealth: bool, gen_kwargs: dict):
    """Shared generation pipeline for the /ideogram and /sdxl commands.

    ``gen_kwargs`` carries the model-specific parameters (plus prompt/seed).
    The embed description and saved retry parameters are built from
    ``gen_kwargs`` so each command exposes only its own settings.
    """
    spec = config["models"][model]["t2i"]
    await interaction.response.send_message(
        content="\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
        ephemeral=stealth,
    )
    progress_msg = await interaction.original_response()
    progress = ProgressUpdater(progress_msg)
    log.info("%s: prompt=%r", model, prompt)
    try:
        images, meta = await run_image(
            spec, on_progress=progress.update,
            model_key=model, **gen_kwargs,
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
            "kwargs": {**gen_kwargs, "seed": None},
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
@app_commands.describe(seed="Seed (optional)")
@app_commands.describe(steps="Sampling steps")
@app_commands.describe(width="Width in pixels, multiple of 64")
@app_commands.describe(height="Height in pixels, multiple of 64")
@app_commands.describe(cfg="CFG guidance scale")
@app_commands.describe(stealth="Ephemeral output, visible only to you")
async def sdxl(interaction: discord.Interaction, prompt: str,
                model: str | None = None,
                negative: str | None = None,
                seed: int | None = None,
                steps: int | None = None,
                width: int | None = None,
                height: int | None = None,
                cfg: float | None = None,
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
        steps = settings["steps"]
    if cfg is None:
        cfg = settings["cfg"]

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
        "cfg": cfg,
        "ckpt_name": model,
    }
    await run_t2i_generation(interaction, "sdxl", prompt, stealth, gen_kwargs)


@bot.tree.command(name="gen_prompt", description="Convert a natural-language prompt into an Ideogram 4 JSON caption prompt")
@app_commands.choices(aspect_ratio=ASPECT_RATIO_CHOICES)
@app_commands.choices(model=LLM_MODEL_CHOICES)
@app_commands.describe(prompt="Natural-language prompt to convert")
@app_commands.describe(megapixels="Target resolution in megapixels")
@app_commands.describe(model="LLM model to use (see /llm_models)")
@app_commands.describe(max_tokens="Max tokens for the LLM (optional)")
@app_commands.describe(temperature="LLM temperature, 0-1 (optional)")
async def gen_prompt(interaction: discord.Interaction, prompt: str, megapixels: int, aspect_ratio: str,
                     model: str | None = None, max_tokens: int | None = None,
                     temperature: float | None = None):
    if await ban_guard(interaction):
        return
    if not await check_cooldown(interaction):
        await interaction.response.send_message(
            content="\u23f3 Please wait before requesting another prompt.", ephemeral=True
        )
        return

    if nsfw_blocked(interaction, prompt):
        await interaction.response.send_message(
            content="\u26a0\ufe0f That prompt appears to be NSFW. Please run it in an NSFW channel.", ephemeral=True
        )
        msg = await interaction.original_response()
        schedule_message_deletion(msg)
        return

    await interaction.response.defer(ephemeral=True)
    msg = await interaction.original_response()
    llm_cfg = config.get("llm", {})
    chosen_model = model or llm_cfg.get("default_model") or ""
    if not chosen_model:
        await reply_error(interaction, "\u274c No model selected and no default model configured.", target=msg)
        return
    log.info("gen_prompt: prompt=%r megapixels=%s aspect_ratio=%s model=%r", prompt, megapixels, aspect_ratio, chosen_model)
    try:
        # Load the model and probe which reasoning-effort values the endpoint
        # accepts; only those become selectable options in the view.
        await llm_model_load(chosen_model)
        supported = await probe_reasoning_efforts(chosen_model)
        view = ThinkingView(
            chosen_model, supported, prompt, megapixels, aspect_ratio,
            max_tokens, temperature, llm_cfg,
        )
        await msg.edit(
            content="\U0001f9e0 Select the reasoning effort for " + chosen_model + ":",
            view=view,
        )
    except Exception as exc:
        log.exception("gen_prompt setup failed")
        await reply_error(interaction, f"\u274c Prompt generation failed: {exc}", target=msg)


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

@bot.tree.command(name="llm_models", description="List the LLM models available on the configured endpoint")
async def llm_models(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    msg = await interaction.original_response()
    try:
        models = await fetch_llm_models()
        await msg.edit(
            content="\U0001f916 Available LLM models:\n\n" + "\n".join(f"\u2022 {m}" for m in models)
        )
    except Exception as exc:
        log.warning("llm_models failed: %s", exc)
        await reply_error(interaction, f"\u274c Could not list models: {exc}", target=msg)


@bot.tree.command(name="upscale", description="Upscale an image from your gallery")
@app_commands.choices(model=UPSCALE_CHOICES)
@app_commands.describe(image="Image to upscale (attach from your gallery)")
@app_commands.describe(prompt="Prompt to guide the upscale (optional)")
@app_commands.describe(negative="Negative prompt (optional)")
@app_commands.describe(strength="Denoise strength, 0-1 (optional)")
@app_commands.describe(scale="Upscale factor (optional, e.g. 2 or 3)")
@app_commands.describe(stealth="Ephemeral output, visible only to you")
async def upscale(interaction: discord.Interaction, model: str, image: discord.Attachment,
                  prompt: str | None = None,
                  negative: str | None = None, strength: float | None = None,
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
        async with aiohttp.ClientSession() as session:
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
                ),
            )
            return

        await interaction.response.send_message(
            content="\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
            ephemeral=stealth,
        )
        progress_msg = await interaction.original_response()
        progress = ProgressUpdater(progress_msg)
        images, meta = await run_image(
            spec, on_progress=progress.update, model_key=model, prompt=prompt,
            negative=negative, strength=strength, image_filename=uploaded_name,
            scale=scale, input_longest_side=input_longest_side,
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
        content="\U0001f3a8 Generating image\u2026 [\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%",
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
        async with aiohttp.ClientSession() as session:
            async with session.get(str(image.url)) as resp:
                resp.raise_for_status()
                data1 = await resp.read()
        uploaded1 = await comfy.upload_image(data1, f"discord_{uuid_hex()}.png")
        uploaded_files = [uploaded1]
        if need_two:
            async with aiohttp.ClientSession() as session:
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
        images, meta = await run_image(
            spec, on_progress=progress.update, model_key=model, **gen_kwargs
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


def uuid_hex() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


DISCORD_IMAGE_LIMIT = 18 * 1024 * 1024  # Stay comfortably under Discord's ~20 MB upload limit


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
    return lines


@bot.tree.command(name="settings", description="View or set your personal generation defaults")
@app_commands.describe(positive_prompt="Default positive prompt")
@app_commands.describe(negative_prompt="Default negative prompt")
@app_commands.describe(cfg="Default CFG guidance scale (SDXL only)")
@app_commands.describe(steps="Default sampling steps (SDXL only)")
@app_commands.choices(ideogram_quality=QUALITY_CHOICES)
@app_commands.choices(ideogram_aspect_ratio=ASPECT_RATIO_CHOICES)
@app_commands.describe(ideogram_quality="Default Ideogram quality preset (Turbo / Default / Quality)")
@app_commands.describe(ideogram_megapixels="Default Ideogram resolution in megapixels")
@app_commands.describe(ideogram_aspect_ratio="Default Ideogram aspect ratio preset")
@app_commands.choices(img2img_sampler=SAMPLER_CHOICES)
@app_commands.describe(img2img_cfg="Default img2img CFG guidance scale")
@app_commands.describe(img2img_steps="Default img2img sampling steps")
@app_commands.describe(img2img_sampler="Default img2img sampler")
@app_commands.describe(img2img_megapixels="Default img2img resolution in megapixels")
@app_commands.describe(stealth="Default privacy (ephemeral) for all your generations")
@app_commands.describe(sdxl_checkpoint="Default SDXL checkpoint (models/checkpoints), used only by /sdxl")
@app_commands.describe(view="Set to true to only view your current settings")
async def settings(interaction: discord.Interaction,
                    positive_prompt: str | None = None,
                    negative_prompt: str | None = None,
                    cfg: float | None = None,
                    steps: int | None = None,
                    ideogram_quality: str | None = None,
                    ideogram_megapixels: int | None = None,
                    ideogram_aspect_ratio: str | None = None,
                    img2img_cfg: float | None = None,
                    img2img_steps: int | None = None,
                    img2img_sampler: str | None = None,
                    img2img_megapixels: int | None = None,
                    stealth: bool | None = None,
                    sdxl_checkpoint: str | None = None,
                    view: bool = False):
    user_id = interaction.user.id
    await interaction.response.defer(ephemeral=True)

    if view or not (
        positive_prompt or negative_prompt or cfg is not None or steps is not None
        or ideogram_quality is not None or ideogram_megapixels is not None
        or ideogram_aspect_ratio is not None
        or img2img_cfg is not None or img2img_steps is not None
        or img2img_sampler is not None or img2img_megapixels is not None
        or stealth is not None
        or sdxl_checkpoint is not None
    ):
        s = user_settings.get_settings(user_id)
        await interaction.edit_original_response(
            content=f"**Your saved defaults:**\n\n{user_settings.format_settings(s)}\n\n"
                     "Tip: run `/settings` with parameters to update them, or `/reset_settings` to clear them."
        )
        return

    updates = {
        k: v for k, v in {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "cfg": cfg,
            "steps": steps,
            "ideogram_quality": ideogram_quality,
            "ideogram_megapixels": ideogram_megapixels,
            "ideogram_aspect_ratio": ideogram_aspect_ratio,
            "img2img_cfg": img2img_cfg,
            "img2img_steps": img2img_steps,
            "img2img_sampler": img2img_sampler,
            "img2img_megapixels": img2img_megapixels,
            "stealth": stealth,
            "sdxl_checkpoint": sdxl_checkpoint,
        }.items() if v is not None
    }
    s = user_settings.set_settings(user_id, **updates)
    await interaction.edit_original_response(
        content=f"\u2705 Your defaults are now:\n\n{user_settings.format_settings(s)}"
    )


@bot.tree.command(name="reset_settings", description="Clear your personal generation defaults")
async def reset_settings(interaction: discord.Interaction):
    await interaction.response.defer()
    user_settings.reset_settings(interaction.user.id)
    await interaction.edit_original_response(content="\u2705 Your saved defaults have been cleared.")


@bot.tree.command(name="flush", description="Unload all models and execution cache from ComfyUI")
async def flush(interaction: discord.Interaction):
    await interaction.response.defer()
    log = logging.getLogger("bot")
    log.info("flush: freeing ComfyUI memory")
    try:
        await comfy.free_memory()
        await interaction.edit_original_response(content="\U0001f9f9 Done. All models and execution cache have been unloaded from ComfyUI.")
    except ComfyUIError as exc:
        await reply_error(interaction, f"\u274c Flush failed: {exc}")


@bot.event
async def on_ready():
    await bot.tree.sync()
    # Register the persistent GenerationView inside the running event loop.
    # (Calling bot.add_view() before bot.run() creates the view outside the
    # loop, so its internal "stopped" future is None and it silently drops
    # every button click -> buttons appear to time out.)
    persistent_view = GenerationView()
    bot.add_view(persistent_view)
    log.info(
        "Registered persistent GenerationView: persistent=%s, dispatchable=%s",
        persistent_view.is_persistent(), persistent_view.is_dispatchable(),
    )
    print(f"Bot ready. T2I models: {T2I_MODELS}, Upscale models: {UPSCALE_MODELS}, I2I models: {I2I_MODELS}")


# --- Administration ---

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


def main():
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        token = config.get("discord", {}).get("token")
    if not token:
        raise SystemExit(
            "No Discord token found. Set the DISCORD_TOKEN env var or edit "
            "config.yaml (the bot auto-generated it with empty values)."
        )
    user_settings.init_db()
    generation_store.init_db()
    moderation.init_db()
    # Seed the owner as an admin so they appear in admin lists and can
    # manage other admins. The owner check in can_manage() still works
    # independently, but this keeps the admins table consistent.
    if BOT_OWNER_ID:
        moderation.promote(BOT_OWNER_ID)
        log.info("Owner %s seeded as admin", BOT_OWNER_ID)
    # NOTE: the persistent GenerationView is registered inside on_ready()
    # (where an event loop is running), not here, so it can actually dispatch
    # button clicks. Registering it here, before bot.run(), would leave its
    # internal "stopped" future as None and drop every click.
    bot.run(token)


if __name__ == "__main__":
    main()
