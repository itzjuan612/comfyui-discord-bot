import asyncio

import aiohttp
import discord
from discord import app_commands

from core import log, config, comfy, ComfyUIError
from http_session import get_session


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
    session = get_session()
    try:
        async with session.get(llm_base_url() + "/api/v1/models", headers=headers,
                                timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                data = await resp.json()
            else:
                data = None
    except Exception:
        data = None
    if data is None:
        # Endpoint not available (non-LM-Studio server) or malformed response;
        # fall back to the standard OpenAI models endpoint.
        async with session.get(llm_base_url() + "/v1/models", headers=headers,
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
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
        session = get_session()
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
        session = get_session()
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
    session = get_session()
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
    session = get_session()
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


# Global cache of the LLM model list, fetched lazily (see refresh_llm_models).
# "models" holds the ids; "fetched_at" is the loop time of the last successful fetch.
llm_model_cache: dict = {"models": [], "fetched_at": None}

# How long a cached model list is considered fresh before re-fetching.
LLM_MODEL_TTL_SECONDS = 300.0


async def refresh_llm_models(force: bool = False) -> list[str]:
    """Fetch the LLM model list into the global cache.

    Called once at startup (from ``on_ready``, as a background task) and
    periodically by ``llm_model_refresh_loop``. Failures keep the previous
    cache so a transient endpoint outage never breaks slash-command
    autocomplete.
    """
    global llm_model_cache
    now = asyncio.get_running_loop().time()
    if not force and llm_model_cache["fetched_at"] is not None:
        if now - llm_model_cache["fetched_at"] < LLM_MODEL_TTL_SECONDS:
            return llm_model_cache["models"]
    try:
        models = await fetch_llm_models()
    except Exception as exc:
        log.warning("Could not refresh LLM model list: %s", exc)
        return llm_model_cache["models"]
    llm_model_cache = {"models": models, "fetched_at": now}
    log.info("LLM model list refreshed: %d models", len(models))
    return models


async def get_llm_models() -> list[str]:
    """Return cached model ids, fetching them first if the cache is stale."""
    now = asyncio.get_running_loop().time()
    if llm_model_cache["fetched_at"] is None or now - llm_model_cache["fetched_at"] >= LLM_MODEL_TTL_SECONDS:
        await refresh_llm_models()
    return llm_model_cache["models"]


async def llm_model_refresh_loop() -> None:
    """Refresh the cached model list every LLM_MODEL_TTL_SECONDS."""
    while True:
        await asyncio.sleep(LLM_MODEL_TTL_SECONDS)
        await refresh_llm_models(force=True)


async def llm_model_autocomplete(interaction: discord.Interaction, current_input: str):
    """Dynamic autocomplete for /gen_prompt's ``model`` parameter.

    Suggests LLM model ids from the cached list (fetched lazily and refreshed
    on a schedule), filtered by what the user has typed so far.
    """
    try:
        models = await get_llm_models()
    except Exception as exc:
        log.warning("Could not fetch LLM models for autocomplete: %s", exc)
        return []
    if current_input:
        matches = [m for m in models if current_input.lower() in m.lower()]
    else:
        matches = models
    # Discord allows at most 25 autocomplete choices.
    return [app_commands.Choice(name=m, value=m) for m in matches[:25]]


async def _sdxl_lora_autocomplete(interaction: discord.Interaction, current_input: str):
    """Dynamic autocomplete for /sdxl's ``lora1``/``lora2`` parameters.

    Suggests LoRA files from ComfyUI's models/loras folder, filtered by what
    the user has typed so far.
    """
    try:
        loras = await comfy.fetch_loras()
    except Exception as exc:
        log.warning("Could not fetch LoRA list for autocomplete: %s", exc)
        return []
    if current_input:
        matches = [l for l in loras if current_input.lower() in l.lower()]
    else:
        matches = loras
    # Discord allows at most 25 autocomplete choices.
    return [app_commands.Choice(name=l, value=l) for l in matches[:25]]


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
