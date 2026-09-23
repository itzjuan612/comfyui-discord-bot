import asyncio
import json
import logging
import time
import uuid

import aiohttp

from http_session import get_session

log = logging.getLogger("comfyui_client")


def _extract_progress(
    payload: dict,
    progress_node_id: str | None = None,
    progress_total: float | None = None,
) -> float | None:
    """Return the sampler's 0..1 progress from a ComfyUI progress payload.

    Only the sampler node drives the bar. Other nodes emit progress too —
    notably TextGenerate, which reports one unit per generated token for the
    prompt enhancer — and must be ignored, otherwise the bar races to a high
    percentage before image sampling even begins.

    ``progress_node_id`` is the sampler's node id (spec ``steps_node``) and
    ``progress_total`` its step count, used as a fallback when a payload
    reports a different (subgraph-expanded) node id.

    Returns None when the payload carries no usable sampler progress; the
    caller should then hold the bar steady.
    """
    nodes = payload.get("nodes")
    if isinstance(nodes, dict):
        if progress_node_id is not None:
            node_data = nodes.get(progress_node_id)
            if not (isinstance(node_data, dict) and node_data.get("state") == "running"):
                # Subgraph expansion can rename execution ids; fall back to
                # the running node whose max equals the sampler's step count.
                node_data = None
                if progress_total is not None:
                    for other in nodes.values():
                        if (
                            isinstance(other, dict)
                            and other.get("state") == "running"
                            and float(other.get("max", 0) or 0) == float(progress_total)
                        ):
                            node_data = other
                            break
                if node_data is None:
                    return None
            maximum = float(node_data["max"]) if node_data.get("max") is not None else 1.0
            value = float(node_data["value"]) if node_data.get("value") is not None else 0.0
            if maximum <= 0:
                return None
            return value / maximum
        # Sampler id unknown: pick the running node with the largest max
        # (the sampling node that drives generation time).
        best = None
        for node_data in nodes.values():
            if not isinstance(node_data, dict) or node_data.get("state") != "running":
                continue
            if best is None or node_data.get("max", 0) > best.get("max", 0):
                best = node_data
        if best is None:
            return None
        maximum = float(best["max"]) if best.get("max") is not None else 1.0
        value = float(best["value"]) if best.get("value") is not None else 0.0
        if maximum <= 0:
            return None
        return value / maximum
    if "value" in payload:
        if payload["value"] is None:
            return None
        value = float(payload["value"])
        maximum = float(payload["max"]) if payload.get("max") is not None else 100.0
        if maximum <= 0:
            return None
        if progress_node_id is not None:
            node_at = payload.get("node")
            if node_at is not None and str(node_at) != progress_node_id:
                # Not the sampler (e.g. TextGenerate tokens). Allow when the
                # max still matches the sampler's step count: subgraph
                # expansion may report a different id for the same node.
                if progress_total is None or maximum != float(progress_total):
                    return None
            elif node_at is None and progress_total is not None and maximum != float(progress_total):
                return None
        return value / maximum
    return None


class ComfyUIError(Exception):
    pass


class ComfyUIClient:
    """Async client for the ComfyUI HTTP API."""

    # How long a model listing stays fresh before re-querying ComfyUI.
    MODEL_LIST_CACHE_TTL = 60.0

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # folder -> (names, monotonic fetch time) for /models/{folder} listings.
        self._list_cache: dict[str, tuple[list[str], float]] = {}

    async def queue_prompt(self, workflow: dict) -> tuple[str, str]:
        """Queue a workflow (dict of nodes).

        Returns ``(prompt_id, client_id)``. A fresh client_id is generated
        for every prompt so progress tracking never collides with lingering
        WebSocket connections from previous generations.
        """
        client_id = uuid.uuid4().hex
        payload = {"prompt": workflow, "client_id": client_id}
        session = get_session()
        async with session.post(f"{self.base_url}/prompt", json=payload) as resp:
            data = await resp.json()
            if resp.status != 200 or "error" in data:
                raise ComfyUIError(data.get("error", f"HTTP {resp.status}"))
            return data["prompt_id"], client_id

    def _ws_url(self) -> str:
        """WebSocket URL derived from the HTTP base URL."""
        return self.base_url.replace("https://", "wss://").replace("http://", "ws://")

    async def wait_for_result(
        self,
        prompt_id: str,
        client_id: str,
        timeout: float = 300.0,
        on_progress=None,
        progress_node_id: str | None = None,
        progress_total: float | None = None,
    ) -> list[str]:
        """Poll /history until the prompt finishes; returns output filenames.

        If ``on_progress`` is given, a WebSocket connection is opened and
        ComfyUI's live progress values (0.0 - 1.0) are forwarded to it while
        the prompt executes. ``progress_node_id``/``progress_total`` identify
        the sampler node so progress reported by other nodes (e.g. the
        TextGenerate prompt enhancer's token counter) is ignored.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        seen = False
        finished = False

        async def watch_progress() -> None:
            # Keep reconnecting until the prompt finishes. A single dropped
            # or failed connection must not permanently disable progress:
            # on reconnect, ComfyUI re-sends the current progress state, so
            # we pick up where we left off.
            while not seen and not finished:
                ws = None
                try:
                    session = get_session()
                    ws = await session.ws_connect(
                        self._ws_url() + "/ws?clientId=" + client_id
                    )
                    async for msg in ws:
                        if seen or finished:
                            break
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            data = json.loads(msg.data)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        payload = data.get("data", data)
                        if payload.get("prompt_id") != prompt_id:
                            continue
                        _handle_progress(payload, on_progress)
                except asyncio.CancelledError:
                    # Close the socket with a proper WebSocket close frame so
                    # ComfyUI's server sees a clean shutdown instead of a
                    # forced TCP reset (ConnectionResetError).
                    if ws is not None:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                    raise
                except Exception as exc:
                    if ws is not None:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                    log.warning("progress websocket error: %s; reconnecting", exc)
                    await asyncio.sleep(0.5)

        def _handle_progress(payload, on_progress):
            fraction = _extract_progress(payload, progress_node_id, progress_total)
            if fraction is not None:
                on_progress(fraction)

        async def poll_history() -> list[str]:
            nonlocal seen
            while loop.time() < deadline:
                session = get_session()
                async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                    data = await resp.json()
                entry = data.get(prompt_id)
                if entry:
                    status = entry.get("status", {})
                    # ComfyUI >= 0.33 uses "status_str"; older versions used "status_name".
                    status_name = status.get("status_str") or status.get("status_name") or ""
                    if status_name == "error":
                        msg = status.get("status_message") or status.get("status_str")
                        raise ComfyUIError(msg or "ComfyUI prompt error")
                    if status_name == "success":
                        seen = True
                        images = []
                        for node_id, outputs in entry.get("outputs", {}).items():
                            for img in outputs.get("images", []):
                                # Some versions tag with "image_type", newer ones with "type".
                                kind = img.get("image_type", img.get("type", "output"))
                                if kind == "output":
                                    images.append(img["filename"])
                        if images:
                            log.info("prompt %s finished with %d images", prompt_id, len(images))
                            return images
                await asyncio.sleep(1.0)

            if seen:
                # Prompt finished but produced no "output" images (e.g. workflow has
                # no SaveImage / preview node). Fail fast instead of polling to timeout.
                raise ComfyUIError("Prompt completed but returned no images")
            raise ComfyUIError(f"Timed out after {timeout}s waiting for prompt {prompt_id}")

        progress_task = asyncio.ensure_future(watch_progress()) if on_progress else None
        try:
            return await poll_history()
        finally:
            if progress_task is not None:
                finished = True
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

    async def _fetch_model_list(self, folder: str, force: bool = False) -> list[str]:
        """List files in a ComfyUI models folder, with a TTL cache.

        The result is cached for ``MODEL_LIST_CACHE_TTL`` seconds so repeated
        calls (autocomplete keystrokes, availability checks, fallbacks) don't
        hammer ComfyUI. Pass ``force=True`` to bypass the cache and refresh
        immediately.
        """
        now = time.monotonic()
        cached = self._list_cache.get(folder)
        if not force and cached is not None and now - cached[1] < self.MODEL_LIST_CACHE_TTL:
            return cached[0]
        session = get_session()
        async with session.get(f"{self.base_url}/models/{folder}") as resp:
            if resp.status != 200:
                raise ComfyUIError(f"Could not list {folder} (HTTP {resp.status})")
            data = await resp.json()
        # ComfyUI's GET /models/{folder} returns a bare JSON array of plain
        # filename strings (e.g. ["SDXL.safetensors", "flux.safetensors"]).
        # Tolerate a dict wrapper and dict items for forward compatibility.
        items = data.get(folder) if isinstance(data, dict) else data
        names = []
        for item in items or []:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and "name" in item:
                names.append(item["name"])
        self._list_cache[folder] = (names, time.monotonic())
        return names

    async def fetch_loras(self, force: bool = False) -> list[str]:
        """List LoRA files available in ComfyUI's models/loras folder."""
        return await self._fetch_model_list("loras", force=force)

    async def fetch_diffusion_models(self, force: bool = False) -> list[str]:
        """List model files available in ComfyUI's models/diffusion_models folder.

        Unlike checkpoints, diffusion models (UNet/DiT files such as the
        Z-Image turbo model) live in their own folder, so a dedicated fetcher
        is needed for autocomplete and availability checks.
        """
        return await self._fetch_model_list("diffusion_models", force=force)

    async def fetch_checkpoints(self, force: bool = False) -> list[str]:
        """List checkpoint files available in ComfyUI's models/checkpoints folder."""
        return await self._fetch_model_list("checkpoints", force=force)

    async def free_memory(self) -> None:
        """Ask ComfyUI to unload all loaded models, freeing VRAM and RAM.

        ComfyUI exposes /free for this. Recent builds accept POST; older
        builds only accept GET, so fall back gracefully.
        """
        session = get_session()
        try:
            # ComfyUI's /free endpoint requires JSON body flags:
            #   unload_models -> unload all loaded models
            #   free_memory -> reset execution cache + gc
            # Both default to false, so we must send them explicitly.
            async with session.post(
                f"{self.base_url}/free",
                json={"unload_models": True, "free_memory": True},
            ) as resp:
                if resp.status in (404, 405):
                    # Older ComfyUI: GET only.
                    async with session.get(f"{self.base_url}/free") as resp2:
                        if resp2.status >= 400:
                            raise ComfyUIError(f"Could not free memory (HTTP {resp2.status})")
                elif resp.status >= 400:
                    raise ComfyUIError(f"Could not free memory (HTTP {resp.status})")
        except aiohttp.ClientError:
            raise ComfyUIError("Could not free memory (connection error)")
        log.info("ComfyUI memory freed")

    async def wait_for_output_text(self, prompt_id: str, timeout: float = 300.0, target_node: str | None = None) -> str:
        """Poll /history until the prompt finishes; returns the text output.

        If ``target_node`` is given, only that node's output is considered,
        avoiding numeric outputs from intermediate nodes (e.g. ComfyMathExpression).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        session = get_session()
        while loop.time() < deadline:
            async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                data = await resp.json()
            entry = data.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                status_name = status.get("status_str") or status.get("status_name") or ""
                if status_name == "error":
                    msg = status.get("status_message") or status.get("status_str")
                    raise ComfyUIError(msg or "ComfyUI prompt error")
                if status_name == "success":
                    outputs = entry.get("outputs", {})
                    if target_node is not None:
                        outputs = {target_node: outputs.get(target_node, {})}
                    for outputs_dict in outputs.values():
                        # Scan every output key (STRING, TEXT, output, ANY, ...)
                        # and return the first string value found.
                        for items in outputs_dict.values():
                            if not isinstance(items, list):
                                continue
                            for text in items:
                                if isinstance(text, dict):
                                    content = text.get("content")
                                elif isinstance(text, str):
                                    content = text
                                else:
                                    # Skip numeric/other scalar outputs
                                    # (e.g. ComfyMathExpression results).
                                    continue
                                if content:
                                    return content
                    raise ComfyUIError("Prompt completed but returned no text output")
            await asyncio.sleep(0.5)
        raise ComfyUIError(f"Timed out after {timeout}s waiting for prompt {prompt_id}")

    async def fetch_image(self, filename: str) -> bytes:
        """Fetch a saved image. ComfyUI serves output images via the /view endpoint."""
        session = get_session()
        async with session.get(
            f"{self.base_url}/view",
            params={"filename": filename},
        ) as resp:
            if resp.status == 404:
                raise ComfyUIError(f"Image not found: {filename}")
            resp.raise_for_status()
            return await resp.read()

    async def upload_image(self, data: bytes, filename: str) -> str:
        """Upload an image into ComfyUI's user folder; returns the stored name."""
        form = aiohttp.MultipartWriter("form-data")
        form.append(
            filename,
            headers={"Content-Disposition": 'form-data; name="filename"'},
        )
        form.append(
            data,
            headers={"Content-Disposition": f'form-data; name="image"; filename="{filename}"'},
        )
        session = get_session()
        async with session.post(f"{self.base_url}/upload/image", data=form) as resp:
            data_json = await resp.json()
            if "error" in data_json:
                raise ComfyUIError(data_json["error"])
            return data_json.get("name", filename)
