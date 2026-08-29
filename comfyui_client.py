import asyncio
import json
import logging
import time
import uuid

import aiohttp

log = logging.getLogger("comfyui_client")


class ComfyUIError(Exception):
    pass


class ComfyUIClient:
    """Async client for the ComfyUI HTTP API."""

    # How long a checkpoint list stays fresh before re-querying ComfyUI.
    CHECKPOINT_CACHE_TTL = 60.0

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._ckpt_cache: list[str] | None = None
        self._ckpt_cache_time: float = 0.0

    async def queue_prompt(self, workflow: dict) -> tuple[str, str]:
        """Queue a workflow (dict of nodes).

        Returns ``(prompt_id, client_id)``. A fresh client_id is generated
        for every prompt so progress tracking never collides with lingering
        WebSocket connections from previous generations.
        """
        client_id = uuid.uuid4().hex
        payload = {"prompt": workflow, "client_id": client_id}
        async with aiohttp.ClientSession() as session:
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
    ) -> list[str]:
        """Poll /history until the prompt finishes; returns output filenames.

        If ``on_progress`` is given, a WebSocket connection is opened and
        ComfyUI's live progress values (0.0 - 1.0) are forwarded to it while
        the prompt executes.
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
                    async with aiohttp.ClientSession() as session:
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
            """Extract progress from a ComfyUI websocket payload and forward it."""
            nodes = payload.get("nodes")
            if isinstance(nodes, dict):
                # progress_state carries per-node progress. Pick the running
                # node with the largest max (the sampling node that drives
                # generation time).
                best = None
                for node_data in nodes.values():
                    if not isinstance(node_data, dict) or node_data.get("state") != "running":
                        continue
                    if best is None or node_data.get("max", 0) > best.get("max", 0):
                        best = node_data
                if best is not None:
                    maximum = float(best.get("max", 1) or 1)
                    value = float(best.get("value", 0))
                    on_progress(value / maximum)
            elif "value" in payload:
                # Legacy flat format fallback.
                value = float(payload["value"])
                maximum = float(payload.get("max", 100) or 100)
                on_progress(value / maximum)

        async def poll_history() -> list[str]:
            nonlocal seen
            while loop.time() < deadline:
                async with aiohttp.ClientSession() as session:
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

    async def fetch_checkpoints(self, force: bool = False) -> list[str]:
        """List checkpoint files available in ComfyUI's models/checkpoints folder.

        The result is cached for ``CHECKPOINT_CACHE_TTL`` seconds so repeated
        calls (e.g. one per generation) don't hammer ComfyUI. Pass
        ``force=True`` to bypass the cache and refresh immediately.
        """
        now = time.monotonic()
        if not force and self._ckpt_cache is not None and now - self._ckpt_cache_time < self.CHECKPOINT_CACHE_TTL:
            return self._ckpt_cache
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/models/checkpoints") as resp:
                if resp.status != 200:
                    raise ComfyUIError(f"Could not list checkpoints (HTTP {resp.status})")
                data = await resp.json()
        # ComfyUI's GET /models/{folder} returns a bare JSON array of plain
        # filename strings (e.g. ["SDXL.safetensors", "flux.safetensors"]).
        # Tolerate a dict wrapper and dict items for forward compatibility.
        items = data.get("checkpoints") if isinstance(data, dict) else data
        names = []
        for item in items:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and "name" in item:
                names.append(item["name"])
        self._ckpt_cache = names
        self._ckpt_cache_time = time.monotonic()
        return self._ckpt_cache

    async def free_memory(self) -> None:
        """Ask ComfyUI to unload all loaded models, freeing VRAM and RAM.

        ComfyUI exposes /free for this. Recent builds accept POST; older
        builds only accept GET, so fall back gracefully.
        """
        async with aiohttp.ClientSession() as session:
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
        while loop.time() < deadline:
            async with aiohttp.ClientSession() as session:
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
        async with aiohttp.ClientSession() as session:
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
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/upload/image", data=form) as resp:
                data_json = await resp.json()
                if "error" in data_json:
                    raise ComfyUIError(data_json["error"])
                return data_json.get("name", filename)
