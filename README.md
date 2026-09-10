# Juan's ComfyUI Discord Bot

A Discord bot that exposes ComfyUI image generation workflows as slash commands. Users can generate text-to-image, image-to-image edits, upscaling, and LLM-assisted prompt engineering directly from Discord channels.

<p><img src="banner.png" alt="Juan's ComfyUI Discord Bot" width="800"></p>

## Features

| Feature | Description |
| --- | --- |
| :art: Ideogram (`/ideogram`) | Generate professional images with Ideogram 4 by using JSON to carefully sketch your image. Supports seed, quality preset, megapixels, and aspect ratio. |
| :art: Stable Diffusion XL (`/sdxl`) | Generate images with versatile SDXL models. Supports prompt, negative prompt, seed, steps, width/height, and CFG. Can use any checkpoint in ComfyUI's `models/checkpoints` folder via the `model` parameter, plus up to two LoRAs (`lora1`, `lora2`) with a unified `lora_strength`. Automatically uses separate CLIP/VAE loaders for checkpoints that lack a bundled text encoder/VAE (split checkpoints). |
| :art: Z-Image Turbo & Base (`/zimage`) | Generate high quality images with very fast and light Z-Image models. Supports natural language prompts, up to two LoRAs, multiple Z-Image models, and also batch images. Other params are steps, width/height, and CFG. |
| :pencil2: Image-to-Image (`/img2img`) | Edit one image or combine two images using Flux 2 Klein 4B Base. Supports cfg, steps, sampler, megapixels. |
| :mag: Upscale (`/upscale`) | Upscale an attached image using SDXL, SeedVR2, or FlashVSR. SDXL shows a picker to use any checkpoint in `models/checkpoints`. Supports custom scale factor, prompt, negative, and strength. |
| :speech_balloon: Prompt Generation (`/gen_prompt`) | Converts a natural-language idea into a structured Ideogram 4 JSON caption using an LLM (OpenAI-compatible endpoint). Includes reasoning-effort selection. |
| :clipboard: LLM Model List (`/llm_models`) | Lists all models available on the configured LLM endpoint. |
| :bar_chart: Progress Bar | Live progress bar updated via ComfyUI WebSocket during generation. |
| :dark_sunglasses: Stealth Mode | Ephemeral messages visible only to the requesting user. |
| :repeat: Retry / Delete Buttons | Persistent buttons on every output message. Retry re-rolls the seed; Delete removes the message (owner or admins can delete any). |
| :rocket: Upscale 2x Button | One-click upscale of any generated image, with model picker. For SDXL images, the SDXL option reuses the checkpoint the image was created with; for non-SDXL images the SDXL option is hidden (SDXL upscale works best with SDXL checkpoints). |
| :paintbrush: Edit Image Button | Opens a modal to run the Flux 2 Klein 4B single-image edit workflow on an output. |
| :gear: Per-User Settings (`/settings`, `/reset_settings`) | Saved defaults for prompts, CFG, steps, sampler, quality, megapixels, aspect ratio, and stealth. Stored in SQLite. |
| :shield: NSFW Guardrail | Keyword-based prompt filter + CPU ONNX image check using the EraX-NSFW-V1.0 detector (runs in RAM, no GPU). NSFW content is blocked unless the channel is Discord-marked NSFW. |
| :police_officer: Moderation | Ban/unban users, promote/demote admins, view user list. Stored in SQLite. |
| :broom: Flush (`/flush`) | Unloads all ComfyUI models and execution cache to free VRAM/RAM. |
| :hammer_and_wrench: Admin Panel (`/admin`) | Owner can restart the bot, manage admins, and manage bans. |
| :stopwatch: Cooldown | Per-user 20-second minimum interval between generations. |
| :hourglass: Job Queueing | Serializes resource-heavy work so concurrent users don't collapse resources. Configurable `queueing.mode` (see below). Users waiting in line see their queue position. |
| :floppy_disk: Memory Management | Automatically frees ComfyUI memory when switching between workflows, or when choosing a different checkpoint (`model`) for `/sdxl`. Reusing the same checkpoint causes no flush. |

## Commands Reference

### `/ideogram`

| Parameter | Description |
| --- | --- |
| `prompt` | Text prompt (required) |
| `seed` | Seed (optional) |
| `quality` | Ideogram preset: Turbo / Default / Quality |
| `megapixels` | Target resolution in MP |
| `aspect_ratio` | Aspect ratio preset |
| `stealth` | Ephemeral output |

### `/sdxl` and `/zimage`

| Parameter | Description |
| --- | --- |
| `prompt` | Text prompt (required) |
| `model` | Model filename, SDXL is located in `models/checkpoints` while Z-Image is in `models/diffusion_models` |
| `negative` | Negative prompt |
| `steps` | Sampling steps |
| `width` / `height` | Dimensions, multiple of 64 |
| `sampler` / `scheduler` | Sampler and scheduler (user or workflow default if none is chosen) |
| `cfg` | Guidance scale |
| `lora1` / `lora2` | First and Second LoRA name (optional; dynamic autocomplete from `models/loras`) |
| `seed` | Seed (optional) |
| `lora_strength` | Unified LoRA strength applied to both LoRAs (optional, default 1.0) |
| `stealth` | Ephemeral output |
| `batch_size` | **`/zimage` only**, determines the number of images the model will generate at once |

> **Any checkpoint:** Pass any file name present in ComfyUI's `models/checkpoints` folder via `model` to generate with a different checkpoint instead of the workflow's default. The bot verifies the file exists before running the workflow, and the embed reports the checkpoint actually used. Switching to a different checkpoint frees ComfyUI memory; reusing the same checkpoint does not. If the workflow's default checkpoint (e.g. `SDXL.safetensors`) is missing from `models/checkpoints`, the bot automatically falls back to an available checkpoint (preferring one whose name contains "sdxl") and the embed reports the checkpoint actually used. The `model` parameter uses dynamic autocomplete: as you type, the bot filters the live checkpoint list (cached for 60 seconds), so newly added checkpoint files appear without a bot restart or slash-command re-sync.

> **LoRAs:** `lora1` and `lora2` accept any file in ComfyUI's `models/loras` folder (with dynamic autocomplete). Leaving a slot empty disables that LoRA loader; selecting a file enables it. A single `lora_strength` value is applied to both the model and clip branches of every active LoRA.

> **Split checkpoints:** Some checkpoints do not ship with their own text encoder and VAE. The SDXL workflow contains a `PrimitiveBoolean` switch driving WAS-node-suite `CLIP Input Switch` and `VAE Input Switch` nodes: when `False` it uses the checkpoint's bundled CLIP/VAE, when `True` it uses the separate `qwen_3_06b_base` CLIP and `qwen_image_vae` VAE. The bot detects a split checkpoint automatically: if generation fails because the checkpoint has no valid bundled CLIP, it flips the switch to `True` and retries. Each newly detected split checkpoint is cached (in memory and persisted in `user_settings.db`), so subsequent generations for that checkpoint use the separate loaders directly without an error/retry round trip.

### `/img2img`

| Parameter | Description |
| --- | --- |
| `workflow` | `single` (1 image edit) or `multi` (2 images combine) |
| `image` | First input image |
| `prompt` | Description of the desired edit |
| `image2` | Second input image (required for multi) |
| `cfg` | CFG (optional) |
| `steps` | Steps (optional) |
| `sampler` | Sampler name (optional) |
| `megapixels` | Target MP (optional) |
| `stealth` | Ephemeral output |

### `/gen_prompt`

| Parameter | Description |
| --- | --- |
| `prompt` | Natural-language idea to convert |
| `megapixels` | Target resolution in MP |
| `aspect_ratio` | Aspect ratio preset |
| `model` | LLM model to use (see `/llm_models`) |
| `max_tokens` | Max tokens (optional) |
| `temperature` | LLM temperature 0-1 (optional) |

After selecting the reasoning effort, the bot composes the prompt via the Ideogram prompt-gen workflow, calls the LLM, and posts the resulting JSON caption.

### `/upscale`

| Parameter | Description |
| --- | --- |
| `model` | `sdxl`, `seedvr2`, or `flashvsr` |
| `image` | Image to upscale (attach from gallery) |
| `prompt` | Optional guidance prompt |
| `negative` | Optional negative prompt |
| `strength` | Denoise strength 0-1 (SDXL only) |
| `scale` | Upscale factor (e.g. 2 or 3) |
| `stealth` | Ephemeral output |

> **SDXL checkpoint picker:** Choosing `sdxl` uploads your image, then shows an ephemeral picker listing every checkpoint in ComfyUI's `models/checkpoints` folder (plus a Default option that uses the workflow's own checkpoint). Selecting a checkpoint runs the SDXL upscale workflow with that checkpoint; the embed reports the checkpoint actually used. Switching to a different checkpoint frees ComfyUI memory; reusing the same one does not.

### `/settings` / `/reset_settings`

View or update per-user defaults:

- `negative_prompt`, `width`, `height`
- `sdxl_checkpoint`, `sdxl_cfg`, `sdxl_steps`, `sdxl_sampler`, `sdxl_scheduler` (SDXL)
- `zimage_model`, `zimage_cfg`, `zimage_steps`, `zimage_sampler`, `zimage_scheduler` (Z-Image)
- `ideogram_quality`, `ideogram_megapixels`, `ideogram_aspect_ratio` (Ideogram)
- `img2img_cfg`, `img2img_steps`, `img2img_sampler`, `img2img_megapixels` (img2img Flux.2 Klein)
- `stealth` (default privacy)

### `/flush`

Unloads all models and execution cache from ComfyUI.

### `/admin`

Owner/admin panel: restart bot, promote/demote admins, ban/unban users, view ban list.

## Architecture

```
+--------------+     HTTP API      +--------------+
|  Discord     | ----------------> |   ComfyUI    |
|   Bot        | <---------------- |  (server)    |
| (main.py)    |   WebSocket       +--------------+
|              |
|              | ---- HTTP (v1/chat/completions) ----> LLM endpoint
|              |
| SQLite DBs:  |
| - user_settings.db
| - generations.db
| - moderation.db
+--------------+
```

Key modules:

| File | Role |
| --- | --- |
| `main.py` | Entry point: inits the SQLite DBs, seeds the owner, and runs the bot (`python main.py`). Accepts `DISCORD_TOKEN` env var as an alternative to the `config.yaml` token. |
| `bot.py` | Shared bot instance: creates the Bot, imports the cogs (slash commands), defines `on_ready` (registers the persistent `GenerationView`, syncs slash commands, starts the LLM model refresh loop), and closes the shared HTTP session on shutdown (`on_close`) |
| `core.py` | Shared runtime: config, owner ID, model/upscale/i2i model lists, sampler & aspect-ratio choices, NSFW guard config, cooldown tracking, memory-freeing state, progress bar, image compression, and error replies |
| `http_session.py` | Single shared `aiohttp.ClientSession` reused by all HTTP/WebSocket calls (ComfyUI, LLM, Discord image downloads); created lazily and closed on bot shutdown |
| `comfyui_client.py` | Async HTTP/WS client for ComfyUI (`/prompt`, `/history`, `/view`, `/upload/image`, `/free`); caches the checkpoint list (60s TTL) and lists LoRA files (`/models/loras`) |
| `workflow.py` | Loads workflow JSON, converts graph format to API format, patches node inputs from config specs (including LoRA wiring and the SDXL split-checkpoint switch), and runs image/text workflows with automatic split-checkpoint retry |
| `llm_client.py` | OpenAI-compatible LLM client: model listing, load/unload, reasoning-effort probing, cached model list |
| `job_queue.py` | Configurable serial job queue (unified or separate lanes) that serializes ComfyUI generations and LLM jobs; provides queue-position/waiting messages |
| `config_loader.py` | Loads `config.yaml` (auto-generates it with defaults if missing) and backfills workflow model defaults |
| `user_settings.py` | Per-user defaults (SQLite) |
| `generation_store.py` | Persisted generation params per message (SQLite) |
| `moderation.py` | Admin/ban lists (SQLite) |
| `nsfw_guard.py` | Keyword NSFW filter + EraX-NSFW-V1.0 CPU ONNX image detector |
| `download_erax.py` | Downloads EraX-NSFW-V1.0 detector on boot if it's missing |
| `diag.py` | Standalone diagnostic: prints local vs. server-side slash commands for debugging sync issues |
| `cogs/` | Slash commands: `generation.py` (/ideogram, /sdxl, /zimage, /upscale, /img2img, /flush), `llm.py` (/gen_prompt, /llm_models), `settings.py` (/settings, /reset_settings), `admin.py` (/admin) |
| `ui/views.py` | All interactive UI: generation buttons (Retry/Delete/Upscale/Edit), checkpoint picker, reasoning-effort picker, and admin panel views |
| `ui/autocomplete.py` | Discord slash-command autocomplete handlers (LLM model, SDXL checkpoint, LoRA) |

## Setup

Refer to https://github.com/itzjuan612/comfyui-discord-bot/wiki/Installation-and-Setup for the setup guide

## LLM Endpoint Configuration

The bot supports two ways to configure the LLM:

**Option A - Local server (e.g., LM Studio, Ollama):**

```yaml
llm:
  address: "192.168.100.239"
  port: 1234
```

**Option B - Cloud API (e.g., OpenAI):**

```yaml
llm:
  url: "https://api.openai.com"
  api_token: "sk-..."
```

The bot calls `<base_url>/v1/chat/completions` and can also query `/v1/models` or `/api/v1/models` (LM Studio) to list available models. It optionally loads/unloads models via LM Studio's `/api/v1/models/load` and `/api/v1/models/unload` endpoints.

## Job Queueing

Resource-heavy work — ComfyUI image generation and the `/gen_prompt` LLM pipeline (model load, reasoning-effort probe, generation, model unload) — is routed through an internal job queue so that concurrent users cannot exhaust the machine. Set the scheduling mode in `config.yaml`:

```yaml
queueing:
  mode: unified   # or "separate"
```

- **`unified`** (default) — one single serial queue. Image generation and prompt generation never run at the same time (at most 1 concurrent job). Safest against resource collapse.
- **`separate`** — two independent serial lanes (`comfyui` and `llm`). Each lane runs one job at a time, but the two lanes run in parallel, so an image generation and a prompt generation may run concurrently (up to 2 jobs, one per resource).

Users whose job must wait see their queue position prepended to the progress message, e.g. `⏳ You're #1 in line.` (the first waiting job is #1). When no job is running, no prefix is shown.

## Storage

Three SQLite databases are created automatically in the project folder:

| Database | Purpose |
| --- | --- |
| `user_settings.db` | Per-user default prompts, CFG, steps, sampler, stealth, etc., plus a cache of SDXL split checkpoints (those requiring separate CLIP/VAE loaders). |
| `generations.db` | Stores generation parameters per Discord message ID (for retry/delete) |
| `moderation.db` | Admin and ban lists |

## File Structure

```
comfyuidiscord/
|-- main.py                 # Entry point (launches the bot)
|-- bot.py                  # Shared bot instance + cogs (slash commands)
|-- core.py                 # Shared runtime: config, choices, cooldowns, progress bar
|-- http_session.py         # Shared aiohttp session (connection reuse)
|-- comfyui_client.py       # ComfyUI HTTP/WS client
|-- workflow.py             # Workflow loading, graph->API conversion, spec patching
|-- llm_client.py           # LLM client: models, load/unload, reasoning-effort probing
|-- job_queue.py            # Configurable serial job queue (unified/separate lanes)
|-- config.yaml             # Configuration (auto-generated if missing)
|-- config.example.yaml     # Configuration template
|-- config_loader.py        # YAML loader + model-default backfill
|-- generation_store.py     # Generation params (SQLite)
|-- moderation.py           # Admin/ban (SQLite)
|-- nsfw_guard.py           # NSFW text + EraX-NSFW-V1.0 CPU ONNX image detector
|-- download_erax.py        # Downloads/exports the NSFW ONNX model (nano/small/medium)
|-- user_settings.py        # Per-user settings (SQLite)
|-- generations.db          # Generation params database (SQLite, auto-created)
|-- moderation.db           # Admin/ban database (SQLite, auto-created)
|-- user_settings.db        # Per-user settings database (SQLite, auto-created)
|-- diag.py                 # Slash-command sync diagnostic
|-- requirements.txt
|-- start.bat
|-- LICENSE.txt
|-- banner.png
|-- .gitignore
|-- models/
|   `-- erax_nsfw.onnx        # EraX-NSFW-V1.0 CPU ONNX detector (auto-downloaded if missing)
|-- cogs/
|   |-- __init__.py         # Package marker
|   |-- generation.py       # /ideogram, /sdxl, /upscale, /img2img, /flush
|   |-- llm.py              # /gen_prompt, /llm_models
|   |-- settings.py         # /settings, /reset_settings
|   `-- admin.py            # /admin
|-- ui/
|   |-- __init__.py         # Package marker
|   `-- views.py            # All interactive UI (buttons, pickers, modals, admin panel)
|   `-- autocomplete.py     # Discord slash-command autocomplete handlers
`-- workflows/
    |-- t2i/
    |   |-- sdxl_t2i.json
    |   |-- Ideogram_4_generator.json
    |   `-- z_image.json
    |-- i2i/
    |   |-- image_flux2_klein_image_edit_4b_base.json
    |   `-- image_flux2_klein_multi_image_edit_4b_base.json
    |-- upscale/
    |   |-- sdxl_upscale.json
    |   |-- SeedVR2.json
    |   `-- FlashVSR.json
    `-- ideogram_prompt_gen/
        `-- ideogram4_prompt_gen.json
```

## Notes

- The bot requires ComfyUI to be running before starting.
- The `--allow-cors` flag is passed to `main.py` (it does not affect ComfyUI itself).
- Images larger than 18 MB are automatically re-encoded as JPEG to stay under Discord's upload limit.
- The NSFW image check uses **EraX-NSFW-V1.0** (a YOLO11 nano detector exported to ONNX) running entirely on CPU via ONNX Runtime, so it uses RAM rather than VRAM and does not contend with ComfyUI. If the model is missing, the bot auto-downloads and exports it on first boot (you can pick nano/small/medium, defaulting to nano).
- Persistent buttons (Retry, Delete, Upscale, Edit) survive bot restarts via a globally registered Discord View.
- All HTTP and WebSocket traffic (ComfyUI API, LLM endpoint, Discord image downloads) goes through a single shared `aiohttp.ClientSession` (`http_session.py`) instead of creating a new session per request. This enables TCP Keep-Alive connection reuse, avoids repeated handshakes, and prevents file-descriptor exhaustion during concurrent generations. The session is created lazily (and recreated if the event loop changes) and closed once when the bot shuts down.
- **LLM model caching:** The LLM model list is cached and refreshed every 300 seconds by a background loop. A failed refresh keeps the previous cache so slash-command autocomplete never breaks.
- **Reasoning-effort probing:** Before showing the `/gen_prompt` reasoning-effort picker, the bot probes the endpoint with minimal 1-token requests to discover which effort values (`low`, `medium`, `high`, `xhigh`, `on`, `off`) it accepts. Only accepted values appear as options, plus "API default" (sends no effort tag). Results are cached per model.
- **LLM load/unload:** `/gen_prompt` loads the chosen model before composing the prompt and unloads it after completion or timeout (via LM Studio's `/api/v1/models/load` / `/api/v1/models/unload`). On non-LM-Studio servers these calls are silently skipped.
- **Restart:** The `/admin` "Restart Bot" button spawns a fresh process with the same command-line arguments and exits immediately.
- **`/settings` view mode:** Passing `view=true` (or no parameters) displays your saved defaults without changing them.
- **Transient error auto-deletion:** Error messages (NSFW blocks, cooldowns, generation failures) are deleted automatically after 5 seconds to keep the channel clean.
- **`diag.py`:** Standalone diagnostic that prints local vs. server-side slash commands for debugging command sync issues.
- **Output embed:** Each generation embed lists the seed, steps, CFG, sampler, and the checkpoint actually used, so you can see exactly what ran.
- **SDXL split-checkpoint cache:** Checkpoints that lack a bundled CLIP/VAE are detected on first use (error + retry) and then cached in memory and persisted in `user_settings.db`. On subsequent runs the bot flips the workflow's switch node to use the separate `qwen_3_06b_base` CLIP and `qwen_image_vae` VAE directly, avoiding the error/retry round trip. The cache loads at startup, so the retry cost is paid only once per checkpoint.
- **SDXL LoRAs:** `/sdxl` accepts `lora1` and `lora2` (each with dynamic autocomplete over `models/loras`). An empty slot disables that LoRA loader (the node is removed from the graph and the model/clip chain is rewired around it); a selected file enables it. A single `lora_strength` value is applied to both the model and clip branches of every active LoRA.
- **Random seeds:** When `seed` is omitted, the bot picks a random seed in the range 0 to 2^32-1; the Retry button re-rolls a fresh seed for a new image.
- **Aspect-ratio short forms:** Aspect ratio inputs accept short forms like `16:9`, which are auto-normalized to the exact ResolutionSelector label (e.g. `16:9 (Widescreen)`).
- **NSFW negative prompts:** The NSFW keyword filter ignores negative prompts, so a negative prompt containing "nsfw" (meaning "exclude nsfw") is not blocked.
