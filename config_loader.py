import os

import yaml


DEFAULT_CONFIG = """\
# ComfyUI Discord Bot configuration
# Copy this file to config.yaml and fill in your values.
# The bot will auto-generate this file if config.yaml is missing.

comfyui:
  base_url: "http://localhost:8188"

# NSFW guardrail: refuse NSFW prompts unless run in a Discord-marked NSFW channel.
nsfw:
  # Optional extra keywords beyond the built-in list.
  extra_terms: []
  # Image-level NSFW check (lightweight CPU ONNX classifier).
  # Runs on every generated image; NSFW output is allowed only in
  # Discord-marked NSFW channels.
  image_check: true
  image_threshold: 0.5

discord:
  token: ""

# Predefined bot owner. The owner can delete any
# generation, restart the bot, and manage admins/bans via /admin.
# Set owner.id to your own Discord user ID (enable Developer Mode to see it).
owner:
  id: 0   # your Discord user ID (enable Developer Mode to see it)

# LLM (OpenAI-compatible) endpoint used by /gen_prompt.
# The bot sends <base URL>/v1/chat/completions to the workflow.
#
# Option A (local server): provide address + port (port optional, defaults
#   to the standard port). Example:
#   address: "192.168.100.239"
#   port: 8000
# Option B (cloud API like OpenAI): provide the full base URL in `url`
#   and your api_token. Example:
#   url: "https://api.openai.com"
#   api_token: "sk-..."
# You may also use `scheme: "https"` together with address/port instead of url.
llm:
  address: "127.0.0.1"   # your LLM server IP or hostname
  port: 1234                  # your LLM server port (optional for HTTPS endpoints)
  # url: "https://api.openai.com"   # alternative: full base URL
  # scheme: "https"                 # if using address/port with an HTTPS endpoint
  api_token: ""               # optional; leave empty if the API needs no auth
  default_model: ""             # optional default model if the user doesn't choose one
  timeout: 300                  # seconds to wait for the LLM to respond (default 300)
  thinking_default: "medium"    # default reasoning effort used when /gen_prompt omits thinking

models:
  # default_model: the file the workflow loads by default (override to swap models).
  sdxl:
    t2i:
      file: workflows/t2i/sdxl_t2i.json
      default_model: "SDXL.safetensors"
      prompt_node: 6
      negative_node: 7
      seed_node: 114
      seed_key: noise_seed
      steps_node: 114
      latent_node: 5
      cfg_node: 114
      sampler_node: 114
      switch_node: 107
      lora1_node: 108
      lora2_node: 109
    upscale:
      file: workflows/upscale/sdxl_upscale.json
      default_model: "SDXL.safetensors"
      prompt_node: 5
      negative_node: 6
      seed_node: 7
      steps_node: 7
      denoise_node: 7
      cfg_node: 7
      image_node: 1
      scale_node: 2
      scale_key: scale_by
  seedvr2:
    upscale:
      file: workflows/upscale/SeedVR2.json
      default_model: "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors"
      seed_node: 12
      seed_key: seed
      image_node: 13
      # SeedVR2 uses absolute pixel targets: value = longest side * scale.
      scale_node: 12
      scale_keys: [resolution, max_resolution]
      scale_mode: resolution
  flashvsr:
    upscale:
      file: workflows/upscale/FlashVSR.json
      default_model: "FlashVSR-v1.1"
      seed_node: 10
      seed_key: seed
      image_node: 13
      scale_node: 10
      scale_key: scale
  ideogram:
    t2i:
      # Node IDs in this workflow are prefixed "98:" (from ComfyUI's
      # graph export), so they must be quoted as strings in YAML.
      file: workflows/t2i/Ideogram_4_generator.json
      default_model: "ideogram4_fp8_scaled.safetensors"
      model_node: "98:23"
      # The workflow has a second UNet loader for the unconditional branch.
      default_model_unconditional: "ideogram4_unconditional_fp8_scaled.safetensors"
      model_node_unconditional: "98:154"
      prompt_node: "98:24"
      prompt_key: text
      # Seed flows through the Random Number node (its "seed" input seeds the RNG).
      seed_node: "190"
      seed_key: seed
      # ResolutionSelector node: megapixels + aspect ratio (replaces width/height).
      resolution_node: "37"
      megapixels_key: megapixels
      aspect_ratio_key: aspect_ratio
      # Quality preset (Turbo/Default/Quality) via the CustomCombo node.
      quality_node: "98:156"
      quality_key: choice
  flux2_klein:
    i2i_single:
      file: workflows/i2i/image_flux2_klein_image_edit_4b_base.json
      default_model: "flux-2-klein-base-4b-fp8.safetensors"
      prompt_node: "75:74"
      negative_node: "75:67"
      seed_node: "75:73"
      seed_key: noise_seed
      steps_node: "75:62"
      cfg_node: "75:63"
      sampler_node: "75:61"
      image_node: "76"
      megapixels_nodes: ["75:80"]
    i2i_multi:
      file: workflows/i2i/image_flux2_klein_multi_image_edit_4b_base.json
      default_model: "flux-2-klein-base-4b-fp8.safetensors"
      prompt_node: "92:113"
      negative_node: "92:87"
      seed_node: "92:105"
      seed_key: noise_seed
      steps_node: "92:115"
      cfg_node: "92:114"
      sampler_node: "92:102"
      image_nodes: ["76", "81"]
      megapixels_nodes: ["92:110", "92:85"]
"""


def _generate_default_config(path: str) -> None:
    """Write a default config.yaml so the user knows what to fill in."""
    with open(path, "w") as f:
        f.write(DEFAULT_CONFIG)
    print(f"[config] Generated default config at {path}")
    print("[config] Edit it to set your Discord token, ComfyUI URL, and LLM endpoint.")


# Default model files for each workflow, keyed by model name -> workflow type.
# Older config.yaml files may lack these keys; backfill them so the bot
# applies explicit defaults instead of relying on the workflow's built-in values.
WORKFLOW_MODEL_DEFAULTS = {
    "sdxl": {
        "t2i": {"default_model": "SDXL.safetensors"},
        "upscale": {"default_model": "SDXL.safetensors"},
    },
    "seedvr2": {
        "upscale": {
            "default_model": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors"
        },
    },
    "flashvsr": {
        "upscale": {"default_model": "FlashVSR-v1.1"},
    },
    "flux2_klein": {
        "i2i_single": {"default_model": "flux-2-klein-base-4b-fp8.safetensors"},
        "i2i_multi": {"default_model": "flux-2-klein-base-4b-fp8.safetensors"},
    },
    "ideogram": {
        "t2i": {
            "default_model": "ideogram4_fp8_scaled.safetensors",
            "model_node": "98:23",
            "default_model_unconditional": "ideogram4_unconditional_fp8_scaled.safetensors",
            "model_node_unconditional": "98:154",
        },
    },
}


def _backfill_model_defaults(config: dict) -> None:
    """Fill in missing model-default keys for every workflow (in place).

    Existing values are never overwritten, so user overrides are preserved.
    Only workflows already present in the config get backfilled.
    """
    models = config.get("models") or {}
    for model_name, workflow in models.items():
        if not isinstance(workflow, dict):
            continue
        defaults_by_type = WORKFLOW_MODEL_DEFAULTS.get(model_name)
        if not defaults_by_type:
            continue
        for workflow_type, defaults in defaults_by_type.items():
            spec = workflow.get(workflow_type)
            if not isinstance(spec, dict):
                continue
            for key, value in defaults.items():
                spec.setdefault(key, value)


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    if not os.path.exists(path):
        _generate_default_config(path)

    with open(path) as f:
        config = yaml.safe_load(f) or {}
    _backfill_model_defaults(config)
    return config
