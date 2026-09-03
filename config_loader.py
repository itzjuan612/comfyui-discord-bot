import os
import copy

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
  # Image-level NSFW check (EraX-NSFW-V1.0, lightweight CPU ONNX detector).
  # Runs on every generated image; NSFW output is allowed only in
  # Discord-marked NSFW channels. image_threshold is the detection
  # confidence [0..1] that flags an image as NSFW.
  image_check: true
  image_threshold: 0.3

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
      sampler_node: 7
      image_node: 1
      scale_node: 2
      scale_key: scale_by
      switch_node: 203
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
  zimage:
    t2i:
      # Node IDs in this workflow are prefixed "57:" (from ComfyUI's graph
      # export), so they must be quoted as strings in YAML.
      file: workflows/t2i/z_image.json
      default_model: "z_image_turbo_int8_convrot.safetensors"
      prompt_node: "57:27"
      negative_node: "57:65"
      seed_node: "57:3"
      seed_key: seed
      steps_node: "57:3"
      cfg_node: "57:3"
      sampler_node: "57:3"
      latent_node: "57:13"
      # Two LoRA loaders in the model-only chain (UNETLoader -> 57:63 -> 57:64
      # -> ModelSamplingAuraFlow -> KSampler). lora1 is the one closer to the
      # UNet loader, lora2 the one closer to the sampler, so the chain order is
      # preserved when reconnected.
      lora1_node: "57:63"
      lora2_node: "57:64"
      # Unified LoRA strength via the PrimitiveFloat node (drives both loaders).
      lora_strength_node: "57:67"
      # Model chain endpoints for LoRA reconnection (start = UNet loader,
      # end = ModelSamplingAuraFlow which feeds the KSampler).
      model_chain_start: "57:28"
      model_chain_end: "57:11"  
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


def _default_models_spec() -> dict:
    """Return the model->workflow-type->key->value defaults, derived from
    ``DEFAULT_CONFIG`` so the backfill always matches the bundled template.

    Deriving the backfill from ``DEFAULT_CONFIG`` (instead of a separate
    hardcoded table) means that whenever you release a new model command that
    requires a new workflow, simply adding it to ``DEFAULT_CONFIG`` is enough
    for existing config.yaml files to pick up every new key automatically.
    """
    models = yaml.safe_load(DEFAULT_CONFIG) or {}
    spec = models.get("models") or {}
    return {
        model_name: {
            workflow_type: dict(workflow_spec)
            for workflow_type, workflow_spec in workflows.items()
            if isinstance(workflow_spec, dict)
        }
        for model_name, workflows in spec.items()
        if isinstance(workflows, dict)
    }


def _backfill_model_defaults(config: dict) -> tuple[None, dict]:
    """Backfill missing keys from ``DEFAULT_CONFIG`` into the loaded config.

    Every model in the template is added to the config, and every key within
    each workflow type is filled in. Existing values are never overwritten,
    so user overrides are preserved.

    Returns ``(None, missing)`` where ``missing`` records exactly what was
    added, so the file can be updated in place without touching user values
    or comments.

    Because the defaults are derived from ``DEFAULT_CONFIG``, releasing a new
    model/workflow (i.e. adding it to ``DEFAULT_CONFIG``) automatically
    backfills all of its keys into existing config.yaml files.
    """
    defaults = _default_models_spec()
    models = config.setdefault("models", {})
    missing = {"models": [], "workflows": [], "keys": []}
    for model_name, workflows in defaults.items():
        spec = models.get(model_name)
        if spec is None or not isinstance(spec, dict):
            models[model_name] = dict(workflows)
            missing["models"].append(model_name)
            spec = models[model_name]
            for workflow_type, defaults_by_type in workflows.items():
                missing["workflows"].append((model_name, workflow_type))
        else:
            for workflow_type, defaults_by_type in workflows.items():
                inner = spec.get(workflow_type)
                if inner is None or not isinstance(inner, dict):
                    spec[workflow_type] = dict(defaults_by_type)
                    missing["workflows"].append((model_name, workflow_type))
                else:
                    for key, value in defaults_by_type.items():
                        if key not in inner:
                            inner[key] = value
                            missing["keys"].append((model_name, workflow_type, key))
    return None, missing


def _find_block_end(lines: list[str], start: int, indent: int) -> int:
    """Index of the first line that ends the block starting at ``start``.

    A mapping block at ``indent`` spaces ends at the first subsequent
    non-blank line whose indentation is less than or equal to ``indent``.
    Blank lines are skipped (they do not end the block). This preserves
    comments and existing structure while locating where to append new keys.
    """
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        cur = len(line) - len(line.lstrip())
        if cur <= indent:
            break
        i += 1
    return i


def _yaml_value(value) -> str:
    """Render a scalar/list value as a YAML string (lists inline)."""
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_value(v) for v in value) + "]"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return '"' + str(value) + '"'


def _find_models_line(lines: list[str]) -> int | None:
    """Index of the top-level ``models:`` line, or None."""
    for i, line in enumerate(lines):
        if line.strip() == "models:":
            return i
    return None


def _scan_model_blocks(lines: list[str], models_line: int) -> dict[str, tuple[int, int]]:
    """Map each existing model name to ``(start_index, block_end_index)``.

    Only lines within the ``models`` section (before the next top-level key)
    are considered, so keys from other top-level sections are ignored.
    """
    section_end = _find_block_end(lines, models_line, 0)
    blocks = {}
    for i in range(models_line + 1, section_end):
        stripped = lines[i].strip()
        if stripped and lines[i].startswith("  ") and not lines[i].startswith("    ") and stripped.endswith(":"):
            name = stripped.rstrip(":")
            blocks[name] = (i, _find_block_end(lines, i, 2))
    return blocks


def _existing_keys(lines: list[str], start: int, end: int) -> set[str]:
    """The set of key names present in the lines ``[start, end)``."""
    keys = set()
    for j in range(start, end):
        s = lines[j].strip()
        if s and not s.startswith("#") and ":" in s:
            keys.add(s.split(":", 1)[0])
    return keys


def _persist_backfilled_text(path: str, text: str, missing: dict) -> None:
    """Append missing model/workflow/key lines to the config file on disk.

    Only the lines that were actually missing are inserted, at the correct
    indentation, so the file keeps all of its comments and the user's
    existing values. The file is rewritten only when something was added.

    Each existing model block is rebuilt independently: missing keys are
    inserted inside the correct workflow sub-block, and missing workflow types
    are appended to the model. Entirely-missing models are appended at the end
    of the ``models`` section.
    """
    if not missing["models"] and not missing["workflows"] and not missing["keys"]:
        return

    lines = text.splitlines()
    defaults = _default_models_spec()
    models_line = _find_models_line(lines)

    if models_line is None:
        lines.append("models:")
        for model_name in missing["models"]:
            lines.extend(_model_block_lines(model_name, defaults))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return

    model_blocks = _scan_model_blocks(lines, models_line)

    # Rebuild each existing model block, bottom-to-top so earlier indices stay valid.
    for name in sorted(model_blocks.keys(), key=lambda n: model_blocks[n][0], reverse=True):
        start, end = model_blocks[name]
        block_lines = list(lines[start:end])  # local index 0..(end-start-1)

        # Positions of each template workflow type within this model block.
        wf_idx_by_type = {}
        for workflow_type in defaults[name]:
            wf_idx_by_type[workflow_type] = _find_key_at_indent(lines, start + 1, end, 4, workflow_type)

        # Insert missing keys inside their workflow sub-blocks. Process the
        # sub-blocks from bottom to top so earlier positions don't shift.
        existing_wf = [t for t in defaults[name] if wf_idx_by_type[t] is not None]
        for workflow_type in sorted(existing_wf, key=lambda t: wf_idx_by_type[t], reverse=True):
            wf_idx = wf_idx_by_type[workflow_type]
            wf_end = _find_block_end(lines, wf_idx, 4)
            local_wf_end = wf_end - start
            have = _existing_keys(lines, wf_idx, wf_end)
            spec = defaults[name][workflow_type]
            for key in spec:
                if key not in have:
                    block_lines.insert(local_wf_end, "      " + key + ": " + _yaml_value(spec[key]))

        # Append workflow types that are missing entirely to the end of the block.
        for workflow_type in defaults[name]:
            if wf_idx_by_type[workflow_type] is None:
                block_lines.extend(_workflow_block_lines(workflow_type, defaults[name][workflow_type]))

        lines[start:end] = block_lines

    # Append entirely-missing models at the (recomputed) end of the models section.
    if missing["models"]:
        end = _find_block_end(lines, models_line, 0)
        block = []
        for model_name in missing["models"]:
            block.extend(_model_block_lines(model_name, defaults))
        lines[end:end] = block

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _find_key_at_indent(lines: list[str], start: int, end: int, indent: int, key: str) -> int | None:
    """Index of a key line at exactly ``indent`` spaces within [start, end)."""
    target = " " * indent + key + ":"
    for i in range(start, end):
        if lines[i] == target or lines[i].startswith(target):
            return i
    return None


def _model_block_lines(model_name: str, defaults: dict) -> list[str]:
    """YAML lines for an entire model block (indentation: model=2, type=4, keys=6)."""
    out = ["  " + model_name + ":"]
    for workflow_type, spec in defaults[model_name].items():
        out.append("    " + workflow_type + ":")
        for key, value in spec.items():
            out.append("      " + key + ": " + _yaml_value(value))
    return out


def _workflow_block_lines(workflow_type: str, spec: dict) -> list[str]:
    """YAML lines for a single workflow-type block (type=4, keys=6)."""
    out = ["    " + workflow_type + ":"]
    for key, value in spec.items():
        out.append("      " + key + ": " + _yaml_value(value))
    return out


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    if not os.path.exists(path):
        _generate_default_config(path)

    with open(path) as f:
        text = f.read()
    config = yaml.safe_load(text) or {}
    _missing = _backfill_model_defaults(config)[1]
    _persist_backfilled_text(path, text, _missing)
    return config
