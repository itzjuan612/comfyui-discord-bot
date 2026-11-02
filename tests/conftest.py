"""Offline test setup.

``workflow`` imports the application's ``core`` module for config, logging, the
ComfyUI client and user settings. The helpers under test are pure functions,
so a stand-in is installed instead of requiring discord.py, aiohttp, a
``config.yaml`` and a running ComfyUI.
"""

import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "core" not in sys.modules:
    core = types.ModuleType("core")
    core.config = {"models": {}}
    core.comfy = None
    core.log = logging.getLogger("test")
    core.ComfyUIError = Exception
    core.last_workflow_by_model = {}
    core.last_ckpt_by_model = {}
    core.user_settings = types.SimpleNamespace(
        is_split_checkpoint=lambda name: False,
        mark_split_checkpoint=lambda name: None,
    )
    core.normalize_aspect_ratio = lambda value: value
    sys.modules["core"] = core
