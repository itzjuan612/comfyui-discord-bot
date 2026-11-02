"""Tests for the spec patchers that rewrite an API graph before queueing."""

import copy

import pytest

from workflow import _current_model as workflow_current_model
from workflow import apply_spec

BASE_SPEC = {
    "prompt_node": "20",
    "negative_node": "21",
    "seed_node": "30",
    "steps_node": "30",
    "cfg_node": "30",
    "sampler_node": "30",
    "latent_node": "25",
}


def base_graph() -> dict:
    """A small API-format graph carrying one of every node the patchers touch."""
    return {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "orig.safetensors"}},
        "11": {"class_type": "UNETLoader", "inputs": {"unet_name": "orig-unet.safetensors"}},
        "12": {"class_type": "CLIPLoader", "inputs": {"clip_name": "orig-te.safetensors"}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": "old prompt", "clip": ["10", 1]}},
        "21": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old negative", "clip": ["10", 1]},
        },
        "25": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "30": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "model": ["10", 0],
            },
        },
    }


def patched(kwargs=None, spec=None, graph=None):
    api = copy.deepcopy(graph if graph is not None else base_graph())
    apply_spec(api, spec if spec is not None else BASE_SPEC, **(kwargs or {}))
    return api


def test_text_and_sampling_inputs():
    api = patched(
        {
            "prompt": "a cat",
            "negative": "blurry",
            "seed": 7,
            "steps": 30,
            "cfg": 4.5,
            "sampler": "dpmpp_2m",
            "scheduler": "karras",
        }
    )
    assert api["20"]["inputs"]["text"] == "a cat"
    assert api["21"]["inputs"]["text"] == "blurry"
    assert api["30"]["inputs"]["seed"] == 7
    assert api["30"]["inputs"]["steps"] == 30
    assert api["30"]["inputs"]["cfg"] == 4.5
    assert api["30"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert api["30"]["inputs"]["scheduler"] == "karras"


def test_denoise_strength_only_with_denoise_node():
    spec = {**BASE_SPEC, "denoise_node": "30"}
    assert patched({"strength": 0.4}, spec)["30"]["inputs"]["denoise"] == 0.4
    # Without a denoise_node the value is ignored, not written anywhere.
    assert "denoise" not in patched({"strength": 0.4})["30"]["inputs"]


def test_latent_size_needs_both_dimensions():
    assert patched({"width": 1024})["25"]["inputs"] == {
        "width": 512,
        "height": 512,
        "batch_size": 1,
    }
    api = patched({"width": 1024, "height": 768})
    assert (api["25"]["inputs"]["width"], api["25"]["inputs"]["height"]) == (1024, 768)


def test_batch_size_is_independent_of_dimensions():
    assert patched({"batch_size": 4})["25"]["inputs"]["batch_size"] == 4


def test_model_selection_writes_the_first_loader():
    assert (
        patched({"ckpt_name": "new.safetensors"})["10"]["inputs"]["ckpt_name"] == "new.safetensors"
    )


def test_model_node_pins_the_loader():
    spec = {**BASE_SPEC, "model_node": "11"}
    api = patched({"ckpt_name": "diT.safetensors"}, spec)
    assert api["11"]["inputs"]["unet_name"] == "diT.safetensors"
    assert api["10"]["inputs"]["ckpt_name"] == "orig.safetensors"


def test_model_key_handles_an_unknown_loader_class():
    graph = base_graph()
    graph["11"] = {"class_type": "SomeNewLoader", "inputs": {"path": "orig.safetensors"}}
    spec = {**BASE_SPEC, "model_node": "11", "model_key": "path"}
    assert (
        patched({"ckpt_name": "new.safetensors"}, spec, graph)["11"]["inputs"]["path"]
        == "new.safetensors"
    )


def test_unknown_loader_class_without_model_key_raises():
    graph = base_graph()
    graph["11"] = {"class_type": "SomeNewLoader", "inputs": {"path": "orig.safetensors"}}
    with pytest.raises(ValueError, match="model_key"):
        patched({"ckpt_name": "new.safetensors"}, {**BASE_SPEC, "model_node": "11"}, graph)


def test_model_node_pointing_at_a_missing_node_raises():
    with pytest.raises(ValueError):
        patched({"ckpt_name": "new.safetensors"}, {**BASE_SPEC, "model_node": "999"})


def test_model_selection_without_a_loader_raises():
    graph = base_graph()
    del graph["10"]
    del graph["11"]
    with pytest.raises(ValueError):
        patched({"ckpt_name": "new.safetensors"}, graph=graph)


def test_second_model_loader_for_the_unconditional_branch():
    spec = {
        **BASE_SPEC,
        "model_node": "11",
        "model_node_unconditional": "11",
        "default_model_unconditional": "uncond.safetensors",
    }
    assert patched({}, spec)["11"]["inputs"]["unet_name"] == "uncond.safetensors"


def test_unconditional_model_uses_its_declared_key():
    graph = base_graph()
    graph["11"] = {"class_type": "DualLoader", "inputs": {"model_b": "orig"}}
    spec = {
        **BASE_SPEC,
        "model_node_unconditional": "11",
        "model_key_unconditional": "model_b",
        "default_model_unconditional": "uncond.safetensors",
    }
    assert patched({}, spec, graph)["11"]["inputs"]["model_b"] == "uncond.safetensors"


def test_text_encoder_default_and_override():
    spec = {**BASE_SPEC, "text_encoder_node": "12", "default_text_encoder": "default.safetensors"}
    assert patched({}, spec)["12"]["inputs"]["clip_name"] == "default.safetensors"
    assert (
        patched({"text_encoder": "mine.safetensors"}, spec)["12"]["inputs"]["clip_name"]
        == "mine.safetensors"
    )


def test_prompt_can_enter_through_an_intermediate_node():
    spec = {**BASE_SPEC, "prompt_input_node": "40", "prompt_input_key": "text"}
    graph = base_graph()
    graph["40"] = {"class_type": "GoogleTranslateTextNode", "inputs": {"text": ""}}
    api = patched({"prompt": "hola"}, spec, graph)
    assert api["40"]["inputs"]["text"] == "hola"
    # The encoder consumes that node's output (the prompt was written there),
    # so its own text input becomes a link to it.
    assert api["20"]["inputs"]["text"] == ["40", 0]


def test_translation_switch_is_inverted():
    spec = {**BASE_SPEC, "translate_node": "40", "translate_key": "manual_translate"}
    graph = base_graph()
    graph["40"] = {"class_type": "GoogleTranslateTextNode", "inputs": {"manual_translate": False}}
    assert patched({}, spec, graph)["40"]["inputs"]["manual_translate"] is True
    assert patched({"translate": True}, spec, graph)["40"]["inputs"]["manual_translate"] is False


def test_enhance_off_removes_chain_and_rewires_the_encoder():
    spec = {
        **BASE_SPEC,
        "prompt_node": "20",
        "prompt_input_node": "40",
        "prompt_key": "prompt",
        "enhance_nodes": ["41", "42"],
    }
    graph = base_graph()
    graph["40"] = {"class_type": "GoogleTranslateTextNode", "inputs": {"text": ""}}
    graph["41"] = {"class_type": "StringConcatenate", "inputs": {}}
    graph["42"] = {"class_type": "TextGenerate", "inputs": {}}
    graph["20"]["inputs"]["prompt"] = ["42", 0]
    api = patched({}, spec, graph)
    assert "41" not in api and "42" not in api
    assert api["20"]["inputs"]["prompt"] == ["40", 0]


def test_enhance_rewire_uses_the_default_prompt_key():
    # No prompt_key in the spec: the rewire must write the same input the
    # prompt was written to ("text"), not a hard-coded "prompt".
    spec = {**BASE_SPEC, "prompt_node": "20", "prompt_input_node": "40", "enhance_nodes": ["41"]}
    graph = base_graph()
    graph["40"] = {"class_type": "GoogleTranslateTextNode", "inputs": {"text": ""}}
    graph["41"] = {"class_type": "TextGenerate", "inputs": {}}
    api = patched({}, spec, graph)
    assert api["20"]["inputs"]["text"] == ["40", 0]
    assert "prompt" not in api["20"]["inputs"]


def test_enhance_on_keeps_the_chain():
    spec = {**BASE_SPEC, "enhance_nodes": ["41"]}
    graph = base_graph()
    graph["41"] = {"class_type": "TextGenerate", "inputs": {}}
    assert "41" in patched({"enhance": True}, spec, graph)


def test_enhance_switch_node():
    spec = {**BASE_SPEC, "enhance_node": "38", "enhance_key": "switch"}
    graph = base_graph()
    graph["38"] = {"class_type": "ComfySwitchNode", "inputs": {"switch": True}}
    assert patched({}, spec, graph)["38"]["inputs"]["switch"] is False
    assert patched({"enhance": True}, spec, graph)["38"]["inputs"]["switch"] is True


def test_single_image_input():
    spec = {**BASE_SPEC, "image_node": "50"}
    graph = base_graph()
    graph["50"] = {"class_type": "LoadImage", "inputs": {"image": "old.png"}}
    assert patched({"image_filename": "new.png"}, spec, graph)["50"]["inputs"]["image"] == "new.png"


def test_multiple_image_inputs():
    spec = {**BASE_SPEC, "image_nodes": ["50", "51"]}
    graph = base_graph()
    graph["50"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
    graph["51"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
    api = patched({"image_files": ["a.png", "b.png"]}, spec, graph)
    assert api["50"]["inputs"]["image"] == "a.png"
    assert api["51"]["inputs"]["image"] == "b.png"


def test_second_image_removed_when_not_supplied():
    spec = {
        **BASE_SPEC,
        "image2_node": "60",
        "image2_key": "images.image_2",
        "image2_refs": [["20", "images.image1"]],
    }
    graph = base_graph()
    graph["60"] = {"class_type": "LoadImage", "inputs": {"image": "ref.png"}}
    graph["20"]["inputs"]["images.image_2"] = ["60", 0]
    graph["20"]["inputs"]["images.image1"] = ["60", 0]
    api = patched({}, spec, graph)
    assert "60" not in api
    assert "images.image_2" not in api["20"]["inputs"]
    assert "images.image1" not in api["20"]["inputs"]


def test_second_image_kept_when_supplied():
    spec = {**BASE_SPEC, "image2_node": "60"}
    graph = base_graph()
    graph["60"] = {"class_type": "LoadImage", "inputs": {"image": "ref.png"}}
    assert (
        patched({"image2_filename": "second.png"}, spec, graph)["60"]["inputs"]["image"]
        == "second.png"
    )


def lora_graph() -> dict:
    graph = base_graph()
    graph["13"] = {"class_type": "CLIP Input Switch", "inputs": {}}
    graph["108"] = {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "a.safetensors", "model": ["10", 0], "clip": ["13", 1]},
    }
    graph["109"] = {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "b.safetensors", "model": ["108", 0], "clip": ["108", 1]},
    }
    graph["30"]["inputs"]["model"] = ["109", 0]
    graph["20"]["inputs"]["clip"] = ["109", 1]
    graph["21"]["inputs"]["clip"] = ["109", 1]
    return graph


def test_two_lora_chain_is_rebuilt_in_order():
    spec = {**BASE_SPEC, "lora1_node": "108", "lora2_node": "109"}
    api = patched(
        {"lora1": "a2.safetensors", "lora2": "b2.safetensors", "lora_strength": 0.6},
        spec,
        lora_graph(),
    )
    assert api["108"]["inputs"]["model"] == ["10", 0]
    assert api["108"]["inputs"]["clip"] == ["13", 0]
    assert api["109"]["inputs"]["model"] == ["108", 0]
    assert api["109"]["inputs"]["clip"] == ["108", 1]
    assert api["30"]["inputs"]["model"] == ["109", 0]
    assert api["20"]["inputs"]["clip"] == ["109", 1]
    assert api["108"]["inputs"]["strength_model"] == 0.6


def test_disabled_lora_is_removed_and_the_chain_closed():
    spec = {**BASE_SPEC, "lora1_node": "108", "lora2_node": "109"}
    api = patched({"lora1": "a.safetensors", "lora2": ""}, spec, lora_graph())
    assert "109" not in api
    assert api["30"]["inputs"]["model"] == ["108", 0]
    assert api["20"]["inputs"]["clip"] == ["108", 1]


def test_unified_lora_strength_node():
    spec = {
        **BASE_SPEC,
        "lora1_node": "63",
        "lora2_node": "64",
        "lora_strength_node": "67",
        "model_chain_start": "28",
        "model_chain_end": "11",
    }
    graph = base_graph()
    graph["28"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "u"}}
    graph["67"] = {"class_type": "PrimitiveFloat", "inputs": {"value": 1.0}}
    graph["63"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["28", 0]}}
    graph["64"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["63", 0]}}
    graph["11"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["64", 0]}}
    api = patched({"lora1": "a.safetensors", "lora2": "", "lora_strength": 0.35}, spec, graph)
    assert api["67"]["inputs"]["value"] == 0.35
    assert "64" not in api
    assert api["63"]["inputs"]["model"] == ["28", 0]
    assert api["11"]["inputs"]["model"] == ["63", 0]


def test_single_lora_chain_keeps_the_attention_patch():
    spec = {**BASE_SPEC, "lora1_node": "733", "model_chain_start": "126", "lora": "default"}
    graph = base_graph()
    graph["126"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "u"}}
    graph["733"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["126", 0]}}
    graph["734"] = {"class_type": "SageAttention", "inputs": {"model": ["733", 0]}}
    api = patched({"lora": "mine.safetensors", "lora_strength": 0.5}, spec, graph)
    assert api["733"]["inputs"]["lora_name"] == "mine.safetensors"
    assert api["733"]["inputs"]["strength_model"] == 0.5


def test_blank_lora_name_removes_the_single_loader():
    spec = {**BASE_SPEC, "lora1_node": "733", "model_chain_start": "126", "lora": "default"}
    graph = base_graph()
    graph["126"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "u"}}
    graph["733"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["126", 0]}}
    graph["734"] = {"class_type": "SageAttention", "inputs": {"model": ["733", 0]}}
    api = patched({"lora": "none"}, spec, graph)
    assert "733" not in api
    assert api["734"]["inputs"]["model"] == ["126", 0]


def test_resolution_selector_and_quality():
    spec = {
        **BASE_SPEC,
        "resolution_node": "37",
        "quality_node": "36",
        "megapixels_key": "megapixels",
        "aspect_ratio_key": "aspect_ratio",
    }
    graph = base_graph()
    graph["37"] = {
        "class_type": "ResolutionSelector",
        "inputs": {"megapixels": 1, "aspect_ratio": "1:1"},
    }
    graph["36"] = {"class_type": "CustomCombo", "inputs": {"choice": "Default"}}
    api = patched({"megapixels": 2, "aspect_ratio": "16:9", "quality": "Quality"}, spec, graph)
    assert api["37"]["inputs"] == {"megapixels": 2, "aspect_ratio": "16:9"}
    assert api["36"]["inputs"]["choice"] == "Quality"


def test_dynamic_combo_nodes_get_the_resize_mode():
    spec = {
        **BASE_SPEC,
        "megapixels_nodes": ["17"],
        "megapixels_key": "resolution.megapixels",
        "megapixels_combo": ["17", "resolution", "resize"],
    }
    graph = base_graph()
    graph["17"] = {
        "class_type": "ReferenceMultiLatent",
        "inputs": {"resolution.megapixels": 1, "resolution": "keep"},
    }
    api = patched({"megapixels": 3}, spec, graph)
    assert api["17"]["inputs"]["resolution.megapixels"] == 3
    assert api["17"]["inputs"]["resolution"] == "resize"


def test_scale_factor():
    spec = {**BASE_SPEC, "scale_node": "2", "scale_key": "scale_by"}
    graph = base_graph()
    graph["2"] = {"class_type": "UpscaleImage", "inputs": {"scale_by": 1}}
    assert patched({"scale": 2}, spec, graph)["2"]["inputs"]["scale_by"] == 2
    assert patched({"scale": 1.5}, spec, graph)["2"]["inputs"]["scale_by"] == 1.5


def test_scale_as_absolute_resolution():
    spec = {
        **BASE_SPEC,
        "scale_node": "12",
        "scale_keys": ["resolution", "max_resolution"],
        "scale_mode": "resolution",
    }
    graph = base_graph()
    graph["12"] = {
        "class_type": "SeedVR2LoadDiTModel",
        "inputs": {"resolution": 0, "max_resolution": 0},
    }
    api = patched({"scale": 2, "input_longest_side": 900}, spec, graph)
    assert api["12"]["inputs"]["resolution"] == 1800
    assert api["12"]["inputs"]["max_resolution"] == 1800


def test_absolute_scale_requires_input_dimensions():
    spec = {**BASE_SPEC, "scale_node": "12", "scale_mode": "resolution"}
    with pytest.raises(ValueError):
        patched({"scale": 2}, spec)


def test_encode_resolution_budget():
    spec = {**BASE_SPEC, "encode_resolution_node": "725"}
    graph = base_graph()
    graph["725"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"resolution": 1376}}
    assert patched({"resolution": 1024}, spec, graph)["725"]["inputs"]["resolution"] == 1024


def test_missing_spec_nodes_are_ignored():
    # A spec key that is None must be a no-op, not a KeyError.
    spec = {**BASE_SPEC, "image_node": None, "denoise_node": None}
    assert patched({"image_filename": "a.png", "strength": 0.5}, spec) == base_graph()


def test_unknown_kwargs_are_ignored():
    assert patched({"unrelated": "value"}) == base_graph()


def test_current_model_prefers_the_spec_node():
    graph = base_graph()
    assert workflow_current_model(graph, {"model_node": "11"}) == "orig-unet.safetensors"


def test_current_model_falls_back_to_the_first_loader():
    # No model_node (e.g. SDXL): the first known loader class wins.
    assert workflow_current_model(base_graph(), {}) == "orig.safetensors"


def test_current_model_returns_none_when_the_graph_has_no_loader():
    graph = {"30": {"class_type": "KSampler", "inputs": {}}}
    assert workflow_current_model(graph, {}) is None
