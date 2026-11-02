"""Tests for the runtime helpers around the spec patchers."""

import asyncio

import workflow


def test_clamp_kwargs_bounds_user_values():
    kwargs = {"steps": 999, "cfg": 0.1, "width": 32, "seed": 5}
    workflow._clamp_kwargs(kwargs)

    assert kwargs["steps"] == 150
    assert kwargs["cfg"] == 0.5
    assert kwargs["width"] == 64
    # Unbounded and absent keys are untouched.
    assert kwargs["seed"] == 5
    assert "height" not in kwargs


def test_run_text_workflow_applies_patches(monkeypatch):
    recorded = {}

    class FakeComfy:
        async def queue_prompt(self, api_workflow):
            recorded["workflow"] = api_workflow
            return "prompt-id", "client-id"

        async def wait_for_output_text(self, prompt_id, target_node=None):
            recorded["prompt_id"] = prompt_id
            recorded["target"] = target_node
            return "generated text"

    monkeypatch.setattr(workflow, "comfy", FakeComfy())
    monkeypatch.setattr(
        workflow,
        "load_workflow",
        lambda path: {"7": {"class_type": "SomeNode", "inputs": {"value": 1}}},
    )

    result = asyncio.run(
        workflow.run_text_workflow("unused.json", {"7": {"value": 42}}, target_node="7")
    )

    assert result == "generated text"
    assert recorded["workflow"]["7"]["inputs"]["value"] == 42
    assert recorded["prompt_id"] == "prompt-id"
    assert recorded["target"] == "7"
