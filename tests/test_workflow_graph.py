"""Tests for the graph-format -> API-format converter."""

import json
from pathlib import Path

import pytest

from workflow import graph_to_api

WORKFLOW_FILES = sorted((Path(__file__).resolve().parent.parent / "workflows").rglob("*.json"))


def _graph_only(path: Path) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if "nodes" in data else None


def test_named_widgets_win_over_positional_values():
    graph = {
        "nodes": [
            {
                "id": 1,
                "type": "CLIPTextEncode",
                "inputs": [{"name": "text", "widget": {"name": "text"}, "link": None}],
                "widgets_values": ["positional"],
                "widgets_values_named": {"text": "named"},
            }
        ]
    }
    assert graph_to_api(graph)["1"]["inputs"]["text"] == "named"


def test_positional_widgets_are_consumed_in_order():
    graph = {
        "nodes": [
            {
                "id": 2,
                "type": "KSampler",
                "inputs": [
                    {"name": "seed", "widget": {"name": "seed"}, "link": None},
                    {"name": "steps", "widget": {"name": "steps"}, "link": None},
                ],
                "widgets_values": [42, 20],
            }
        ]
    }
    assert graph_to_api(graph)["2"]["inputs"] == {"seed": 42, "steps": 20}


def test_links_override_widget_values():
    graph = {
        "links": [[7, 1, 0, 2, 0, "MODEL"]],
        "nodes": [
            {
                "id": 2,
                "type": "KSampler",
                "inputs": [
                    {"name": "model", "type": "MODEL", "widget": {"name": "model"}, "link": 7}
                ],
                "widgets_values": ["stale-widget-value"],
            }
        ],
    }
    assert graph_to_api(graph)["2"]["inputs"]["model"] == ["1", 0]


def test_note_nodes_are_dropped():
    graph = {
        "nodes": [
            {"id": 1, "type": "Note", "inputs": [], "widgets_values": ["read me"]},
            {"id": 2, "type": "MarkdownNote", "inputs": [], "widgets_values": ["me too"]},
            {"id": 3, "type": "EmptyLatentImage", "inputs": [], "widgets_values": [512, 512, 1]},
        ]
    }
    assert set(graph_to_api(graph)) == {"3"}


def test_node_ids_are_strings_and_links_resolve():
    graph = {
        "links": [[1, 10, 0, 20, 0, "MODEL"]],
        "nodes": [
            {"id": 10, "type": "CheckpointLoaderSimple", "inputs": [], "widgets_values": ["m"]},
            {
                "id": 20,
                "type": "KSampler",
                "inputs": [{"name": "model", "link": 1}],
                "widgets_values": [1, 20, 7.0],
            },
        ],
    }
    api = graph_to_api(graph)
    assert api["20"]["inputs"]["model"] == ["10", 0]
    assert api["20"]["class_type"] == "KSampler"


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.stem)
def test_shipped_workflows_convert_cleanly(path: Path):
    graph = _graph_only(path)
    if graph is None:
        pytest.skip("API-format workflow: nothing to convert")

    api = graph_to_api(graph)
    assert api, "converter produced an empty graph"

    link_targets = {
        str(node["id"]) for node in graph["nodes"] if node["type"] not in ("Note", "MarkdownNote")
    }
    assert set(api) == link_targets
    for node_id, node in api.items():
        assert node["class_type"] not in ("Note", "MarkdownNote")
        for name, value in node["inputs"].items():
            is_link = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            if is_link:
                assert value[0] in api, f"node {node_id} input {name} links to missing {value[0]}"
