"""CI check: every checked-in ComfyUI workflow must be importable.

The per-project workflow import tool loads a workflow through
``_load_graph`` (JSON parse, object shape, duplicate node IDs, size/nesting
limits). A hand-maintained ``workflows/*.json`` whose schema drifts from that
loader would only surface when a user imports it. This test fails at commit
time by running every checked-in workflow graph through the same loader.

Metadata files (``.profile.json`` sidecars, ``capabilities.json``,
``profile-matrix.json``) are not workflow graphs and are skipped.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from feverslop.domain.workflow_import_inspector import _load_graph

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1] / "workflows"


def _is_graph(obj: object) -> bool:
    """A ComfyUI workflow graph maps node ids to nodes with a class_type."""
    return (
        isinstance(obj, dict)
        and any(isinstance(v, dict) and "class_type" in v for v in obj.values())
    )


def _workflow_graphs() -> list[Path]:
    graphs = []
    for path in sorted(WORKFLOWS_ROOT.rglob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if _is_graph(obj):
            graphs.append(path)
    return graphs


class CheckedInWorkflowsSchemaTests(unittest.TestCase):
    def test_every_checked_in_workflow_is_importable(self):
        graphs = _workflow_graphs()
        self.assertGreater(
            len(graphs),
            0,
            "no workflow graphs found under workflows/; check the layout",
        )
        for path in graphs:
            with self.subTest(workflow=path):
                _load_graph(path.read_text(encoding="utf-8-sig"))

    def test_loader_rejects_duplicate_node_ids(self):
        # An int key and the string key that normalize to the same id collide.
        broken = {1: {"class_type": "A", "inputs": {}}, "1": {"class_type": "B", "inputs": {}}}
        with self.assertRaises(ValueError):
            _load_graph(broken)

    def test_loader_rejects_non_object_graph(self):
        with self.assertRaises(TypeError):
            _load_graph([1, 2, 3])
