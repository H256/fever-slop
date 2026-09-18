"""Focused tests for the generic, spec-driven workflow inspector/validator."""

from __future__ import annotations

import unittest

from feverslop.domain.workflow_import import WorkflowFamilySpec
from feverslop.domain.workflow_import_inspector import inspect_workflow, validate_workflow


def _spec(
    pipeline: str = "minimax-h3",
    core_class_types: tuple[str, ...] = ("MiniMaxH3Video",),
    required_inputs: tuple[str, ...] = ("audio", "image"),
    output_class_types: tuple[str, ...] = ("VHS_VideoCombine",),
    seed_class_type: str | None = "Seed",
    seed_input: str | None = "seed",
) -> WorkflowFamilySpec:
    return WorkflowFamilySpec(
        pipeline=pipeline,
        core_class_types=core_class_types,
        required_inputs=required_inputs,
        output_class_types=output_class_types,
        seed_class_type=seed_class_type,
        seed_input=seed_input,
    )


def _good_graph() -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {}},
        "2": {
            "class_type": "MiniMaxH3Video",
            "inputs": {"audio": [3, 0], "image": [1, 0]},
        },
        "3": {"class_type": "LoadAudio", "inputs": {}},
        "4": {"class_type": "Seed", "inputs": {"seed": 1}},
        "5": {
            "class_type": "VHS_VideoCombine",
            "inputs": {"images": [2, 0]},
        },
    }


class InspectCompatibilityTests(unittest.TestCase):
    def test_auto_compatible_happy_path(self):
        analysis = inspect_workflow(_good_graph(), spec=_spec())
        self.assertEqual("auto_compatible", analysis.compatibility)
        self.assertIsNotNone(analysis.mapping)
        mapping = analysis.mapping
        self.assertEqual("2", mapping.core_node_id)
        self.assertEqual("5", mapping.output_node_id)
        self.assertEqual(("audio", "image"), mapping.required_inputs)
        self.assertEqual("4", mapping.seed_node_id)

    def test_missing_core_is_unsupported(self):
        graph = _good_graph()
        graph["2"]["class_type"] = "NotACore"
        analysis = inspect_workflow(graph, spec=_spec())
        self.assertEqual("unsupported", analysis.compatibility)
        self.assertTrue(any(i.code == "missing_core" for i in analysis.issues))

    def test_missing_output_is_unsupported(self):
        graph = _good_graph()
        del graph["5"]
        analysis = inspect_workflow(graph, spec=_spec())
        self.assertEqual("unsupported", analysis.compatibility)
        self.assertTrue(any(i.code == "missing_output" for i in analysis.issues))

    def test_core_not_upstream_of_output_needs_confirmation(self):
        graph = {
            "1": {"class_type": "MiniMaxH3Video", "inputs": {"audio": 1, "image": 1}},
            "2": {"class_type": "VHS_VideoCombine", "inputs": {}},
            "3": {"class_type": "Seed", "inputs": {"seed": 1}},
        }
        analysis = inspect_workflow(graph, spec=_spec())
        self.assertEqual("needs_confirmation", analysis.compatibility)
        self.assertIsNone(analysis.mapping)

    def test_missing_required_input_needs_confirmation(self):
        graph = _good_graph()
        del graph["2"]["inputs"]["audio"]
        analysis = inspect_workflow(graph, spec=_spec())
        self.assertEqual("needs_confirmation", analysis.compatibility)
        self.assertTrue(any(i.code == "missing_required_inputs" for i in analysis.issues))

    def test_missing_seed_needs_confirmation(self):
        graph = _good_graph()
        del graph["4"]
        analysis = inspect_workflow(graph, spec=_spec())
        self.assertEqual("needs_confirmation", analysis.compatibility)
        self.assertTrue(any(i.code == "missing_seed" for i in analysis.issues))


class InspectLimitTests(unittest.TestCase):
    def test_rejects_graph_over_node_limit(self):
        graph = {
            str(i): {"class_type": "MiniMaxH3Video", "inputs": {"audio": 1, "image": 1}}
            for i in range(2_001)
        }
        with self.assertRaises(ValueError):
            inspect_workflow(graph, spec=_spec())

    def test_rejects_string_over_byte_limit(self):
        graph = _good_graph()
        graph["1"]["_meta"] = {"title": "x" * (64 * 1024 + 1)}
        with self.assertRaises(ValueError):
            inspect_workflow(graph, spec=_spec())

    def test_rejects_deep_nesting(self):
        graph = _good_graph()
        nested = "leaf"
        for _ in range(40):
            nested = [nested]
        graph["1"]["_meta"] = {"title": nested}
        with self.assertRaises(ValueError):
            inspect_workflow(graph, spec=_spec())

    def test_rejects_non_object(self):
        with self.assertRaises(TypeError):
            inspect_workflow([1, 2, 3], spec=_spec())

    def test_rejects_invalid_json_string(self):
        with self.assertRaises(ValueError):
            inspect_workflow("{not json", spec=_spec())


class ValidateTests(unittest.TestCase):
    def test_valid_when_auto_compatible(self):
        report = validate_workflow(_good_graph(), spec=_spec())
        self.assertTrue(report.valid)
        self.assertEqual((), report.issues)

    def test_invalid_when_unsupported(self):
        graph = _good_graph()
        graph["2"]["class_type"] = "NotACore"
        report = validate_workflow(graph, spec=_spec())
        self.assertFalse(report.valid)

    def test_invalid_when_graph_raises(self):
        with self.assertRaises(TypeError):
            inspect_workflow([1], spec=_spec())
        report = validate_workflow([1], spec=_spec())
        self.assertFalse(report.valid)
        self.assertTrue(any(i.code == "invalid_graph" for i in report.issues))


if __name__ == "__main__":
    unittest.main()
