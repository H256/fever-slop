import unittest

from feverslop.adapters.workflow_patcher import WorkflowPatcher


class WorkflowPatchSpecTests(unittest.TestCase):
    def test_patch_spec_sets_input_from_dotted_context_path(self):
        patcher = WorkflowPatcher({
            "1": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}}
        })

        patcher.apply_patch_spec(
            [{"op": "set_input", "target": {"title": "#PROMPT"}, "input": "text", "value_from": "scene.ltx.prompt"}],
            {"scene": {"ltx": {"prompt": "patched"}}},
        )

        self.assertEqual("patched", patcher.get()["1"]["inputs"]["text"])

    def test_patch_spec_removes_node_and_bridges_explicit_connections(self):
        patcher = WorkflowPatcher({
            "1": {"inputs": {}, "_meta": {"title": "#MODEL"}},
            "2": {"inputs": {"model": ["1", 0]}, "_meta": {"title": "#LORA"}},
            "3": {"inputs": {"model": ["2", 0]}, "_meta": {"title": "#SAMPLER"}},
        })

        patcher.apply_patch_spec(
            [
                {
                    "op": "remove_node",
                    "target": {"title": "#LORA"},
                    "bridge": [{"from_input": "model", "to": {"title": "#SAMPLER", "input": "model"}}],
                }
            ],
            {},
        )

        self.assertNotIn("2", patcher.get())
        self.assertEqual(["1", 0], patcher.get()["3"]["inputs"]["model"])

    def test_patch_spec_inserts_node_between_two_connections(self):
        patcher = WorkflowPatcher({
            "1": {"inputs": {}, "_meta": {"title": "#MODEL"}},
            "3": {"inputs": {"model": ["1", 0]}, "_meta": {"title": "#SAMPLER"}},
        })

        patcher.apply_patch_spec(
            [
                {
                    "op": "insert_node_between",
                    "new_node_id": "2",
                    "node": {"class_type": "LoraLoader", "inputs": {"strength_model": 1.0}, "_meta": {"title": "#LORA"}},
                    "source": {"title": "#MODEL", "output": 0},
                    "target": {"title": "#SAMPLER", "input": "model"},
                    "new_node_input": "model",
                    "new_node_output": 0,
                }
            ],
            {},
        )

        self.assertEqual(["1", 0], patcher.get()["2"]["inputs"]["model"])
        self.assertEqual(["2", 0], patcher.get()["3"]["inputs"]["model"])
