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

    def test_remove_node_raises_on_dangling_wire_ref(self):
        """Removing a node that feeds an unbridged consumer raises."""
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "test.png"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"image": ["1", 0], "steps": 20},
            },
        }
        patcher = WorkflowPatcher(workflow)
        spec = [
            {
                "op": "remove_node",
                "target": {"node_id": "1"},
                # no bridge entries — node 2 still references node 1
            },
        ]
        with self.assertRaises(ValueError) as ctx:
            patcher.apply_patch_spec(spec)
        error_msg = str(ctx.exception)
        self.assertIn("dangling", error_msg.lower())
        self.assertIn("2", error_msg)
        # Node should still exist
        self.assertIn("1", patcher.workflow)

    def test_remove_node_all_consumers_bridged_succeeds(self):
        """Removal succeeds when bridge covers all downstream references."""
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "test.png"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"image": ["1", 0], "steps": 20},
            },
            "3": {
                "class_type": "LoadImage",
                "inputs": {"image": "other.png"},
            },
        }
        patcher = WorkflowPatcher(workflow)
        spec = [
            {
                "op": "remove_node",
                "target": {"node_id": "1"},
                "bridge": [
                    {
                        "from_input": "image",  # from LoadImage's input
                        "to": {"node_id": "2", "input": "image"},
                    },
                ],
            },
        ]
        patcher.apply_patch_spec(spec)
        self.assertNotIn("1", patcher.workflow)

    def test_remove_leaf_node_no_consumers(self):
        """Removing a leaf node with no downstream references succeeds."""
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "test.png"},
            },
            "2": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "output"},
            },
        }
        patcher = WorkflowPatcher(workflow)
        spec = [
            {
                "op": "remove_node",
                "target": {"node_id": "1"},
            },
        ]
        patcher.apply_patch_spec(spec)
        self.assertNotIn("1", patcher.workflow)
        self.assertIn("2", patcher.workflow)

if __name__ == "__main__":
    unittest.main()
