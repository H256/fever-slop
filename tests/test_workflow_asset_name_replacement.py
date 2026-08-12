import unittest

from feverslop.adapters.prepared_workflow import (
    _replace_asset_names_in_workflow,
)


class TestReplaceAssetNames(unittest.TestCase):
    """Verify that asset name replacement is scoped to node inputs only."""

    def setUp(self):
        self.replacements = {
            "old_image.png": "new_image.png",
            "old_video.mp4": "new_video.mp4",
        }

    def test_replaces_asset_name_in_input(self):
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "old_image.png"},
            },
        }
        result = _replace_asset_names_in_workflow(workflow, self.replacements)
        self.assertEqual(result["1"]["inputs"]["image"], "new_image.png")

    def test_does_not_replace_class_type(self):
        workflow = {
            "1": {
                "class_type": "old_image.png",
                "inputs": {"seed": 42},
            },
        }
        result = _replace_asset_names_in_workflow(workflow, self.replacements)
        self.assertEqual(result["1"]["class_type"], "old_image.png")

    def test_does_not_replace_meta_title(self):
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "_meta": {"title": "old_image.png"},
                "inputs": {"image": "other.png"},
            },
        }
        result = _replace_asset_names_in_workflow(workflow, self.replacements)
        self.assertEqual(result["1"]["_meta"]["title"], "old_image.png")

    def test_does_not_replace_wire_reference(self):
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"conditioning": ["old_image.png", 0]},
            },
        }
        result = _replace_asset_names_in_workflow(workflow, self.replacements)
        self.assertEqual(result["1"]["inputs"]["conditioning"], ["old_image.png", 0])

    def test_replaces_multiple_inputs(self):
        workflow = {
            "1": {
                "class_type": "DualInput",
                "inputs": {
                    "image": "old_image.png",
                    "video": "old_video.mp4",
                },
            },
        }
        result = _replace_asset_names_in_workflow(workflow, self.replacements)
        self.assertEqual(result["1"]["inputs"]["image"], "new_image.png")
        self.assertEqual(result["1"]["inputs"]["video"], "new_video.mp4")

    def test_passes_through_non_string_values(self):
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.5,
                    "enabled": True,
                },
            },
        }
        result = _replace_asset_names_in_workflow(workflow, {})
        self.assertEqual(result["1"]["inputs"]["seed"], 42)
        self.assertEqual(result["1"]["inputs"]["steps"], 20)
        self.assertEqual(result["1"]["inputs"]["cfg"], 7.5)
        self.assertEqual(result["1"]["inputs"]["enabled"], True)

    def test_replaces_in_nested_dict_within_inputs(self):
        workflow = {
            "1": {
                "class_type": "CustomNode",
                "inputs": {
                    "config": {"file": "old_image.png"},
                },
            },
        }
        result = _replace_asset_names_in_workflow(workflow, self.replacements)
        self.assertEqual(result["1"]["inputs"]["config"]["file"], "new_image.png")

    def test_leaves_long_prompt_unchanged(self):
        """An exact match only replaces the entire string, not a substring."""
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "a photo of old_image.png and other stuff",
                },
            },
        }
        result = _replace_asset_names_in_workflow(workflow, self.replacements)
        self.assertEqual(
            result["1"]["inputs"]["text"],
            "a photo of old_image.png and other stuff",
        )

    def test_empty_workflow(self):
        result = _replace_asset_names_in_workflow({}, self.replacements)
        self.assertEqual(result, {})

    def test_no_replacements(self):
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "photo.png"},
            },
        }
        result = _replace_asset_names_in_workflow(workflow, {})
        self.assertEqual(result["1"]["inputs"]["image"], "photo.png")
        # Original not mutated
        self.assertIsNot(result, workflow)
        self.assertIsNot(result["1"], workflow["1"])
        self.assertIsNot(result["1"]["inputs"], workflow["1"]["inputs"])


if __name__ == "__main__":
    unittest.main()
