from collections import Counter
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ("video_ltxv_ingredients_2stage_v5.json", False, False),
    ("video_ltxv_ingredients_audio_2stage_v5.json", True, False),
    ("video_ltxv_ingredients_2stage_gguf_v5.json", False, True),
    ("video_ltxv_ingredients_audio_2stage_gguf_v5.json", True, True),
)
ARCHIVED_V4_WORKFLOWS = (
    "video_ltxv_ingredients_2stage_v4.json",
    "video_ltxv_ingredients_audio_2stage_v4.json",
    "video_ltxv_ingredients_2stage_gguf_v4.json",
    "video_ltxv_ingredients_audio_2stage_gguf_v4.json",
)
COMMON_ANCHORS = {
    "#INGREDIENTS",
    "#PROMPT_NEGATIVE",
    "#SEED",
    "#WIDTH",
    "#HEIGHT",
    "#FRAMES",
    "#FRAMERATE",
    "#SAVE_VIDEO",
}


class IngredientsWorkflowFileTests(unittest.TestCase):
    def test_workflows_preserve_anchor_and_two_stage_graph_contracts(self):
        for name, has_input_audio, is_gguf in WORKFLOWS:
            with self.subTest(workflow=name):
                workflow = self._load(name)
                titles = Counter(
                    node.get("_meta", {}).get("title")
                    for node in workflow.values()
                    if str(node.get("_meta", {}).get("title", "")).startswith("#")
                )
                required_anchors = COMMON_ANCHORS | {"#PROMPT_RELAY"} | (
                    {"#LOAD_AUDIO", "#TRIM_AUDIO"} if has_input_audio else set()
                )
                self.assertTrue(required_anchors.issubset(titles))
                self.assertTrue(all(titles[title] == 1 for title in required_anchors))
                self._assert_links_exist(workflow)
                self._assert_image_and_framerate_connections(workflow)
                self._assert_cleanup_boundary(workflow, sampler_id="4829")
                self._assert_cleanup_boundary(workflow, sampler_id="5207")
                self._assert_prompt_relay_graph(workflow)
                if is_gguf:
                    self._assert_gguf_loaders(workflow)

    def test_v4_workflows_are_archived_without_root_aliases(self):
        for name in ARCHIVED_V4_WORKFLOWS:
            with self.subTest(workflow=name):
                self.assertFalse((ROOT / "workflows" / name).exists())
                self.assertTrue((ROOT / "workflows" / "old" / name).is_file())

    def _assert_prompt_relay_graph(self, workflow):
        relay_id, relay = self._only_item(workflow, title="#PROMPT_RELAY")
        self.assertEqual("PromptRelayEncode", relay["class_type"])
        self.assertFalse(any(
            node.get("_meta", {}).get("title") == "#PROMPT_POSITIVE"
            for node in workflow.values()
        ))
        conditioning = self._only_node(workflow, class_type="LTXVConditioning")
        nag = self._only_node(workflow, class_type="LTX2_NAG")
        self.assertEqual([relay_id, 1], conditioning["inputs"]["positive"])
        self.assertEqual([relay_id, 0], nag["inputs"]["model"])
        for input_name in ("model", "clip", "latent"):
            self.assertTrue(self._is_link(relay["inputs"][input_name]))
        latent_id, _ = relay["inputs"]["latent"]
        self.assertEqual("EmptyLTXVLatentVideo", workflow[latent_id]["class_type"])

    @staticmethod
    def _load(name):
        return json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))

    def _assert_links_exist(self, workflow):
        node_ids = set(workflow)
        for node_id, node in workflow.items():
            for input_name, value in node.get("inputs", {}).items():
                if self._is_link(value):
                    self.assertIn(value[0], node_ids, f"{node_id}.{input_name} has a dangling link")

    def _assert_image_and_framerate_connections(self, workflow):
        preprocess = self._only_node(workflow, class_type="LTXVPreprocess")
        resize_id, _ = preprocess["inputs"]["image"]
        self.assertEqual("ResizeImageMaskNode", workflow[resize_id]["class_type"])

        framerate_id, framerate = self._only_item(workflow, title="#FRAMERATE")
        self.assertEqual("PrimitiveFloat", framerate["class_type"])
        float_to_int = self._only_node(workflow, class_type="LTXFloatToInt")
        save_video = self._only_node(workflow, title="#SAVE_VIDEO")
        self.assertEqual([framerate_id, 0], float_to_int["inputs"]["a"])
        self.assertEqual([framerate_id, 0], save_video["inputs"]["frame_rate"])

    def _assert_cleanup_boundary(self, workflow, *, sampler_id):
        cleanup_id, cleanup = next(
            (node_id, node)
            for node_id, node in workflow.items()
            if node["class_type"] == "VRAMCleanup"
            and node["inputs"].get("anything") == [sampler_id, 0]
        )
        consumers = [
            node
            for node in workflow.values()
            if any(value == [cleanup_id, 0] for value in node.get("inputs", {}).values())
        ]
        self.assertEqual(1, len(consumers))
        self.assertEqual("LTXVSeparateAVLatent", consumers[0]["class_type"])
        self.assertTrue(cleanup["inputs"]["offload_model"])
        self.assertTrue(cleanup["inputs"]["offload_cache"])

    def _assert_gguf_loaders(self, workflow):
        classes = {node["class_type"] for node in workflow.values()}
        self.assertIn("UnetLoaderGGUF", classes)
        self.assertIn("DualCLIPLoaderGGUF", classes)
        self.assertNotIn("CheckpointLoaderSimple", classes)
        self.assertNotIn("LoraLoaderModelOnly", classes)
        vae_names = {
            node["inputs"]["vae_name"]
            for node in workflow.values()
            if node["class_type"] == "VAELoaderKJ"
        }
        self.assertEqual(
            {"LTX23_video_vae_bf16.safetensors", "LTX23_audio_vae_bf16.safetensors"},
            vae_names,
        )

    @staticmethod
    def _is_link(value):
        return (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
        )

    @staticmethod
    def _only_node(workflow, *, class_type=None, title=None):
        _, node = IngredientsWorkflowFileTests._only_item(
            workflow, class_type=class_type, title=title
        )
        return node

    @staticmethod
    def _only_item(workflow, *, class_type=None, title=None):
        matches = [
            (node_id, node)
            for node_id, node in workflow.items()
            if (class_type is None or node["class_type"] == class_type)
            and (title is None or node.get("_meta", {}).get("title") == title)
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected one node for class_type={class_type!r}, title={title!r}; got {len(matches)}"
            )
        return matches[0]


if __name__ == "__main__":
    unittest.main()
