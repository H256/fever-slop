import json
import unittest
from pathlib import Path


class LTXMSRWorkflowFileTests(unittest.TestCase):
    def test_api_workflow_matches_original_loaders_and_msr_anchors(self):
        workflow_path = Path("workflows/video_ltxv_msr_1actor_1background_v1.json")
        workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))

        self.assertNotIn("nodes", workflow)
        self.assertEqual("UnetLoaderGGUF", workflow["1"]["class_type"])
        self.assertEqual("LTX-2.3-22B-distilled-1.1-Q6_K.gguf", workflow["1"]["inputs"]["unet_name"])
        self.assertEqual("VAELoaderKJ", workflow["2"]["class_type"])
        self.assertEqual("DualCLIPLoaderGGUF", workflow["3"]["class_type"])

        titles = {
            node.get("_meta", {}).get("title")
            for node in workflow.values()
            if isinstance(node, dict)
        }
        self.assertNotIn("#STARTFRAME", titles)
        for title in (
            "#MSR_ACTOR_1",
            "#MSR_BACKGROUND",
            "#MSR_FRAME_COUNT",
            "#PROMPT_RELAY",
            "#FRAMES",
            "#FRAMERATE",
            "#SAVE_VIDEO",
        ):
            self.assertIn(title, titles)

        msr_node = next(node for node in workflow.values() if node.get("_meta", {}).get("title") == "#MSR_FRAME_COUNT")
        self.assertEqual("LiconMSR", msr_node["class_type"])
        self.assertEqual(["14", 0], msr_node["inputs"]["1"])
        self.assertEqual(["15", 0], msr_node["inputs"]["background"])

        guide_node = next(node for node in workflow.values() if node.get("_meta", {}).get("title") == "#MSR_GUIDE")
        self.assertEqual("LTXAddVideoICLoRAGuide", guide_node["class_type"])
        self.assertIn("tile_overlap", guide_node["inputs"])
        self.assertNotIn("overlap", guide_node["inputs"])

        relay_node = next(node for node in workflow.values() if node.get("_meta", {}).get("title") == "#PROMPT_RELAY")
        self.assertEqual("PromptRelayEncode", relay_node["class_type"])
        self.assertIn("global_prompt", relay_node["inputs"])
        self.assertIn("local_prompts", relay_node["inputs"])
        self.assertIn("segment_lengths", relay_node["inputs"])
