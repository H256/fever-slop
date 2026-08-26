import json
import unittest
from pathlib import Path

from feverslop.adapters.sequence_to_sheet_workflow import (
    MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE,
)


def workflow_with_titles(*titles: str, include_vram_cleanup: bool = False) -> dict:
    workflow = {
        str(index): {"inputs": {}, "_meta": {"title": title}}
        for index, title in enumerate(titles, start=1)
    }
    if include_vram_cleanup:
        workflow["vram"] = {"inputs": {}, "class_type": "VRAMCleanup"}
    return workflow


class SequenceToSheetWorkflowProfileTests(unittest.TestCase):
    def test_minimax_profile_requires_i2va_anchors(self):
        workflow = workflow_with_titles(
            "#MEGAPIXELS",
            "#PROMPT",
            "#SEED",
            "#FRAMECOUNT",
            "#STARTFRAME",
            "#TURBO_LORA",
            "#SAVE_VIDEO",
            include_vram_cleanup=True,
        )

        self.assertEqual((), MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE.validate(workflow))

    def test_missing_titles_are_reported_with_exact_anchor_names(self):
        missing = MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE.validate(workflow_with_titles("#MEGAPIXELS"))

        self.assertIn("#STARTFRAME", missing)
        self.assertIn("#TURBO_LORA", missing)
        self.assertIn("#SAVE_VIDEO", missing)

    def test_profiles_require_a_vram_cleanup_node(self):
        workflow = workflow_with_titles("#MEGAPIXELS")

        missing = MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE.validate(workflow)

        self.assertIn("class_type:VRAMCleanup", missing)

    def test_minimax_reference_workflow_is_patcher_compatible_without_external_media_inputs(self):
        workflow_path = Path(__file__).parents[1] / "workflows" / "sequence" / "minimax_h3" / "sequence_to_sheet_minimax_h3_i2va_v1.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

        self.assertEqual((), MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE.validate(workflow))
        self.assertFalse(any(node.get("class_type") == "LoadAudio" for node in workflow.values()))
        self.assertFalse(any(node.get("class_type") == "LoadVideo" for node in workflow.values()))
        self.assertTrue(any(node.get("class_type") == "MiniMaxH3ImageToVideo" for node in workflow.values()))
        self.assertTrue(any(node.get("class_type") == "LoraLoaderModelOnly" for node in workflow.values()))
        self.assertGreaterEqual(
            sum(node.get("class_type") == "VRAMCleanup" for node in workflow.values()),
            1,
        )


if __name__ == "__main__":
    unittest.main()
