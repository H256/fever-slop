import unittest
import json
from pathlib import Path

from feverslop.adapters.sequence_to_sheet_workflow import (
    LTX_SEQUENCE_TO_SHEET_PROFILE,
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
    def test_ltx_profile_accepts_positive_prompt_fallback(self):
        workflow = workflow_with_titles(
            "#WIDTH",
            "#HEIGHT",
            "#STARTFRAME",
            "#FRAMES",
            "#FRAMERATE",
            "#PROMPT_POSITIVE",
            "#SEED",
            "#SAVE_VIDEO",
            include_vram_cleanup=True,
        )

        self.assertEqual((), LTX_SEQUENCE_TO_SHEET_PROFILE.validate(workflow))

    def test_minimax_profile_requires_all_reference_slots(self):
        workflow = workflow_with_titles(
            "#MEGAPIXELS",
            "#R2V_COMBINE",
            "#PROMPT",
            "#SEED",
            "#FRAMECOUNT",
            *(f"#REF_{index}" for index in range(1, 10)),
            "#SAVE_VIDEO",
            include_vram_cleanup=True,
        )

        self.assertEqual((), MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE.validate(workflow))

    def test_missing_titles_are_reported_with_exact_anchor_names(self):
        missing = LTX_SEQUENCE_TO_SHEET_PROFILE.validate(workflow_with_titles("#WIDTH"))

        self.assertIn("#HEIGHT", missing)
        self.assertIn("#STARTFRAME", missing)
        self.assertIn("#PROMPT_POSITIVE or #PROMPT", missing)
        self.assertIn("#SAVE_VIDEO", missing)

    def test_profiles_require_a_vram_cleanup_node(self):
        workflow = workflow_with_titles(
            "#WIDTH",
            "#HEIGHT",
            "#STARTFRAME",
            "#FRAMES",
            "#FRAMERATE",
            "#PROMPT_POSITIVE",
            "#SEED",
            "#SAVE_VIDEO",
        )

        missing = LTX_SEQUENCE_TO_SHEET_PROFILE.validate(workflow)

        self.assertIn("class_type:VRAMCleanup", missing)

    def test_ltx_reference_workflow_is_patcher_compatible_and_audio_independent(self):
        workflow_path = Path(__file__).parents[1] / "workflows" / "sequence_to_sheet_ltx_v1.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

        self.assertEqual((), LTX_SEQUENCE_TO_SHEET_PROFILE.validate(workflow))
        self.assertFalse(any(node.get("class_type") == "LoadAudio" for node in workflow.values()))
        self.assertGreaterEqual(
            sum(node.get("class_type") == "VRAMCleanup" for node in workflow.values()),
            2,
        )
        self.assertTrue(any(node.get("class_type") == "LTXVEmptyLatentAudio" for node in workflow.values()))

    def test_minimax_reference_workflow_is_patcher_compatible_without_external_media_inputs(self):
        workflow_path = Path(__file__).parents[1] / "workflows" / "sequence_to_sheet_minimax_h3_v1.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

        self.assertEqual((), MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE.validate(workflow))
        self.assertFalse(any(node.get("class_type") == "LoadAudio" for node in workflow.values()))
        self.assertFalse(any(node.get("class_type") == "LoadVideo" for node in workflow.values()))
        self.assertGreaterEqual(
            sum(node.get("class_type") == "VRAMCleanup" for node in workflow.values()),
            1,
        )


if __name__ == "__main__":
    unittest.main()
