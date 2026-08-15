import unittest

from feverslop.adapters.sequence_to_sheet_workflow import (
    LTX_SEQUENCE_TO_SHEET_PROFILE,
    MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE,
)


def workflow_with_titles(*titles: str) -> dict:
    return {
        str(index): {"inputs": {}, "_meta": {"title": title}}
        for index, title in enumerate(titles, start=1)
    }


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
        )

        self.assertEqual((), MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE.validate(workflow))

    def test_missing_titles_are_reported_with_exact_anchor_names(self):
        missing = LTX_SEQUENCE_TO_SHEET_PROFILE.validate(workflow_with_titles("#WIDTH"))

        self.assertIn("#HEIGHT", missing)
        self.assertIn("#STARTFRAME", missing)
        self.assertIn("#PROMPT_POSITIVE or #PROMPT", missing)
        self.assertIn("#SAVE_VIDEO", missing)


if __name__ == "__main__":
    unittest.main()
