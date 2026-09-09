from __future__ import annotations

import unittest

from feverslop.prompting.general_signatures import parse_prompt_result


class PromptResultParsingTests(unittest.TestCase):
    def test_unwraps_dspy_result_envelope(self):
        result = parse_prompt_result({
            "result": {
                "prompt": "A woman crosses the glass chamber.",
                "vocal_performers": [],
            },
        })

        self.assertEqual("A woman crosses the glass chamber.", result.prompt)
        self.assertEqual([], result.vocal_performers)

    def test_normalizes_video_prompt_and_subject_only_vocal_performers(self):
        result = parse_prompt_result({
            "video_prompt": "Slowly, the lead subject turns toward the camera.",
            "vocal_performers": ["lead_subject_01"],
        })

        self.assertEqual("Slowly, the lead subject turns toward the camera.", result.prompt)
        self.assertEqual(
            [{"subject_id": "lead_subject_01", "speaker_id": "S1"}],
            [performer.model_dump() for performer in result.vocal_performers],
        )

    def test_normalizes_single_text_field_and_subject_mapping_without_speaker(self):
        result = parse_prompt_result({
            "generated_motion": "The lead subject slowly turns toward the camera.",
            "vocal_performers": [{"subject_id": "lead_subject_01"}],
        })

        self.assertEqual("The lead subject slowly turns toward the camera.", result.prompt)
        self.assertEqual(
            [{"subject_id": "lead_subject_01", "speaker_id": "S1"}],
            [performer.model_dump() for performer in result.vocal_performers],
        )


if __name__ == "__main__":
    unittest.main()
