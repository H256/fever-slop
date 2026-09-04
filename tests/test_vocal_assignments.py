from __future__ import annotations

import unittest

from feverslop.domain.vocal_assignments import (
    apply_vocal_assignments,
    build_generated_vocal_assignments,
    infer_vocal_performers,
)


class VocalAssignmentTests(unittest.TestCase):
    def test_preserves_the_llm_selected_singer_as_a_structured_performer(self):
        performers = infer_vocal_performers(
            prompt="Mordren Vale sings with passionate lip sync while the band performs.",
            actors=[
                {"id": "mordren_vale", "name": "Mordren Vale"},
                {"id": "aurelius_vane", "name": "Aurelius Vane"},
            ],
        )

        self.assertEqual(
            [{"subject_id": "mordren_vale", "speaker_id": "S1"}],
            performers,
        )

    def test_preserves_lip_syncing_vocalist_without_the_word_sings(self):
        performers = infer_vocal_performers(
            prompt="Mordren Vale screams toward the sky with passionate lip-syncing.",
            actors=[{"id": "mordren_vale", "name": "Mordren Vale"}],
        )

        self.assertEqual(
            [{"subject_id": "mordren_vale", "speaker_id": "S1"}],
            performers,
        )

    def test_builds_scene_local_windows_from_vocal_relay(self):
        assignments = build_generated_vocal_assignments(
            prompt_relay=[
                {"frame_start": 0, "frame_end": 24, "state": "instrumental"},
                {"frame_start": 24, "frame_end": 72, "state": "singing"},
            ],
            fps=24,
            duration_seconds=3.0,
            performers=[{"subject_id": "soren", "speaker_id": "S1"}],
        )

        self.assertEqual([{
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "performers": [{"subject_id": "soren", "speaker_id": "S1"}],
        }], assignments)

    def test_applies_simultaneous_duet_to_visible_cast_and_relay(self):
        segment = {
            "duration": 4.0,
            "references": {"actor_ids": ["observer"]},
            "ltx": {"prompt_relay": [
                {"frame_start": 0, "frame_end": 96, "state": "singing", "lyrics": "Together"},
            ]},
        }
        assignments = [{
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "performers": [
                {"subject_id": "soren", "speaker_id": "S1"},
                {"subject_id": "tamsin", "speaker_id": "S2"},
            ],
        }]

        result = apply_vocal_assignments(segment, assignments, fps=24)

        self.assertEqual(["observer", "soren", "tamsin"], result["references"]["actor_ids"])
        self.assertEqual(2, len(result["references"]["audio_subject_bindings"]))
        self.assertEqual(
            [
                {"subject_id": "soren", "speaker_id": "S1"},
                {"subject_id": "tamsin", "speaker_id": "S2"},
            ],
            result["ltx"]["prompt_relay"][0]["speaker_bindings"],
        )

    def test_rejects_assignment_outside_scene(self):
        with self.assertRaisesRegex(ValueError, "scene duration"):
            apply_vocal_assignments(
                {"duration": 4.0, "references": {}, "ltx": {"prompt_relay": []}},
                [{
                    "start_seconds": 3.0,
                    "end_seconds": 5.0,
                    "performers": [{"subject_id": "soren", "speaker_id": "S1"}],
                }],
                fps=24,
            )


if __name__ == "__main__":
    unittest.main()
