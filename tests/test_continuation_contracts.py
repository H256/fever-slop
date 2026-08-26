import unittest

from feverslop.domain.audio_timing_contract import AudioTimingWindow, validate_audio_timing_windows
from feverslop.domain.continuation_intent import continuation_intents_from_plan
from feverslop.domain.cutless_boundaries import CutlessBoundary, validate_cutless_chain
from feverslop.prompting.creative_field_repair import repair_creative_fields


class ContinuationContractsTests(unittest.TestCase):
    def test_planner_intent_and_absolute_audio_windows(self):
        intent = continuation_intents_from_plan([{"action_id": "a", "requires_continuation": True}])[0]
        self.assertTrue(intent.requires_continuation)
        validate_audio_timing_windows([AudioTimingWindow(0, 4), AudioTimingWindow(4, 8)], song_duration=8)

    def test_cutless_chain_and_field_scoped_repair(self):
        digest = "a" * 64
        validate_cutless_chain([CutlessBoundary("s1", "s2", digest)], ["s1", "s2"])
        self.assertEqual({"camera": "pan", "action": "walk"}, repair_creative_fields(
            {"camera": "shake", "action": "walk"}, ["camera"], {"camera": "pan"}))


if __name__ == "__main__":
    unittest.main()
