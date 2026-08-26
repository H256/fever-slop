import unittest

from feverslop.domain.audio_timing_contract import AudioTimingWindow, validate_audio_timing_windows
from feverslop.domain.continuation_intent import continuation_intents_from_plan
from feverslop.domain.cutless_boundaries import CutlessBoundary, validate_cutless_chain
from feverslop.prompting.dspy_h3_models import PromptPlan
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

    def test_prompt_plan_carries_semantic_continuation_intent(self):
        plan = PromptPlan(
            creative_intent="one continuous action",
            overall_soundscape="wind",
            music_intent="none",
            continuation_intents=[{
                "action_id": "raise-lantern",
                "requires_continuation": True,
                "desired_duration_seconds": 24.0,
            }],
        )

        intent = plan.continuation_intents[0]
        self.assertEqual("raise-lantern", intent.action_id)
        self.assertTrue(intent.requires_continuation)
        self.assertEqual(24.0, intent.desired_duration_seconds)

    def test_prompt_plan_preserves_explicit_hard_cut_as_non_continuation(self):
        plan = PromptPlan(
            creative_intent="two shots",
            overall_soundscape="silence",
            music_intent="none",
            shots=[{
                "shot_number": 1,
                "description": "The door opens.",
                "hard_cut_after": True,
            }],
            continuation_intents=[{
                "action_id": "door-open",
                "requires_continuation": False,
            }],
        )

        self.assertTrue(plan.shots[0].hard_cut_after)
        self.assertFalse(plan.continuation_intents[0].requires_continuation)


if __name__ == "__main__":
    unittest.main()
