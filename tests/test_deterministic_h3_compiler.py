import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.deterministic_h3_compiler import (
    DeterministicH3Compiler,
    creative_shots_from_plan,
    validate_creative_shots_against_plan,
)
from feverslop.prompting.dspy_h3_models import ResolvedPromptPlan, MusicIntent, PlannedShot


class DeterministicH3CompilerTests(unittest.TestCase):
    def setUp(self):
        self.facts = LockedSceneFacts.create(
            scene_id="scene-01",
            facts=[
                {"category": "wardrobe", "key": "hero", "value": "silver cloak", "source_id": "cast:hero"},
                {"category": "location", "key": "primary", "value": "ruined gate", "source_id": "location:gate"},
            ],
        )
        self.shots = [
            CreativeShotPayload(
                shot_id="shot-02", visible_action="The lantern rises.", performance="defiant", camera_behavior="slow push",
            ),
            CreativeShotPayload(
                shot_id="shot-01", visible_action="The singer waits.", performance="restrained grief", transition_intent="hold for continuation",
            ),
        ]

    def test_compiles_stable_base_and_reference_sections(self):
        compiler = DeterministicH3Compiler()
        base = compiler.compile(
            mode="base", facts=self.facts, shots=self.shots,
            shot_windows={"shot-01": (0.0, 4.5), "shot-02": (4.5, 9.0)},
            references={"shot-01": ["<Picture 1>"], "shot-02": ["<Picture 1>", "<Audio 1>"]},
        )
        reference = compiler.compile(
            mode="reference", facts=self.facts, shots=list(reversed(self.shots)),
            shot_windows={"shot-02": (4.5, 9.0), "shot-01": (0.0, 4.5)},
            references={"shot-02": ["<Audio 1>", "<Picture 1>"], "shot-01": ["<Picture 1>"]},
        )

        self.assertIn("BASE PROMPT", base)
        self.assertIn("FULL REFERENCE PROMPT", reference)
        self.assertIn("[Shot 1 | 00:00.000-00:04.500]", base)
        self.assertIn("<Picture 1>", reference)
        self.assertEqual(base, compiler.compile(
            mode="base", facts=self.facts, shots=list(reversed(self.shots)),
            shot_windows={"shot-02": (4.5, 9.0), "shot-01": (0.0, 4.5)},
            references={"shot-02": ["<Audio 1>", "<Picture 1>"], "shot-01": ["<Picture 1>"]},
        ))

    def test_enforces_word_budget(self):
        with self.assertRaisesRegex(ValueError, "word budget"):
            DeterministicH3Compiler(max_words=3).compile(
                mode="base", facts=self.facts, shots=self.shots,
                shot_windows={"shot-01": (0.0, 4.5), "shot-02": (4.5, 9.0)},
            )

    def test_converts_resolved_plan_to_backend_neutral_shots(self):
        plan = ResolvedPromptPlan(
            creative_intent="solemn performance",
            shots=[PlannedShot(shot_number=2, start_seconds=2, end_seconds=4, description="The lantern rises.")],
            overall_soundscape="wind",
            music_intent=MusicIntent.NONE,
        )
        shots = creative_shots_from_plan(plan)
        self.assertEqual("shot-0002", shots[0].shot_id)
        self.assertEqual("The lantern rises.", shots[0].visible_action)
        self.assertEqual("solemn performance", shots[0].performance)

    def test_rejects_unknown_or_missing_plan_shot_ids(self):
        plan = ResolvedPromptPlan(
            creative_intent="solemn performance",
            shots=[
                PlannedShot(shot_number=1, description="The singer waits."),
                PlannedShot(shot_number=2, description="The lantern rises."),
            ],
            overall_soundscape="wind",
            music_intent=MusicIntent.NONE,
        )
        valid = creative_shots_from_plan(plan)
        validate_creative_shots_against_plan(plan, valid)

        with self.assertRaisesRegex(ValueError, "unknown shot ID: shot-0003"):
            validate_creative_shots_against_plan(
                plan,
                [*valid, CreativeShotPayload(
                    shot_id="shot-0003",
                    visible_action="A stranger enters.",
                    performance="alert",
                )],
            )
        with self.assertRaisesRegex(ValueError, "missing creative shot payload: shot-0002"):
            validate_creative_shots_against_plan(plan, valid[:1])


if __name__ == "__main__":
    unittest.main()
