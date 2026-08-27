import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.deterministic_h3_compiler import (
    DeterministicH3Compiler,
    creative_shots_from_plan,
    validate_creative_shots_against_plan,
)
from feverslop.prompting.dspy_h3_models import (
    MusicIntent,
    PlannedShot,
    ReferenceUsage,
    ResolvedPromptPlan,
    SubjectDefinition,
)


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

    def test_compiles_guide_shaped_prompt_from_typed_plan_content(self):
        plan = ResolvedPromptPlan(
            creative_intent="CREATIVE SUMMARY",
            subjects=[],
            reference_usage=[],
            shots=[PlannedShot(
                shot_number=1,
                description="CREATIVE SHOT DESCRIPTION",
                visible_action="CREATIVE ACTION",
                performance="CREATIVE PERFORMANCE",
                start_seconds=0,
                end_seconds=2,
            )],
            overall_soundscape="CREATIVE SOUNDSCAPE",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="t2v",
            plan=plan,
            facts=self.facts,
            shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
        )
        self.assertNotIn("BASE PROMPT", prompt)
        self.assertIn("integrated_multimodal_description:", prompt)
        self.assertIn("CREATIVE SHOT DESCRIPTION", prompt)
        self.assertIn("overall_soundscape: CREATIVE SOUNDSCAPE", prompt)

    def test_compiles_r2v_six_sections_and_authoritative_reference_anchors(self):
        plan = ResolvedPromptPlan(
            creative_intent="CREATIVE R2V SUMMARY",
            subjects=[SubjectDefinition(
                label="<Subject 1>",
                name="singer",
                description="CREATIVE SUBJECT DESCRIPTION",
                source_references=["<Picture 1>"],
            )],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>",
                purpose="audio reuse",
                details="CREATIVE RETENTION DETAIL",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="CREATIVE R2V SHOT",
                start_seconds=0,
                end_seconds=2,
                reference_labels=["<Picture 1>", "<Audio 1>"],
            )],
            overall_soundscape="CREATIVE R2V SOUNDSCAPE",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="r2v",
            plan=plan,
            facts=self.facts,
            shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            prepared_reference_labels=["<Picture 1>", "<Audio 1>", "<Video 1>"],
        )
        fields = [
            "subject_definitions:", "summary:", "retention_analysis:",
            "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
        ]
        self.assertEqual(sorted(prompt.find(field) for field in fields), [prompt.find(field) for field in fields])
        for value in ("CREATIVE SUBJECT DESCRIPTION", "CREATIVE R2V SUMMARY", "CREATIVE RETENTION DETAIL", "CREATIVE R2V SHOT"):
            self.assertIn(value, prompt)
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Audio 1>", prompt)
        self.assertIn("<Video 1>", prompt)

    def test_r2v_compiler_serializes_guide_style_shot_labels_and_retention_markers(self):
        plan = ResolvedPromptPlan(
            creative_intent="a polished augmented metropolis at twilight",
            subjects=[SubjectDefinition(
                label="<Subject 1>",
                name="Elara",
                description="a young woman with sharp features",
                source_references=["<Picture 1>"],
            )],
            reference_usage=[
                ReferenceUsage(
                    reference_label="<Picture 1>",
                    purpose="identity",
                    details="Elara's face and dark hair remain recognizable.",
                ),
                ReferenceUsage(
                    reference_label="<Audio 1>",
                    purpose="audio reuse",
                    details="The original soundtrack is reused for rhythm continuity.",
                ),
            ],
            shots=[PlannedShot(
                shot_number=1,
                description="A wide shot shows the city.",
                camera_behavior="slow panoramic sweep with small amplitude",
                start_seconds=0,
                end_seconds=4,
                reference_labels=[],
                involved_subjects=["Elara"],
            )],
            overall_soundscape="Neon streets hum softly.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v",
            plan=plan,
            facts=self.facts,
            shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 4.0)},
            references={"shot-0001": []},
            prepared_reference_labels=["<Picture 1>", "<Audio 1>"],
            reference_metadata=[
                {"label": "<Picture 1>", "kind": "picture", "copy_mode": "reference"},
                {"label": "<Audio 1>", "kind": "audio", "copy_mode": "fully_copy"},
            ],
        )

        detailed = prompt.split("detailed_description: ", 1)[1].split(
            "\n\noverall_soundscape:", 1,
        )[0]
        self.assertLess(detailed.index("cinematic visual style"), detailed.index("[Shot 1]"))
        self.assertIn("<Subject 1>", detailed)
        self.assertIn("slow panoramic sweep with small amplitude", detailed)
        self.assertIn("<Subject 1> are visible in the shot.", detailed)
        self.assertIn("<Audio 1> active in the soundtrack.", detailed)
        self.assertNotIn("<Picture 1>.", detailed)
        self.assertNotIn("<Picture 1>: fully_preserved", prompt)
        self.assertIn("<Audio 1>: fully_copy", prompt)
        self.assertNotIn("..", detailed)

    def test_r2v_compiler_places_labels_in_natural_prose_and_defines_audio(self):
        plan = ResolvedPromptPlan(
            creative_intent="a singer in a luminous city",
            subjects=[
                SubjectDefinition(
                    label="<Subject 1>", name="Lead Singer", description="a man with sharp features",
                    source_references=["<Picture 1>"],
                ),
                SubjectDefinition(
                    label="<Subject 2>", name="Augmented Metropolis", description="a glowing city",
                    source_references=["<Picture 2>"],
                ),
            ],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>", purpose="audio reuse", details="the complete soundtrack is copied",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="The camera slowly tracks backward as the singer raises his hands.",
                camera_behavior="slowly tracking backward",
                start_seconds=0,
                end_seconds=4,
                involved_subjects=["Lead Singer", "Augmented Metropolis"],
            )],
            overall_soundscape="A clean electronic hum.",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 4.0)},
            reference_metadata=[
                {"label": "<Picture 1>", "kind": "picture", "role": "subject"},
                {"label": "<Picture 2>", "kind": "picture", "role": "environment"},
                {"label": "<Audio 1>", "kind": "audio", "copy_mode": "fully_copy"},
            ],
        )
        self.assertIn("<Subject 1> is a man with sharp features in <Picture 1>.", prompt)
        self.assertIn("<Subject 2> is a glowing city in <Picture 2>.", prompt)
        self.assertIn("<Audio 1> is", prompt)
        self.assertNotIn("References in this shot:", prompt)
        self.assertNotIn("Camera movement:", prompt)
        self.assertIn("<Subject 1>", prompt.split("[Shot 1]", 1)[1])
        self.assertIn("<Subject 2>", prompt.split("[Shot 1]", 1)[1])

    def test_compiles_mode_specific_frame_instructions(self):
        plan = ResolvedPromptPlan(
            creative_intent="intent",
            shots=[PlannedShot(shot_number=3, description="action", start_seconds=0, end_seconds=4)],
            overall_soundscape="N/A",
            music_intent=MusicIntent.NONE,
        )
        compiler = DeterministicH3Compiler()
        for mode, expected in (
            ("i2v", "at 0.00 seconds into the target video"),
            ("fl2v", "Picture 2 (from Shot 1) aligns with the 4.00-second mark"),
            ("l2v", "<Picture 1> (from [Shot 3]) aligns with the 4.00-second mark"),
        ):
            with self.subTest(mode=mode):
                prompt = compiler.compile(
                    mode=mode,
                    plan=plan,
                    facts=self.facts,
                    shots=self.shots[:1],
                    shot_windows={"shot-02": (0.0, 4.0)},
                    duration_seconds=4,
                )
                self.assertIn(expected, prompt)

    def test_enforces_word_budget(self):
        with self.assertRaisesRegex(ValueError, "word budget"):
            DeterministicH3Compiler(max_words=3).compile(
                mode="base", facts=self.facts, shots=self.shots,
                shot_windows={"shot-01": (0.0, 4.5), "shot-02": (4.5, 9.0)},
            )

    def test_converts_resolved_plan_to_backend_neutral_shots(self):
        plan = ResolvedPromptPlan(
            creative_intent="solemn performance",
            shots=[PlannedShot(
                shot_number=2,
                start_seconds=2,
                end_seconds=4,
                description="The lantern rises.",
                camera_behavior="slow push",
            )],
            overall_soundscape="wind",
            music_intent=MusicIntent.NONE,
        )
        shots = creative_shots_from_plan(plan)
        self.assertEqual("shot-0002", shots[0].shot_id)
        self.assertEqual("The lantern rises.", shots[0].visible_action)
        self.assertEqual("solemn performance", shots[0].performance)
        self.assertEqual("slow push", shots[0].camera_behavior)

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
