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
            style_opening="Hyperreal live-action imagery uses cyan and amber practical lighting.",
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
        self.assertLess(detailed.index("Hyperreal live-action imagery"), detailed.index("[Shot 1]"))
        self.assertNotIn("deliberate visual continuity", detailed)
        self.assertIn("<Subject 1>", detailed)
        self.assertIn("slow panoramic sweep with small amplitude", detailed)
        self.assertIn("<Subject 1> is visible in the shot.", detailed)
        self.assertIn("<Audio 1> fully copied as the complete soundtrack", detailed)
        self.assertNotIn("<Picture 1>.", detailed)
        self.assertNotIn("<Picture 1>: fully_preserved", prompt)
        self.assertIn("<Audio 1>: fully_copy", prompt)
        self.assertNotIn("..", detailed)

    def test_base_modes_compile_exact_shot_timestamps_and_frame_relationships(self):
        plan = ResolvedPromptPlan(
            creative_intent="A continuous movement.",
            shots=[
                PlannedShot(
                    shot_number=1,
                    description="A cyclist starts beside a silver bicycle.",
                    start_seconds=0,
                    end_seconds=3,
                    reference_labels=["<Picture 1>"],
                ),
                PlannedShot(
                    shot_number=2,
                    description="The cyclist opens an umbrella.",
                    start_seconds=3,
                    end_seconds=6,
                    reference_labels=["<Picture 2>"],
                ),
            ],
            overall_soundscape="Rain falls steadily.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="fl2v",
            plan=plan,
            facts=self.facts,
            shots=self.shots,
            shot_windows={"shot-01": (0, 3), "shot-02": (3, 6)},
            duration_seconds=6,
        )

        self.assertTrue(prompt.startswith(
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            "Picture 2 (from Shot 2) aligns with the 6.00-second mark of the target video."
        ))
        self.assertIn("[Shot 1] The shot begins from <Picture 1>, preserving its composition.", prompt)
        self.assertIn("[Shot 2] At 00:03.000,", prompt)
        self.assertIn("ends on <Picture 2>", prompt)

    def test_r2v_compiler_places_labels_in_natural_prose_and_defines_audio(self):
        plan = ResolvedPromptPlan(
            creative_intent="Lead Singer in The Augmented Metropolis",
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
        summary = prompt.split("summary:", 1)[1].split("retention_analysis:", 1)[0]
        self.assertIn("<Audio 1>", summary)
        self.assertIn("fully copied", summary)
        self.assertNotIn("full_mix -", prompt)
        self.assertNotIn("References in this shot:", prompt)
        self.assertNotIn("Camera movement:", prompt)
        self.assertIn("<Subject 1>", summary)
        self.assertNotIn("The <Subject", prompt)
        self.assertNotIn("<Subject 1> are", prompt)
        self.assertIn("<Subject 1>", prompt.split("[Shot 1]", 1)[1])
        self.assertIn("<Subject 2>", prompt.split("[Shot 1]", 1)[1])

    def test_r2v_compiler_canonicalizes_dialogue_and_audio_anchors(self):
        plan = ResolvedPromptPlan(
            creative_intent="The singer performs.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Lead Singer", description="a singer",
                source_references=["<Picture 1>"],
            )],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>", purpose="vocals", details="vocals stem",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="<Subject 1> sings 'So I blink' with intense eyes.",
                start_seconds=0, end_seconds=2,
            )],
            overall_soundscape="The vocal line So I blink continues over the scene.",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            prepared_reference_labels=["<Picture 1>", "<Audio 1>"],
            dialogue_language="German",
            reference_metadata=[{
                "label": "<Audio 1>", "kind": "audio", "name": "vocals",
                "description": "vocals stem", "copy_mode": "partially_copy",
            }],
        )
        self.assertIn("<d>[German] So I blink.</d>", prompt)
        self.assertIn("<Subject 1> (S1) sings", prompt)
        self.assertNotIn("<d>en So I blink</d>", prompt)
        self.assertIn("<Audio 1> is the vocal stem and is partially copied", prompt)
        self.assertNotIn("overall_soundscape: The vocal line So I blink", prompt)

    def test_r2v_compiler_states_audio_copy_relationship_inside_the_shot(self):
        plan = ResolvedPromptPlan(
            creative_intent="A continuous performance.",
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>",
                purpose="audio reuse",
                details="The complete soundtrack is copied.",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="A performer crosses the illuminated room.",
                start_seconds=0,
                end_seconds=4,
                reference_labels=["<Audio 1>"],
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v",
            plan=plan,
            facts=self.facts,
            shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 4.0)},
            prepared_reference_labels=["<Audio 1>"],
            reference_metadata=[{
                "label": "<Audio 1>",
                "kind": "audio",
                "name": "full_mix",
                "copy_mode": "fully_copy",
            }],
        )

        detailed = prompt.split("detailed_description:", 1)[1].split(
            "overall_soundscape:", 1,
        )[0]
        self.assertIn("<Audio 1> fully copied as the complete soundtrack", detailed)
        self.assertNotIn("<Audio 1> active in the soundtrack", detailed)

    def test_r2v_compiler_replaces_subject_aliases_case_insensitively(self):
        plan = ResolvedPromptPlan(
            creative_intent="The lead singer performs.",
            subjects=[SubjectDefinition(
                label="<Subject 1>",
                name="Lead Singer",
                description="A singer in a silver coat",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="The lead singer turns while LEAD SINGER'S hand reaches forward.",
                start_seconds=0,
                end_seconds=4,
                involved_subjects=["Lead Singer"],
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v",
            plan=plan,
            facts=self.facts,
            shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 4.0)},
        )
        detailed = prompt.split("detailed_description:", 1)[1].split(
            "overall_soundscape:", 1,
        )[0]

        self.assertNotRegex(detailed, r"(?i)\blead singer\b")
        self.assertIn("<Subject 1> turns", detailed)
        self.assertIn("<Subject 1>'S hand", detailed)

    def test_creative_projection_removes_compiler_owned_labels_from_fields(self):
        plan = ResolvedPromptPlan(
            creative_intent="intent",
            shots=[PlannedShot(
                shot_number=1,
                description="action",
                visible_action="<Picture 9> move toward the window",
                performance="singing",
                start_seconds=0,
                end_seconds=2,
            )],
            overall_soundscape="sound",
            music_intent=MusicIntent.NONE,
        )

        projected = creative_shots_from_plan(plan)

        self.assertEqual("move toward the window", projected[0].visible_action)

    def test_r2v_compiler_repairs_duplicate_subject_labels_and_speaker_tags(self):
        plan = ResolvedPromptPlan(
            creative_intent="The singer performs.",
            subjects=[
                SubjectDefinition(
                    label="<Subject 1>", name="Lead Singer",
                    description="<Subject 1> is a man with sharp features",
                    source_references=["<Picture 1>"],
                ),
                SubjectDefinition(
                    label="<Subject 2>", name="Apartment",
                    description="a minimalist apartment",
                ),
            ],
            shots=[PlannedShot(
                shot_number=1,
                description=(
                    "<Subject 2> (S2) is visible. <Subject 1> (S1) sings "
                    "<d>[en] </d>So I blink<</d>."
                ),
                start_seconds=0, end_seconds=2,
            )],
            overall_soundscape="Ambient room tone.",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
        )

        self.assertIn("<Subject 1> is a man with sharp features in <Picture 1>.", prompt)
        self.assertNotIn("<Subject 1> is <Subject 1> is", prompt)
        self.assertIn("<Subject 2> is visible", prompt)
        self.assertNotIn("<Subject 2> (S2)", prompt)
        self.assertIn("<Subject 1> (S1) sings <d>[English] So I blink.</d>", prompt)

    def test_r2v_compiler_canonicalizes_multishot_vocal_reference_plan(self):
        plan = ResolvedPromptPlan(
            creative_intent="The Lead Singer confronts the fracturing apartment.",
            subjects=[
                SubjectDefinition(
                    label="<Subject 1>", name="Lead Singer",
                    description="A man with sharp features", source_references=["<Picture 1>"],
                ),
                SubjectDefinition(
                    label="<Subject 2>", name="Apartment",
                    description="A minimalist apartment", source_references=["<Picture 2>"],
                ),
                SubjectDefinition(
                    label="<Subject 3>", name="floor_cracks",
                    description="Cracks in the floor",
                ),
            ],
            reference_usage=[
                ReferenceUsage(reference_label="<Audio 1>", purpose="vocals", details="vocals stem"),
                ReferenceUsage(reference_label="<Audio 2>", purpose="song", details="full mix"),
            ],
            shots=[
                PlannedShot(
                    shot_number=1, start_seconds=0, end_seconds=2,
                    description=(
                        "The Lead Singer is centered. He sings "
                        "<d>[en] first line</d>. The camera slowly tilts down."
                    ),
                    camera_behavior="Slowly tilting downward from the Lead Singer's gaze.",
                    involved_subjects=["Lead Singer", "Apartment", "floor_cracks"],
                ),
                PlannedShot(
                    shot_number=2, start_seconds=2, end_seconds=3,
                    description="The Lead Singer is silent while the camera continues downward.",
                    camera_behavior="Continuing the downward tilt.",
                    involved_subjects=["Lead Singer", "Apartment", "floor_cracks"],
                ),
                PlannedShot(
                    shot_number=3, start_seconds=3, end_seconds=5,
                    description=(
                        "The Lead Singer, facing the floor, resumes singing "
                        "<d>[en] second line</d>. The camera completes the tilt."
                    ),
                    camera_behavior="Completing the downward tilt.",
                    involved_subjects=["Lead Singer", "Apartment", "floor_cracks"],
                ),
            ],
            overall_soundscape="A musical track featuring vocals and a full mix.",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 5.0)}, dialogue_language="en",
            prepared_reference_labels=["<Picture 1>", "<Picture 2>", "<Audio 1>", "<Audio 2>"],
            reference_metadata=[
                {"label": "<Audio 1>", "kind": "audio", "name": "vocals", "copy_mode": "partially_copy"},
                {"label": "<Audio 2>", "kind": "audio", "name": "full_mix", "copy_mode": "fully_copy"},
            ],
        )
        detailed = prompt.split("detailed_description: ", 1)[1].split("\n\noverall_soundscape:", 1)[0]

        self.assertEqual(2, detailed.count("<Subject 1> (S1)"))
        self.assertIn("<Subject 1> (S1) sings <d>[English] first line.</d>", detailed)
        self.assertNotIn("<Subject 1> (S1) is silent", detailed)
        self.assertNotIn("(S2)", detailed)
        self.assertNotIn("(S3)", detailed)
        self.assertIn("<d>[English] first line.</d>", detailed)
        self.assertIn("<d>[English] second line.</d>", detailed)
        self.assertEqual(1, detailed.count("<Audio 1>"))
        self.assertEqual(1, detailed.count("<Audio 2>"))
        self.assertNotIn("The camera Slowly", detailed)
        self.assertIn("<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3])", prompt)
        self.assertNotIn("overall_soundscape: A musical track", prompt)

    def test_r2v_compiler_assigns_speaker_ids_by_first_vocal_event(self):
        plan = ResolvedPromptPlan(
            creative_intent="Two people speak in a room.",
            subjects=[
                SubjectDefinition(label="<Subject 1>", name="Room", description="An interior room"),
                SubjectDefinition(label="<Subject 2>", name="Alice", description="A woman in a blue coat"),
                SubjectDefinition(label="<Subject 3>", name="Bob", description="A man in a dark coat"),
            ],
            shots=[
                PlannedShot(
                    shot_number=1, start_seconds=0, end_seconds=2,
                    description="Alice says <d>[en] Hello.</d>", involved_subjects=["Room", "Alice"],
                ),
                PlannedShot(
                    shot_number=2, start_seconds=2, end_seconds=4,
                    description="Bob says <d>[en] Welcome.</d>", involved_subjects=["Room", "Bob"],
                ),
            ],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 4.0)}, dialogue_language="en",
        )

        self.assertIn("<Subject 2> (S1) says <d>[English] Hello.</d>", prompt)
        self.assertIn("<Subject 3> (S2) says <d>[English] Welcome.</d>", prompt)
        self.assertNotIn("<Subject 1> (S", prompt)

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
            ("fl2v", "Picture 2 (from Shot 3) aligns with the 4.00-second mark"),
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

    def test_r2v_compiler_inserts_guide_cut_timestamp_for_later_shots(self):
        plan = ResolvedPromptPlan(
            creative_intent="a continuous performance",
            shots=[
                PlannedShot(shot_number=1, description="The singer stands.", start_seconds=0, end_seconds=2),
                PlannedShot(shot_number=2, description="The singer turns.", start_seconds=2.42, end_seconds=4),
            ],
            overall_soundscape="Room tone.",
            music_intent=MusicIntent.NONE,
        )
        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 4.0)},
        )
        detailed = prompt.split("detailed_description: ", 1)[1].split(
            "\n\noverall_soundscape:", 1,
        )[0]
        self.assertIn("[Shot 1]", detailed)
        self.assertIn("[Shot 2] At 00:02.420,", detailed)

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
