import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.deterministic_h3_compiler import (
    DeterministicH3Compiler,
    _insert_authoritative_vocal_event,
    _remove_vocal_clause,
    creative_shots_from_plan,
    plan_with_authoritative_relay,
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

    def test_r2v_compiler_preserves_every_llm_authored_shot_field(self):
        plan = ResolvedPromptPlan(
            creative_intent="A complete reference-guided performance.",
            style_opening="Live-action imagery uses cool practical lighting and crisp contrast.",
            subjects=[],
            reference_usage=[],
            shots=[PlannedShot(
                shot_number=1,
                description=" ".join([
                    "The camera-facing composition remains visually explicit.",
                ] * 72),
                visible_action="The performer raises both hands in a controlled arc.",
                performance="Her expression shifts from restraint to open defiance.",
                camera_behavior="The camera pushes forward slowly at eye level.",
                environmental_motion="Loose paper circles through the cold backlight.",
                transition_intent="The movement settles into a held final pose.",
                start_seconds=0,
                end_seconds=5,
            )],
            overall_soundscape="Paper rustles over a low room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v",
            plan=plan,
            facts=self.facts,
            shots=creative_shots_from_plan(plan),
            shot_windows={"shot-0001": (0.0, 5.0)},
            duration_seconds=5,
        )

        self.assertIn("raises both hands in a controlled arc", prompt)
        self.assertIn("expression shifts from restraint to open defiance", prompt)
        self.assertIn("pushes forward slowly at eye level", prompt)
        self.assertIn("Loose paper circles through the cold backlight", prompt)
        self.assertIn("settles into a held final pose", prompt)

    def test_r2v_compiler_turns_possessive_action_phrase_into_natural_prose(self):
        plan = ResolvedPromptPlan(
            creative_intent="A tired performer studies the skyline.",
            style_opening="Live-action imagery uses cold cyan lighting.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Jack", description="a tired young man",
            )],
            reference_usage=[],
            shots=[PlannedShot(
                shot_number=1,
                description="A wide shot frames Jack beneath the skyline.",
                visible_action="Jack's slow head tilt and slight shoulder slump.",
                performance="Jack remains visibly exhausted.",
                involved_subjects=["Jack"],
                start_seconds=0,
                end_seconds=3,
            )],
            overall_soundscape="A low electrical hum continues.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v",
            plan=plan,
            facts=self.facts,
            shots=creative_shots_from_plan(plan),
            shot_windows={"shot-0001": (0.0, 3.0)},
            duration_seconds=3,
        )

        self.assertIn(
            "The shot shows <Subject 1>'s slow head tilt and slight shoulder slump.",
            prompt,
        )
        self.assertNotIn(". <Subject 1>'s slow head tilt", prompt)

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
        self.assertIn("<Subject 1> is also present in the shot.", detailed)
        self.assertNotIn("<Audio 1> is fully copied as the complete soundtrack", detailed)
        self.assertNotIn("<Audio 1> is visible in the shot", detailed)
        self.assertIn("fully copied as the target video's audio signal", prompt)
        self.assertNotIn("<Picture 1>.", detailed)
        self.assertNotIn("<Picture 1>: fully_preserved", prompt)
        self.assertIn("<Audio 1>: fully_copy", prompt)
        self.assertNotIn("..", detailed)

    def test_missing_environment_label_does_not_steal_following_pronoun(self):
        plan = ResolvedPromptPlan(
            creative_intent="A technician crosses a city.",
            subjects=[
                SubjectDefinition(label="<Subject 1>", name="Jack", description="a technician"),
                SubjectDefinition(label="<Subject 2>", name="Metropolis", description="a cityscape"),
            ],
            shots=[PlannedShot(
                shot_number=1,
                description="He struggles to maintain focus while Jack walks forward.",
                involved_subjects=["Jack", "Metropolis"],
            )],
            overall_soundscape="Quiet ambience.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
        )

        self.assertNotIn("<Subject 2> is visible in the shot. He", prompt)
        self.assertIn("He struggles to maintain focus while <Subject 1> walks forward.", prompt)
        self.assertIn("<Subject 2> is also present in the shot.", prompt)

    def test_camera_behavior_is_rendered_without_meta_wrapper(self):
        plan = ResolvedPromptPlan(
            creative_intent="A camera study.",
            shots=[PlannedShot(
                shot_number=1,
                description="A technician pauses.",
                camera_behavior="The low-angle arc continues slowly.",
            )],
            overall_soundscape="Quiet ambience.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
        )

        self.assertIn("The low-angle arc continues slowly.", prompt)
        self.assertNotIn("camera movement is described as", prompt)

    def test_camera_repairs_double_article_fragment(self):
        plan = ResolvedPromptPlan(
            creative_intent="A camera study.",
            shots=[PlannedShot(
                shot_number=1,
                description="A technician pauses.",
                camera_behavior="The camera the low-angle perspective.",
            )],
            overall_soundscape="Quiet ambience.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
        )

        self.assertIn("The camera maintains the low-angle perspective.", prompt)

    def test_relay_cleanup_removes_empty_lyric_quotes(self):
        plan = ResolvedPromptPlan(
            creative_intent="A vocal close-up.",
            shots=[PlannedShot(
                shot_number=1,
                description="The performer pours emotion into the lyrics ''.",
            )],
            overall_soundscape="Digital static surrounds the vocals.",
            music_intent=MusicIntent.NONE,
        )

        normalized = plan_with_authoritative_relay(
            plan,
            [{"state": "singing", "lyrics": "Hold on"}],
        )

        self.assertNotIn("''", normalized.shots[0].description)

    def test_soundscape_preserves_ambience_when_vocal_clause_is_removed(self):
        plan = ResolvedPromptPlan(
            creative_intent="A glitching city.",
            shots=[PlannedShot(shot_number=1, description="A figure waits.")],
            overall_soundscape=(
                "Subtle digital static and rhythmic glitching sounds surround the scene, "
                "while resonant vocals carry the melody."
            ),
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{"state": "singing", "lyrics": "Hold on"}],
        )

        self.assertIn("Subtle digital static and rhythmic glitching sounds", prompt)
        self.assertNotIn("No additional diegetic ambience", prompt)

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
                reference_label="<Audio 1>", purpose="vocals",
                details="vocals stem; bound to <Subject 1> (S1)",
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
            relay_segments=[{
                "state": "singing", "lyrics": "So I blink",
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
            speaker_bindings=[{
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
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
        retention = prompt.split("retention_analysis:\n", 1)[1].split(
            "\n\ndetailed_description:", 1,
        )[0]
        self.assertNotIn("(S1)", retention)

    def test_r2v_compiler_inserts_authoritative_relay_lyrics(self):
        plan = ResolvedPromptPlan(
            creative_intent="The singer notices a visual glitch.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Jack", description="a young technician with dark hair",
                source_references=["<Picture 1>"],
            )],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>", purpose="vocals", details="vocals stem",
            )],
            shots=[
                PlannedShot(
                    shot_number=1, start_seconds=0, end_seconds=5.92,
                    description=(
                        "Jack watches the light and keeps his mouth closed while a display "
                        "shows an LLM-authored <d>untagged invention</d>. "
                        "A status panel says 'wrong fragment'."
                    ),
                ),
                PlannedShot(
                    shot_number=2, start_seconds=5.92, end_seconds=8.08,
                    description=(
                        "Jack sings invented words that are not in the relay. "
                        "The camera moves closer while neon light pulses. "
                        "He performs the lyrics: Pixels dance on my."
                    ),
                ),
            ],
            overall_soundscape="A low electronic ambience surrounds the performance.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots,
            shot_windows={"shot-01": (0.0, 5.92), "shot-02": (5.92, 8.08)},
            prepared_reference_labels=["<Picture 1>", "<Audio 1>"],
            dialogue_language="English",
            relay_segments=[
                {
                    "start_seconds": 0.0, "end_seconds": 5.92,
                    "state": "instrumental",
                    "prompt": "same scene, instrumental section, character is not singing",
                },
                {
                    "start_seconds": 5.92, "end_seconds": 8.08,
                    "state": "singing",
                    "subject_label": "<Subject 1>", "speaker_id": "S1",
                    "prompt": (
                        "same scene, character sings with expressive lip sync, "
                        "performing the lyrics: Pixels dance on my"
                    ),
                },
            ],
            speaker_bindings=[{
                "audio_label": "<Audio 1>",
                "subject_label": "<Subject 1>",
                "speaker_id": "S1",
            }],
            reference_metadata=[{
                "label": "<Audio 1>", "kind": "audio", "name": "vocals",
                "description": "vocals stem", "copy_mode": "partially_copy",
            }],
        )

        detailed = prompt.split("detailed_description: ", 1)[1].split(
            "\n\noverall_soundscape:", 1,
        )[0]
        shot_1, shot_2 = detailed.split("[Shot 2]", 1)
        self.assertNotIn("<d>", shot_1)
        self.assertIn("untagged invention", shot_1)
        self.assertIn("wrong fragment", shot_1)
        self.assertIn("<Subject 1> (S1) sings", shot_2)
        self.assertIn("<d>[English] Pixels dance on my.</d> from <Audio 1>", shot_2)
        self.assertNotIn("invented words", shot_2)
        self.assertIn("The camera moves closer while neon light pulses.", shot_2)
        self.assertEqual(1, detailed.count("Pixels dance on my"))
        self.assertIn(
            "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)",
            prompt,
        )

    def test_r2v_compiler_does_not_guess_speakers_from_descriptions(self):
        plan = ResolvedPromptPlan(
            creative_intent="A figure speaks.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Unit Seven",
                description="a robot performer", source_references=["<Picture 1>"],
            )],
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description="Unit Seven says <d>[English] Ready.</d>",
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            prepared_reference_labels=["<Picture 1>"],
            speaker_bindings=[{
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
        )

        self.assertNotIn("(S1)", prompt)

    def test_relay_without_subject_does_not_select_the_only_bound_subject(self):
        plan = ResolvedPromptPlan(
            creative_intent="An unseen voice speaks.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Unit Seven", description="a robot",
            )],
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description="Unit Seven faces the camera.",
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{"state": "dialogue", "dialogue": "Ready"}],
            speaker_bindings=[{
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
        )

        self.assertIn("The audible voice says, <d>[English] Ready.</d>", prompt)
        self.assertNotIn("<Subject 1> (S1)", prompt)

    def test_instrumental_relay_removes_vocal_claim_and_preserves_visual_action(self):
        plan = ResolvedPromptPlan(
            creative_intent="A platform scene.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Unit Seven", description="a robot",
            )],
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description=(
                    "Unit Seven sings invented words. "
                    "The camera circles the platform."
                ),
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{"state": "instrumental"}],
        )

        self.assertNotIn("sings invented words", prompt)
        self.assertIn("The camera circles the platform.", prompt)

    def test_relay_cleanup_preserves_visual_clause_and_removes_vocal_synonyms(self):
        plan = ResolvedPromptPlan(
            creative_intent="A platform scene.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Unit Seven", description="a robot",
            )],
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description=(
                    "Unit Seven whispers invented words and raises a lantern. "
                    "Unit Seven chants another invention."
                ),
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{
                "state": "dialogue", "dialogue": "Authoritative line",
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
            speaker_bindings=[{
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
        )

        self.assertNotIn("invented words", prompt)
        self.assertNotIn("another invention", prompt)
        self.assertIn("<Subject 1> raises a lantern.", prompt)

    def test_relay_cleanup_is_actor_agnostic_and_removes_every_vocal_clause(self):
        plan = ResolvedPromptPlan(
            creative_intent="A platform scene.",
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description=(
                    "The camera circles while an android sings invented words. "
                    "A child whispers one invention and chants another while the lantern rises."
                ),
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{"state": "instrumental"}],
        )

        self.assertNotRegex(prompt, r"(?i)\b(?:sings|whispers|chants)\b")
        self.assertIn("The camera circles.", prompt)
        self.assertIn("The lantern rises.", prompt)

    def test_authoritative_plan_discards_section_marker_vocals(self):
        plan = ResolvedPromptPlan(
            creative_intent="A visual scene.",
            shots=[PlannedShot(
                shot_number=1,
                description="An android sings the lyrics [Verse] and raises a lantern.",
                performance="Expressive lip-sync to [Verse].",
            )],
            overall_soundscape="Room tone.",
            music_intent=MusicIntent.NONE,
        )

        normalized = plan_with_authoritative_relay(
            plan,
            [{"state": "singing", "prompt": "performing the lyrics: [Verse]"}],
        )

        self.assertNotIn("Verse", normalized.shots[0].description)
        self.assertNotRegex(normalized.shots[0].description, r"(?i)\bsings?\b")
        self.assertIn("raises a lantern", normalized.shots[0].description)

    def test_authoritative_plan_replaces_malformed_dialogue_with_relay_words(self):
        plan = ResolvedPromptPlan(
            creative_intent="A visual scene.",
            shots=[PlannedShot(
                shot_number=1,
                description=(
                    "An android sings <d>en</d>can't unsee. So I blink and the frame<</d>. "
                    "The camera pushes closer."
                ),
            )],
            overall_soundscape="Room tone.",
            music_intent=MusicIntent.NONE,
        )

        normalized = plan_with_authoritative_relay(
            plan,
            [{"state": "singing", "lyrics": "can't unsee. So I blink and the frame"}],
        )

        description = normalized.shots[0].description
        self.assertNotIn("So<", description)
        self.assertNotIn("audible voice the frame", description.casefold())
        self.assertIn("The camera pushes closer.", description)
        self.assertEqual(1, description.count("can't unsee. So I blink and the frame"))

    def test_compiler_does_not_reparse_canonical_relay_words_as_visual_clauses(self):
        lyrics = "can't unsee. So I blink and the frame"
        plan = ResolvedPromptPlan(
            creative_intent="A visual scene.",
            shots=[PlannedShot(
                shot_number=1,
                description=(
                    "The camera pushes closer. The audible voice sings, "
                    f"<d>[English] {lyrics}.</d>"
                ),
            )],
            overall_soundscape="Room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{"state": "singing", "lyrics": lyrics}],
        )

        detailed = prompt.split("detailed_description: ", 1)[1].split(
            "\n\noverall_soundscape:", 1,
        )[0]
        self.assertEqual(1, detailed.count(lyrics))
        self.assertNotIn("audible voice the frame", detailed.casefold())
        self.assertNotRegex(detailed, r"\[Shot 1\]\s+So\.")

    def test_rejects_non_bijective_speaker_bindings_before_relay_processing(self):
        plan = ResolvedPromptPlan(
            creative_intent="Two units wait.",
            subjects=[
                SubjectDefinition(label="<Subject 1>", name="One", description="a unit"),
                SubjectDefinition(label="<Subject 2>", name="Two", description="a unit"),
            ],
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description="One and Two wait.",
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        with self.assertRaisesRegex(ValueError, "speaker ID S1"):
            DeterministicH3Compiler().compile(
                mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
                shot_windows={"shot-02": (0.0, 2.0)},
                speaker_bindings=[
                    {"subject_label": "<Subject 1>", "speaker_id": "S1"},
                    {"subject_label": "<Subject 2>", "speaker_id": "S1"},
                ],
            )

    def test_relay_uses_its_explicit_subject_with_multiple_speaker_bindings(self):
        plan = ResolvedPromptPlan(
            creative_intent="Two robots exchange a line.",
            subjects=[
                SubjectDefinition(label="<Subject 1>", name="Unit One", description="a red robot"),
                SubjectDefinition(label="<Subject 2>", name="Unit Two", description="a blue robot"),
            ],
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description="Unit Two looks toward Unit One and says Systems ready.",
                involved_subjects=["Unit One", "Unit Two"],
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{
                "state": "dialogue", "dialogue": "Systems ready",
                "subject_label": "<Subject 2>", "speaker_id": "S2",
            }],
            speaker_bindings=[
                {"subject_label": "<Subject 1>", "speaker_id": "S1"},
                {"subject_label": "<Subject 2>", "speaker_id": "S2"},
            ],
        )

        self.assertIn("<Subject 2> (S2) says, <d>[English] Systems ready.</d>", prompt)
        self.assertNotIn("<Subject 1> (S1)", prompt)

    def test_rejects_conflicting_relay_and_speaker_binding_ids(self):
        plan = ResolvedPromptPlan(
            creative_intent="A bound speaker talks.",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Unit One", description="a red robot",
            )],
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description="Unit One faces forward.",
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )

        with self.assertRaisesRegex(ValueError, "conflicting speaker ID"):
            DeterministicH3Compiler().compile(
                mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
                shot_windows={"shot-02": (0.0, 2.0)},
                relay_segments=[{
                    "state": "dialogue", "dialogue": "Ready",
                    "subject_label": "<Subject 1>", "speaker_id": "S2",
                }],
                speaker_bindings=[{
                    "subject_label": "<Subject 1>", "speaker_id": "S1",
                }],
            )

    def test_removes_llm_dialogue_markup_outside_detailed_description(self):
        plan = ResolvedPromptPlan(
            creative_intent="A quiet scene <d>[Martian] invented summary words</d>.",
            style_opening="A restrained style <d>[Martian] invented style words</d>.",
            shots=[PlannedShot(
                shot_number=1, start_seconds=0, end_seconds=2,
                description="A silent figure waits.",
            )],
            overall_soundscape="Room tone <d>[Martian] invented sound words</d>.",
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v", plan=plan, facts=self.facts, shots=self.shots[:1],
            shot_windows={"shot-02": (0.0, 2.0)},
            relay_segments=[{
                "state": "instrumental", "prompt": "no dialogue or singing",
            }],
        )

        self.assertNotIn("<d>", prompt)
        self.assertNotIn("invented summary words", prompt)
        self.assertNotIn("invented style words", prompt)
        self.assertNotIn("invented sound words", prompt)

    def test_r2v_compiler_states_audio_copy_relationship_outside_the_shot(self):
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
        self.assertNotIn("<Audio 1> is fully copied as the complete soundtrack", detailed)
        self.assertIn("fully copied as the target video's audio signal", prompt)
        self.assertNotIn("<Audio 1> active in the soundtrack", detailed)
        self.assertIn(
            "non_diegetic_music: <Audio 1> is directly reused as the complete audience-only score.",
            prompt,
        )

    def test_r2v_compiler_places_partially_copied_music_stem_in_music_section(self):
        plan = ResolvedPromptPlan(
            creative_intent="A performer crosses an illuminated room.",
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>",
                purpose="audio reuse",
                details="The drum stem is partially copied.",
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
            shots=creative_shots_from_plan(plan),
            shot_windows={"shot-0001": (0.0, 4.0)},
            prepared_reference_labels=["<Audio 1>"],
            reference_metadata=[{
                "label": "<Audio 1>",
                "kind": "audio",
                "name": "drums",
                "description": "isolated drum stem",
                "copy_mode": "partially_copy",
            }],
        )

        self.assertIn(
            "non_diegetic_music: The selected audience-only music layer from <Audio 1> "
            "is partially copied into the target video.",
            prompt,
        )

    def test_r2v_compiler_tracks_all_subjects_and_keeps_non_music_ambience(self):
        plan = ResolvedPromptPlan(
            creative_intent="The singer performs in a sterile city.",
            subjects=[
                SubjectDefinition(label="<Subject 1>", name="singer", description="a singer"),
                SubjectDefinition(label="<Subject 2>", name="digital_paradise", description="a sterile city"),
            ],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>",
                purpose="audio reference",
                details="Use the original song for rhythm continuity without copying its signal.",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="The singer crosses the sterile city.",
                start_seconds=0,
                end_seconds=4,
                involved_subjects=["singer", "digital_paradise"],
                reference_labels=["<Audio 1>"],
            )],
            overall_soundscape=(
                "The supplied full_mix song establishes the beat; subtle sterile digital "
                "ambience remains low throughout the scene."
            ),
            music_intent=MusicIntent.NONE,
        )

        prompt = DeterministicH3Compiler().compile(
            mode="r2v",
            plan=plan,
            facts=self.facts,
            shots=creative_shots_from_plan(plan),
            shot_windows={"shot-0001": (0.0, 4.0)},
            prepared_reference_labels=["<Audio 1>"],
            reference_metadata=[{
                "label": "<Audio 1>",
                "kind": "audio",
                "name": "full_mix",
                "description": "original song for beat and rhythm continuity",
                "copy_mode": "fully_copy",
            }],
        )

        summary = prompt.split("summary:", 1)[1].split("retention_analysis:", 1)[0]
        self.assertIn("<Subject 1>", summary)
        self.assertIn("<Subject 2>", summary)
        self.assertIn("overall_soundscape: Subtle sterile digital ambience", prompt)
        self.assertIn("non_diegetic_music: N/A", prompt)
        self.assertIn("<Audio 1>: reference", prompt)
        self.assertNotIn("<Audio 1>: fully_copy", prompt)
        self.assertIn("without copying the source signal", prompt)
        self.assertNotIn("<Audio 1> is the complete soundtrack", prompt)
        detailed = prompt.split("detailed_description:", 1)[1].split(
            "overall_soundscape:", 1,
        )[0]
        self.assertNotIn("<Audio 1>", detailed)

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
            relay_segments=[{
                "state": "singing", "lyrics": "So I blink",
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
            speaker_bindings=[{
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
        )

        self.assertIn("<Subject 1> is a man with sharp features in <Picture 1>.", prompt)
        self.assertNotIn("<Subject 1> is <Subject 1> is", prompt)
        self.assertIn("<Subject 2> is visible", prompt)
        self.assertNotIn("<Subject 2> (S2)", prompt)
        self.assertIn("<Subject 1> (S1) sings, <d>[English] So I blink.</d>", prompt)

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
            relay_segments=[
                {
                    "state": "singing", "lyrics": "first line",
                    "subject_label": "<Subject 1>", "speaker_id": "S1",
                },
                {"state": "instrumental"},
                {
                    "state": "singing", "lyrics": "second line",
                    "subject_label": "<Subject 1>", "speaker_id": "S1",
                },
            ],
            speaker_bindings=[{
                "audio_label": "<Audio 1>",
                "subject_label": "<Subject 1>", "speaker_id": "S1",
            }],
            prepared_reference_labels=["<Picture 1>", "<Picture 2>", "<Audio 1>", "<Audio 2>"],
            reference_metadata=[
                {"label": "<Audio 1>", "kind": "audio", "name": "vocals", "copy_mode": "partially_copy"},
                {"label": "<Audio 2>", "kind": "audio", "name": "full_mix", "copy_mode": "fully_copy"},
            ],
        )
        detailed = prompt.split("detailed_description: ", 1)[1].split("\n\noverall_soundscape:", 1)[0]

        self.assertEqual(2, detailed.count("<Subject 1> (S1)"))
        self.assertIn("<Subject 1> (S1) sings, <d>[English] first line.</d>", detailed)
        self.assertNotIn("<Subject 1> (S1) is silent", detailed)
        self.assertNotIn("(S2)", detailed)
        self.assertNotIn("(S3)", detailed)
        self.assertIn("<d>[English] first line.</d>", detailed)
        self.assertIn("<d>[English] second line.</d>", detailed)
        self.assertIn("</d> from <Audio 1>", detailed)
        self.assertNotIn("<Audio 2>", detailed)
        self.assertNotIn("partially copied", detailed)
        self.assertNotIn("referenced for the target video's rhythm", detailed)
        self.assertNotIn("The camera Slowly", detailed)
        self.assertIn("<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3])", prompt)
        self.assertNotIn("overall_soundscape: A musical track", prompt)

    def test_r2v_compiler_uses_explicit_speaker_bindings(self):
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
            relay_segments=[
                {
                    "state": "dialogue", "dialogue": "Hello",
                    "subject_label": "<Subject 2>", "speaker_id": "S1",
                },
                {
                    "state": "dialogue", "dialogue": "Welcome",
                    "subject_label": "<Subject 3>", "speaker_id": "S2",
                },
            ],
            speaker_bindings=[
                {"subject_label": "<Subject 2>", "speaker_id": "S1"},
                {"subject_label": "<Subject 3>", "speaker_id": "S2"},
            ],
        )

        self.assertIn("<Subject 2> (S1) says, <d>[English] Hello.</d>", prompt)
        self.assertIn("<Subject 3> (S2) says, <d>[English] Welcome.</d>", prompt)
        self.assertNotIn("<Subject 1> (S", prompt)
        retention = prompt.split("retention_analysis:\n", 1)[1].split(
            "\n\ndetailed_description:", 1,
        )[0]
        self.assertNotIn("(S1)", retention)
        self.assertNotIn("(S2)", retention)

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

    def test_remove_vocal_clause_keeps_negated_instrumental_clause(self):
        # Negated vocal clauses ARE the instrumental instruction; removing one
        # would flip an instrumental window into a vocal one.
        self.assertEqual(
            "The performer is not singing",
            _remove_vocal_clause("The performer is not singing"),
        )
        self.assertEqual(
            "The performer never sings",
            _remove_vocal_clause("The performer never sings"),
        )

    def test_remove_vocal_clause_keeps_non_vocal_performance_action(self):
        # "perform" without a vocal object is a stage action to keep.
        self.assertEqual(
            "The performer performs on stage",
            _remove_vocal_clause("The performer performs on stage"),
        )

    def test_remove_vocal_clause_removes_vocal_performance_action(self):
        # "perform" directed at a vocal object is a vocal claim to strip.
        self.assertEqual("", _remove_vocal_clause("The performer performs the lyrics"))

    def test_insert_authoritative_vocal_event_anchors_to_bound_subject(self):
        result = _insert_authoritative_vocal_event(
            "[Shot 1] The singer raises a lantern.",
            {
                "state": "singing",
                "lyrics": "Hold on",
                "subject_label": "<Subject 1>",
                "speaker_id": "S1",
            },
            bound_speaker_ids={"<Subject 1>": "S1"},
        )
        self.assertIn("<Subject 1> (S1) sings, <d>Hold on</d>", result)
        self.assertNotIn("The audible voice", result)

    def test_insert_authoritative_vocal_event_falls_back_to_audible_voice(self):
        result = _insert_authoritative_vocal_event(
            "[Shot 1] The singer raises a lantern.",
            {"state": "singing", "lyrics": "Hold on"},
            bound_speaker_ids={},
        )
        self.assertIn("The audible voice sings, <d>Hold on</d>", result)


if __name__ == "__main__":
    unittest.main()
