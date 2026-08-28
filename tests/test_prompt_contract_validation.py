import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.prompt_contract_validation import (
    validate_h3_prompt_contract,
    validate_h3_prompt_shape,
    validate_prompt_contract,
)
from feverslop.prompting.dspy_h3_models import (
    MusicIntent,
    PlannedShot,
    ResolvedPromptPlan,
    SubjectDefinition,
)


class PromptContractValidationTests(unittest.TestCase):
    def setUp(self):
        self.facts = LockedSceneFacts.create(
            scene_id="scene-01",
            facts=[{"category": "wardrobe", "key": "hero", "value": "silver coat", "source_id": "cast:hero"}],
        )
        self.shots = [
            CreativeShotPayload(shot_id="shot-0001", visible_action="waits", performance="quiet"),
            CreativeShotPayload(shot_id="shot-0002", visible_action="turns", performance="alert"),
        ]
        self.windows = {"shot-0001": (0, 2), "shot-0002": (2, 4)}
        self.prompt = (
            "Scene: scene-01\n"
            "Locked facts:\n"
            "- wardrobe/hero: silver coat\n"
            "[Shot 1 | 00:00.000-00:02.000]\nAction: waits\nPerformance: quiet\n"
            "References: <Picture 1>\n"
            "[Shot 2 | 00:02.000-00:04.000]\nAction: turns\nPerformance: alert\n"
            "References: <Picture 2>"
        )

    def test_accepts_complete_structured_prompt(self):
        issues = validate_prompt_contract(
            self.prompt,
            facts=self.facts,
            shots=self.shots,
            shot_windows=self.windows,
            references={"shot-0001": ["<Picture 1>"], "shot-0002": ["<Picture 2>"]},
            prepared_reference_labels={"<Picture 1>", "<Picture 2>"},
            duration_seconds=4,
        )
        self.assertEqual([], issues)

    def test_returns_stable_source_addressable_issues_without_fact_values(self):
        issues = validate_prompt_contract(
            self.prompt.replace("silver coat", "missing"),
            facts=self.facts,
            shots=self.shots,
            shot_windows={"shot-0001": (0, 3), "shot-0002": (2, 5)},
            references={"shot-0001": ["<Picture 9>"], "shot-0002": ["<Picture 2>"]},
            prepared_reference_labels={"<Picture 1>", "<Picture 2>"},
            duration_seconds=4,
        )

        self.assertEqual(
            ["fact.missing", "reference.unknown", "timing.overlap", "timing.duration_exceeded"],
            [issue.code for issue in issues],
        )
        self.assertEqual("facts[0].value", issues[0].path)
        self.assertEqual("cast:hero", issues[0].source_id)
        self.assertNotIn("silver coat", issues[0].message)

    def test_h3_shape_validator_rejects_internal_compiler_format(self):
        issues = validate_h3_prompt_shape("FULL REFERENCE PROMPT\nScene: scene-01", mode="r2v")
        self.assertEqual(["h3.sections.missing"], [issue.code for issue in issues])

    def test_h3_shape_validator_accepts_compiled_base_format(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] action\n\n"
            "overall_soundscape: N/A\n\n"
            "non_diegetic_music: N/A"
        )
        self.assertEqual([], validate_h3_prompt_shape(prompt, mode="t2v"))

    def test_h3_shape_validator_rejects_implicit_audio_without_gating_creative_length(self):
        prompt = (
            "subject_definitions:\n"
            "<Audio 1> is the complete soundtrack and is fully copied.\n\n"
            "summary: [reference generation] A performance.\n\n"
            "retention_analysis:\n"
            "<Audio 1>: fully_copy - reused 1:1.\n\n"
            "detailed_description: The target video is cinematic.\n"
            "[Shot 1] A performer crosses the room with <Audio 1> active in the soundtrack.\n\n"
            "overall_soundscape: No additional ambience.\n\n"
            "non_diegetic_music: N/A"
        )

        issues = validate_h3_prompt_contract(
            prompt,
            mode="r2v",
            plan=ResolvedPromptPlan(
                creative_intent="A performance.",
                style_opening="The target video is cinematic.",
                shots=[PlannedShot(
                    shot_number=1,
                    description="A performer crosses the room.",
                    start_seconds=0,
                    end_seconds=4,
                    reference_labels=["<Audio 1>"],
                )],
                overall_soundscape="No additional ambience.",
                music_intent=MusicIntent.NONE,
            ),
            reference_metadata=[{"label": "<Audio 1>", "kind": "audio"}],
        )

        self.assertNotIn("h3.detail.too_short", [issue.code for issue in issues])
        self.assertNotIn("h3.audio.relationship_missing", [issue.code for issue in issues])
        self.assertIn("h3.audio.summary_missing", [issue.code for issue in issues])

    def test_h3_contract_rejects_wrong_shot_timing_alias_and_retention_speaker(self):
        rich = " ".join(["Detailed cinematic action and composition remain explicit."] * 55)
        prompt = (
            "subject_definitions:\n"
            "<Subject 1> is the singer in <Picture 1>, wearing a silver coat.\n\n"
            "summary: [reference generation] <Subject 1> performs.\n\n"
            "retention_analysis:\n"
            "<Subject 1> (S1) (appears in [Shot 1], [Shot 2]): fully_preserved - identity retained.\n\n"
            "detailed_description: The target video is cinematic.\n"
            f"[Shot 1] Lead Singer begins moving. {rich}\n"
            "[Shot 2] At 00:04.000, <Subject 1> stops.\n\n"
            "overall_soundscape: Room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        plan = ResolvedPromptPlan(
            creative_intent="A performance.",
            style_opening="The target video is cinematic.",
            subjects=[SubjectDefinition(
                label="<Subject 1>",
                name="Lead Singer",
                description="the singer in a silver coat",
                source_references=["<Picture 1>"],
            )],
            shots=[
                PlannedShot(shot_number=1, description="First.", start_seconds=0, end_seconds=3),
                PlannedShot(shot_number=2, description="Second.", start_seconds=3, end_seconds=6),
            ],
            overall_soundscape="Room tone.",
            music_intent=MusicIntent.NONE,
        )

        issues = validate_h3_prompt_contract(prompt, mode="r2v", plan=plan)
        codes = [issue.code for issue in issues]

        self.assertIn("h3.shot.timestamp", codes)
        self.assertIn("h3.subject.alias", codes)
        self.assertIn("h3.retention.speaker", codes)

    def test_h3_contract_does_not_apply_generation_word_floor_to_video_editing(self):
        prompt = (
            "subject_definitions:\n\n"
            "summary: [reference generation + video editing] The source video is edited.\n\n"
            "retention_analysis:\n\n"
            "detailed_description: Edited source style.\n"
            "[Shot 1] The source shot receives a precise color edit.\n\n"
            "overall_soundscape: Source room tone remains.\n\n"
            "non_diegetic_music: N/A"
        )
        plan = ResolvedPromptPlan(
            creative_intent="Edit the source video.",
            style_opening="Edited source style.",
            shots=[PlannedShot(
                shot_number=1,
                description="The source shot receives a precise color edit.",
                start_seconds=0,
                end_seconds=4,
            )],
            overall_soundscape="Source room tone remains.",
            music_intent=MusicIntent.NONE,
        )

        issues = validate_h3_prompt_contract(prompt, mode="r2v", plan=plan)

        self.assertNotIn("h3.detail.too_short", [issue.code for issue in issues])


if __name__ == "__main__":
    unittest.main()
