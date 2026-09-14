from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.adapters.comfyui_minimax_h3_r2v_backend import (
    ComfyUIMiniMaxH3R2VBackend,
)
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_video import (
    RenderVideoScenesRequest,
    RenderVideoScenesUseCase,
)
from feverslop.application.generate_render_plan import GenerateRenderPlanUseCase
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.h3_audio_delivery import load_h3_audio_delivery
from feverslop.domain.performance_timeline import (
    lean_performance_projection,
    project_performance,
)
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.prompting.concept_prompt_batcher import (
    ConceptPromptBatcher,
    validate_and_annotate_concept_chronology,
)
from feverslop.prompting.dspy_h3_models import (
    MusicIntent,
    PlannedShot,
    ResolvedPromptPlan,
    SubjectDefinition,
)
from feverslop.prompting.dspy_h3_prompt_builder import (
    DspyH3PromptBuilder,
    apply_narrative_continuity_to_h3,
)
from feverslop.prompting.scene_prompt_builder import (
    ScenePromptBuilder,
)
from tests.prompt_fakes import GeneralModulesFake

from feverslop.tools.regression_fixture import (
    evaluate_regression_invariants,
    load_regression_fixture,
    verify_baseline_provenance,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "well_of_youth_regression"
    / "regression_fixture.json"
)
ROOT = Path(__file__).resolve().parents[1]


class _FixedConceptModules:
    def concepts(self, _payload, *, batch=False, silent_mode=False, timeout=None):
        del batch, silent_mode, timeout
        return {
            "segment_015": {
                "concept": "Ravena drinks from the silver cup in the fountain grotto.",
                "references": {"actor_ids": ["ravena"]},
            }
        }

    def repair_concepts(self, _payload, *, timeout=None):
        raise AssertionError(f"valid fixed concept must not be repaired: {timeout}")

    def summary(self, _payload, *, timeout=None):
        del timeout
        return "Ravena reaches the fountain."


class _SequenceConceptModules:
    def __init__(self, responses):
        self.responses = iter(responses)

    def concepts(self, _payload, *, batch=False, silent_mode=False, timeout=None):
        del batch, silent_mode, timeout
        return next(self.responses)

    def repair_concepts(self, _payload, *, timeout=None):
        del timeout
        return next(self.responses)

    def summary(self, _payload, *, timeout=None):
        del timeout
        return next(self.responses)


class _ConceptValidationService:
    def __init__(self, batcher, *, segments, global_context):
        self.batcher = batcher
        self.segments = segments
        self.global_context = global_context

    def execute(self, context):
        context["concept_prompts"] = self.batcher.create_concept_prompts_batched(
            stage1_segments=self.segments,
            story_idea="Ravena completes the fountain rite once.",
            global_context=self.global_context,
        )
        return context


class _RecordingService:
    def __init__(self):
        self.calls = 0

    def execute(self, context):
        self.calls += 1
        return context


class _FixedH3Generator:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            plan=ResolvedPromptPlan(
                creative_intent="Ravena drinks from the silver cup.",
                style_opening="Live-action cinematic imagery uses cool grotto light.",
                subjects=[SubjectDefinition(
                    label="<Subject 1>",
                    name="Ravena",
                    description="Ravena is the visible vocalist.",
                    source_references=["<Picture 1>"],
                )],
                shots=[
                    PlannedShot(
                        shot_number=1,
                        start_seconds=0.0,
                        end_seconds=3.542,
                        description="Ravena drinks from the silver cup.",
                        camera_behavior="static close shot",
                    )
                ],
                overall_soundscape="Quiet grotto ambience.",
                music_intent=MusicIntent.NONE,
            )
        )


class _CaptureAssetUploader:
    def resolve_reference_image_name(self, path, **_kwargs):
        return f"fixture/{Path(path).name}"

    def resolve_reference_audio_name(self, path, **_kwargs):
        return f"fixture/{Path(path).name}"

    def resolve_reference_video_name(self, path, **_kwargs):
        return f"fixture/{Path(path).name}"


class _CaptureOnlyQueue:
    def __init__(self):
        self.workflows = []

    def queue_workflow_and_download_first_video(
        self, workflow, *, scene_number, output_path,
    ):
        self.workflows.append((scene_number, workflow))
        return output_path


class _OfflineClient:
    pass


class FailedRunRegressionFixtureTests(unittest.TestCase):
    @staticmethod
    def _cup_concept(scene, *, story_beat, action, action_phase="completed"):
        return {
            "concept": f"Ravena performs {action.replace('_', ' ')} at the fountain.",
            "references": {
                "actor_ids": ["ravena"],
                "location_id": "fountain_grotto",
            },
            "narrative": {
                **scene["narrative"],
                "story_beat": story_beat,
                "objective": "complete_the_fountain_rite",
                "action": action,
                "action_phase": action_phase,
                "reset_events": scene["narrative"].get("reset_events", []),
            },
        }

    def test_known_bad_cup_slice_stops_before_h3_and_render_services(self):
        fixture = load_regression_fixture(FIXTURE)
        by_scene = {
            scene["scene"]: scene
            for scene in fixture["projections"]["known_bad"]["scenes"]
        }
        segments = [
            {"segment_id": f"segment_{number:03d}", "scene": number}
            for number in (10, 11, 15)
        ]
        bad = {
            "segment_010": self._cup_concept(
                by_scene[10], story_beat="raise_cup", action="raise_cup",
            ),
            "segment_011": self._cup_concept(
                by_scene[11], story_beat="raise_cup", action="raise_cup",
            ),
            "segment_015": self._cup_concept(
                by_scene[15], story_beat="drink_from_cup", action="drink_from_cup",
            ),
        }
        modules = _SequenceConceptModules([bad, {
            "segment_011": bad["segment_011"],
            "segment_015": bad["segment_015"],
        }])
        h3_service = _RecordingService()
        render_service = _RecordingService()
        validator = _ConceptValidationService(
            ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=3),
            segments=segments,
            global_context={
                "actors": [{"id": "ravena", "name": "Ravena"}],
                "invariant_contract": fixture["invariant_contract"],
            },
        )

        with self.assertRaisesRegex(
            ValueError,
            r"segment_011.*duplicates segment_010.*cup_raised.*segment_010",
        ):
            GenerateRenderPlanUseCase(
                pipeline_services=[validator, h3_service, render_service],
            ).execute_services({"request": SimpleNamespace(defer_h3_until_references=False)})

        self.assertEqual(0, h3_service.calls)
        self.assertEqual(0, render_service.calls)

    def test_three_scene_cup_progression_reaches_distinct_render_plan_requests(self):
        fixture = load_regression_fixture(FIXTURE)
        control = fixture["projections"]["corrected_control"]
        control_by_scene = {scene["scene"]: scene for scene in control["scenes"]}
        scene_numbers = (10, 11, 15)
        actions = {
            10: ("acquire_cup", "acquire_cup"),
            11: ("raise_cup", "raise_cup"),
            15: ("drink_from_cup", "drink_from_cup"),
        }
        segments = [{
            "segment_id": f"segment_{number:03d}",
            "scene": number,
            "type": "instrumental",
            "start": float(index),
            "end": float(index + 1),
            "duration": 1.0,
        } for index, number in enumerate(scene_numbers)]
        generated = {
            segment["segment_id"]: self._cup_concept(
                control_by_scene[segment["scene"]],
                story_beat=actions[segment["scene"]][0],
                action=actions[segment["scene"]][1],
            )
            for segment in segments
        }
        global_context = {
            "actors": [{"id": "ravena", "name": "Ravena"}],
            "subject": "Ravena",
            "story_idea": "Ravena completes the fountain rite once.",
            "style": "cinematic grotto",
            "locations": ["fountain_grotto"],
            "structured_locations": [{"id": "fountain_grotto"}],
            "prompt_guidance": {},
            "invariant_contract": fixture["invariant_contract"],
        }
        concepts = ConceptPromptBatcher(
            object(),
            prompt_modules=_SequenceConceptModules([generated, "control summary"]),
            batch_size=3,
        ).create_concept_prompts_batched(
            stage1_segments=segments,
            story_idea=global_context["story_idea"],
            global_context=global_context,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            plan_path = temp / "render_plan.json"
            ScenePromptBuilder(
                object(),
                modules=GeneralModulesFake(
                    zimage="Ravena handles the silver cup at the fountain.",
                    i2v="Ravena completes one deliberate action with the silver cup.",
                ),
            ).build_scene_prompts(
                stage1_segments=segments,
                concept_prompts=concepts,
                scene_details={segment["segment_id"]: {} for segment in segments},
                global_context=global_context,
                output_json_path=scene_path,
                artifact_store=JsonArtifactStore(),
            )
            relay_path.write_text(json.dumps([
                {"scene": number, "prompt_relay": []}
                for number in scene_numbers
            ]), encoding="utf-8")
            build_render_plan(
                scene_path,
                relay_path,
                plan_path,
                VideoSettings(fps=24, width=640, height=352, megapixels=0.2),
                artifact_store=JsonArtifactStore(),
                seed=117210,
            )
            requests = json.loads(plan_path.read_text(encoding="utf-8"))

        validations = [scene["metadata"]["semantic_validation"] for scene in requests]
        self.assertEqual(list(scene_numbers), [scene["scene"] for scene in requests])
        self.assertEqual([117210, 117210, 117210], [scene["seed"] for scene in requests])
        self.assertEqual(3, len({item["state_signature"] for item in validations}))
        self.assertTrue(all(item["outcome"] == "accepted" for item in validations))
        self.assertTrue(all(not item["authorized_reprise"] for item in validations))
        self.assertEqual(
            ["acquired", "raised", "consumed"],
            [scene["metadata"]["narrative"]["props"]["silver_cup"] for scene in requests],
        )

    def test_fixture_is_compact_portable_and_covers_only_reviewed_slices(self):
        fixture = load_regression_fixture(FIXTURE)

        self.assertEqual("feverslop.failed-run-regression/v1", fixture["schema"])
        self.assertEqual(
            [10, 11, 14, 15, 20, 21, 37, 38, 51],
            [scene["scene"] for scene in fixture["projections"]["known_bad"]["scenes"]],
        )
        self.assertEqual(15, fixture["projections"]["known_bad"]["vocal_case"]["scene"])
        for source in fixture["provenance"]["baseline_files"]:
            self.assertFalse(Path(source["path"]).is_absolute())
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")

    def test_chronology_fixture_requires_machine_readable_evidence(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        del payload["chronology_cases"]["corrected_sequence"]["evidence"][
            "milestone_allocation"
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_path = Path(temp_dir) / "regression_fixture.json"
            fixture_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "chronology case 'corrected_sequence'.*milestone_allocation",
            ):
                load_regression_fixture(fixture_path)

    def test_six_scene_chronology_reaches_the_final_render_plan(self):
        fixture = load_regression_fixture(FIXTURE)
        case = fixture["chronology_cases"]["corrected_sequence"]
        segments = []
        generated = {}
        for index, item in enumerate(case["scenes"]):
            scene_number = item["scene"]
            segment_id = f"segment_{scene_number:03d}"
            milestone = item["narrative"]["milestones"][0]
            segments.append({
                "segment_id": segment_id,
                "scene": scene_number,
                "type": "instrumental",
                "start": float(index),
                "end": float(index + 1),
                "duration": 1.0,
            })
            generated[segment_id] = {
                "concept": f"Ravena completes {milestone.replace('_', ' ')}.",
                "references": {
                    "actor_ids": ["ravena"],
                    "location_id": item["narrative"]["location"],
                },
                "narrative": {
                    **item["narrative"],
                    "story_beat": milestone,
                    "objective": "complete_the_well_of_youth_journey",
                    "action": milestone,
                    "action_phase": "completed",
                    "props": {},
                    "reset_events": [],
                },
            }

        global_context = {
            "actors": [{"id": "ravena", "name": "Ravena"}],
            "subject": "Ravena",
            "story_idea": "Ravena completes the ordered Well of Youth journey.",
            "style": "cinematic gothic fantasy",
            "locations": [entry["id"] for entry in fixture["chronology_contract"]["location_order"]],
            "structured_locations": fixture["chronology_contract"]["location_order"],
            "prompt_guidance": {},
            "narrative_contract": fixture["chronology_contract"],
        }
        concepts = ConceptPromptBatcher(
            object(),
            prompt_modules=_SequenceConceptModules([generated, "journey complete"]),
            batch_size=6,
        ).create_concept_prompts_batched(
            stage1_segments=segments,
            story_idea=global_context["story_idea"],
            global_context=global_context,
        )
        concepts = validate_and_annotate_concept_chronology(
            concepts,
            fixture["chronology_contract"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            plan_path = temp / "render_plan.json"
            ScenePromptBuilder(
                object(),
                modules=GeneralModulesFake(
                    zimage="Ravena advances through the ordered journey.",
                    i2v="Ravena completes the current ordered milestone.",
                ),
            ).build_scene_prompts(
                stage1_segments=segments,
                concept_prompts=concepts,
                scene_details={segment["segment_id"]: {} for segment in segments},
                global_context=global_context,
                output_json_path=scene_path,
                artifact_store=JsonArtifactStore(),
            )
            relay_path.write_text(json.dumps([
                {"scene": segment["scene"], "prompt_relay": []}
                for segment in segments
            ]), encoding="utf-8")
            build_render_plan(
                scene_path,
                relay_path,
                plan_path,
                VideoSettings(fps=24, width=640, height=352, megapixels=0.2),
                artifact_store=JsonArtifactStore(),
                seed=1175,
            )
            render_plan = json.loads(plan_path.read_text(encoding="utf-8"))

        expected = case["evidence"]
        final_chronology = render_plan[-1]["metadata"]["semantic_validation"]["chronology"]
        self.assertEqual(expected["scene_order"], [item["scene"] for item in render_plan])
        self.assertEqual(
            list(expected["milestone_allocation"]),
            [item["metadata"]["narrative"]["milestones"][0] for item in render_plan],
        )
        self.assertEqual(
            [f"segment_{scene:03d}" for scene in expected["scene_order"]],
            final_chronology["scene_order"],
        )
        self.assertEqual(
            {
                milestone: f"segment_{scene:03d}"
                for milestone, scene in expected["milestone_allocation"].items()
            },
            final_chronology["milestone_allocation"],
        )
        self.assertEqual("accepted", final_chronology["validation_result"])
        self.assertIsNone(final_chronology["approved_exception"])
        self.assertEqual(
            "ascended_absent",
            render_plan[-1]["metadata"]["narrative"]["cast_states"]["ravena"],
        )

    def test_fixture_authorized_flashback_passes_complete_story_validation(self):
        fixture = load_regression_fixture(FIXTURE)
        case = fixture["chronology_cases"]["authorized_flashback"]
        concepts = {}
        for item in case["scenes"]:
            milestone = item["narrative"]["milestones"][0]
            concepts[f"segment_{item['scene']:03d}"] = {
                "concept": f"Ravena completes {milestone.replace('_', ' ')}.",
                "narrative": {
                    **item["narrative"],
                    "story_beat": (
                        "cave_flashback"
                        if item["narrative"].get("causal_events")
                        else milestone
                    ),
                    "objective": "complete_the_well_of_youth_journey",
                    "action": (
                        "remember_descent"
                        if item["narrative"].get("causal_events")
                        else milestone
                    ),
                    "action_phase": "completed",
                    "props": {},
                    "reset_events": [],
                },
            }

        accepted = validate_and_annotate_concept_chronology(
            concepts,
            fixture["chronology_contract"],
        )

        flashback = accepted["segment_005"]["semantic_validation"]["chronology"]
        self.assertEqual("accepted", case["evidence"]["validation_result"])
        self.assertEqual(case["evidence"]["approved_exception"], flashback["approved_exception"])

    def test_available_preserved_baseline_matches_recorded_hashes(self):
        fixture = load_regression_fixture(FIXTURE)

        checked = verify_baseline_provenance(fixture, ROOT)

        if (ROOT / fixture["provenance"]["source_project"]).is_dir():
            self.assertEqual(6, len(checked))
        else:
            self.assertEqual((), checked)

    def test_fixture_does_not_require_the_ignored_baseline_in_a_clean_clone(self):
        fixture = load_regression_fixture(FIXTURE)

        with tempfile.TemporaryDirectory() as clean_checkout:
            self.assertEqual(
                (), verify_baseline_provenance(fixture, clean_checkout),
            )

    def test_known_bad_projection_identifies_each_preserved_defect_independently(self):
        fixture = load_regression_fixture(FIXTURE)

        report = evaluate_regression_invariants(fixture, "known_bad")

        self.assertEqual(
            {"vocal_delivery", "milestone_uniqueness", "chronology", "adjacent_continuity"},
            set(report.groups),
        )
        for group, result in report.groups.items():
            with self.subTest(group=group):
                self.assertFalse(result.passed)
                self.assertTrue(result.failures)

        messages = "\n".join(report.failures)
        self.assertIn("scene_015.h3.prompt", messages)
        self.assertIn("scene_011.narrative.milestones", messages)
        self.assertIn("scene_038.narrative.milestone_rank", messages)
        self.assertIn("scene_015.incoming.props.silver_cup", messages)

    def test_corrected_control_passes_the_same_objective_invariants(self):
        fixture = load_regression_fixture(FIXTURE)

        report = evaluate_regression_invariants(fixture, "corrected_control")

        self.assertTrue(report.passed, "\n".join(report.failures))
        self.assertTrue(all(group.passed for group in report.groups.values()))

    def test_follow_up_smoke_specs_separate_request_facts_from_visual_review(self):
        fixture = load_regression_fixture(FIXTURE)

        specs = fixture["smoke_specifications"]
        self.assertEqual(
            {"timed_vocal", "cup_progression", "terminal_state", "grotto_cave_boundary"},
            set(specs),
        )
        for name, spec in specs.items():
            with self.subTest(name=name):
                self.assertTrue(spec["request_assertions"])
                self.assertTrue(spec["visual_review"])
                self.assertNotEqual(spec["request_assertions"], spec["visual_review"])

    def test_continuity_evidence_contract_fails_closed_on_missing_scene_field(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["continuity_evidence"]["post_ascent_vocal_scenes"][0].pop(
            "predecessor_id", None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_path = Path(temp_dir) / "regression_fixture.json"
            fixture_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"continuity evidence scene 24.*predecessor_id",
            ):
                load_regression_fixture(fixture_path)

    def test_each_post_ascent_vocal_scene_keeps_ravena_offscreen_through_h3(self):
        fixture = load_regression_fixture(FIXTURE)

        evidence = fixture["continuity_evidence"]
        self.assertEqual(
            [24, 25, 27, 28, 40, 41, 42, 43, 44, 45, 46],
            [item["scene"] for item in evidence["post_ascent_vocal_scenes"]],
        )
        self.assertEqual(
            {
                "predecessor_id", "incoming", "outgoing",
                "transition_events", "requires_continuation",
            },
            set(evidence["continuity_state_fields"]),
        )
        continuity_by_scene = {}
        concepts = {}
        segments = []
        for item in evidence["post_ascent_vocal_scenes"]:
            continuity = {
                "schema": "feverslop.narrative-continuity/v1",
                "scene_id": item["segment_id"],
                **{
                    field: item[field]
                    for field in evidence["continuity_state_fields"]
                },
                "transition": "continuous" if item["requires_continuation"] else "cut",
                "continuation_intent": None,
                "validation_result": "compatible",
            }
            continuity_by_scene[item["scene"]] = continuity
            segments.append({
                "segment_id": item["segment_id"],
                "scene": item["scene"],
                "type": "vocals",
                "start": float(item["scene"]),
                "end": float(item["scene"] + 1),
                "duration": 1.0,
            })
            concepts[item["segment_id"]] = {
                "concept": "Varen and Silas cross the cavern while Ravena sings off-screen.",
                "references": {"actor_ids": item["visual_actor_ids"]},
                "narrative": {
                    "location": item["incoming"]["location"],
                    "cast_states": {"varen": "present", "silas": "present"},
                },
                "semantic_validation": {"continuity": continuity},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene_prompts.json"
            ScenePromptBuilder(
                object(),
                modules=GeneralModulesFake(
                    zimage="Varen and Silas cross the cavern.",
                    i2v="Varen and Silas move through the same light.",
                ),
            ).build_scene_prompts(
                stage1_segments=segments,
                concept_prompts=concepts,
                scene_details={item["segment_id"]: {} for item in evidence["post_ascent_vocal_scenes"]},
                global_context={
                    "actors": [
                        {"id": "ravena", "name": "Ravena"},
                        {"id": "varen", "name": "Varen"},
                        {"id": "silas", "name": "Silas"},
                        {"id": "well_guardian", "name": "Well-Guardian"},
                    ],
                    "subject": "Varen and Silas",
                    "story_idea": "The companions leave after Ravena ascends.",
                    "style": "cinematic gothic fantasy",
                    "locations": ["fountain_grotto", "weeping_caves"],
                    "structured_locations": [
                        {"id": "fountain_grotto"},
                        {"id": "weeping_caves"},
                    ],
                    "audio_subject_bindings": {
                        "vocals": {"subject_id": "ravena", "speaker_id": "S1"},
                    },
                    "prompt_guidance": {},
                },
                output_json_path=scene_path,
                artifact_store=JsonArtifactStore(),
            )
            rendered_scenes = {
                item["scene"]: item
                for item in json.loads(scene_path.read_text(encoding="utf-8"))
            }

        for item in evidence["post_ascent_vocal_scenes"]:
            with self.subTest(scene=item["scene"]):
                references = rendered_scenes[item["scene"]]["references"]
                continuity = continuity_by_scene[item["scene"]]
                h3 = apply_narrative_continuity_to_h3(
                    {"prompt": "Varen and Silas cross the cavern."},
                    {"semantic_validation": {"continuity": continuity}},
                )

                self.assertEqual(item["visual_actor_ids"], references["actor_ids"])
                self.assertNotIn("ravena", references["actor_ids"])
                self.assertNotIn("audio_subject_bindings", references)
                self.assertEqual(
                    item["offscreen_audio_subject_bindings"],
                    references["offscreen_audio_subject_bindings"],
                )
                self.assertEqual(continuity, h3["continuity_plan"])
                self.assertIn("Ravena remains ascended and absent", h3["prompt"])
                self.assertIn("off-screen", h3["prompt"])

    def test_scene_15_reaches_the_final_workflow_consumer_with_machine_readable_result(self):
        fixture = load_regression_fixture(FIXTURE)
        bad_case = fixture["projections"]["known_bad"]["vocal_case"]
        evidence = bad_case["timeline_evidence"]
        workflow_template = (
            ROOT
            / "workflows"
            / "video"
            / "minimax_h3"
            / "r2v_audio_fullmix_guide_1-pass_turbo.json"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            actor = project / "actor.png"
            vocals = project / "vocals.wav"
            song = project / "song.wav"
            actor.write_bytes(b"deterministic actor reference")
            vocals.write_bytes(b"deterministic vocal guide")
            song.write_bytes(b"deterministic full mix")
            performance = lean_performance_projection(project_performance([{
                "type": "vocals",
                "start": bad_case["interval_seconds"][0],
                "end": bad_case["interval_seconds"][1],
                "evidence": {
                    "activity_status": "conflict",
                    "transcript_status": evidence["transcript_status"],
                    "reason_codes": ["transcript_crosses_rms_boundary"],
                },
                "alignment": {
                    "timed_words": evidence["word_timestamps"],
                    "targets": [{"word": "unresolved", "source": "unresolved"}],
                },
            }], *bad_case["interval_seconds"]))
            self.assertEqual("accepted", performance[0].get("transcript_status"))
            self.assertIn("uncertain_vocal_evidence", performance[0]["reason_codes"])
            self.assertIn("unresolved_lyric_alignment", performance[0]["reason_codes"])

            segment = {
                "segment_id": "segment_015",
                "scene": 15,
                "start": bad_case["interval_seconds"][0],
                "end": bad_case["interval_seconds"][1],
                "duration": 3.542,
                "type": "vocals",
                "references": {
                    "actor_ids": ["ravena"],
                    "actor_sheet_paths": [actor.name],
                    "audio_subject_bindings": {
                        "vocals": {
                            "subject_id": evidence["performer_id"],
                            "speaker_id": evidence["speaker_id"],
                        }
                    },
                },
                "performance_intervals": performance,
            }
            concepts = ConceptPromptBatcher(
                object(), prompt_modules=_FixedConceptModules(), batch_size=1,
            ).create_concept_prompts_batched(
                stage1_segments=[segment],
                story_idea="Ravena reaches the well and drinks once before ascending.",
                global_context={
                    "actors": [{"id": "ravena", "name": "Ravena"}],
                    "locations": ["fountain_grotto"],
                },
            )

            generator = _FixedH3Generator()
            delivery = load_h3_audio_delivery(workflow_template)
            h3 = DspyH3PromptBuilder(
                generator, reference_root=project, allow_fallback=False,
            ).build_h3_prompt(
                segment=segment,
                concept=concepts["segment_015"]["concept"],
                scene_details={},
                global_context={"h3_audio_delivery": delivery.to_context()},
                mode="r2v",
                audio_paths={"vocals": vocals},
            )
            relay_phase = generator.requests[0]["relay_segments"][0]
            self.assertEqual("singing", relay_phase["state"])
            self.assertNotIn("performance_fallback", relay_phase)
            self.assertEqual("accepted", relay_phase["transcript_status"])

            store = JsonArtifactStore()
            scene_prompts_path = project / "scene_prompts.json"
            relay_path = project / "relay.json"
            h3_path = project / "h3.json"
            render_plan_path = project / "render_plan.json"
            store.write_json(scene_prompts_path, [{
                **segment,
                "zimage_prompt": "Ravena holds a silver cup in the fountain grotto.",
                "base_concept": concepts["segment_015"]["concept"],
            }])
            store.write_json(relay_path, [{
                "scene": 15,
                "performance_intervals": segment["performance_intervals"],
                "prompt_relay": [{
                    **segment["performance_intervals"][0],
                    "frame_start": 0,
                    "frame_end": 85,
                }],
            }])
            store.write_json(h3_path, [{"segment_id": "segment_015", **h3}])
            build_render_plan(
                scene_prompts_path,
                relay_path,
                render_plan_path,
                VideoSettings(fps=24, width=640, height=352, megapixels=0.2),
                artifact_store=store,
                h3_prompts_json=h3_path,
                project_dir=project,
            )

            queue = _CaptureOnlyQueue()
            render_output = project / "output" / "render"
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=_OfflineClient(),
                workflow_path=workflow_template,
                output_dir=render_output,
                project_dir=project,
                asset_uploader=_CaptureAssetUploader(),
                render_queue=queue,
                postprocess=False,
                seed_offset=117200,
            )
            RenderVideoScenesUseCase(backend, store).execute(
                RenderVideoScenesRequest(
                    render_plan_path=render_plan_path,
                    workflow_path=workflow_template,
                    audio_file=song,
                    storyboard_dir=project / "storyboard",
                    output_dir=render_output,
                    scene_numbers={15},
                    skip_existing=False,
                )
            )

            self.assertEqual(1, len(queue.workflows))
            prepared = queue.workflows[0][1]
            prompt_node = next(
                node for node in prepared.values()
                if (node.get("_meta") or {}).get("title") == "#PROMPT"
            )
            audio_node_id, audio_node = next(
                (node_id, node) for node_id, node in prepared.items()
                if (node.get("_meta") or {}).get("title") == "#AUDIO_1"
            )
            trim_node_id, trim_node = next(
                (node_id, node) for node_id, node in prepared.items()
                if (node.get("_meta") or {}).get("title") == "#TRIM_AUDIO_1"
            )
            guide_node = next(
                node for node in prepared.values()
                if (node.get("_meta") or {}).get("title")
                == "#EXPERIMENTAL_FULLMIX_AUDIO_GUIDE_FRAME_0"
            )
            plan_scene = store.read_render_plan(render_plan_path)[0]
            prepared_request = {
                "prompt": prompt_node["inputs"]["value"],
                "audio_subject_binding": plan_scene["references"][
                    "audio_subject_bindings"
                ]["vocals"],
                "audio_inputs": [{
                    "name": "vocals",
                    "roles": h3["h3_audio_sources"][0]["roles"],
                }],
            }
            report = evaluate_regression_invariants(
                fixture, "known_bad", prepared_request=prepared_request,
            )
            result_path = render_output / "scene_0015" / "workflow.json"
            result = {
                "cast_applied": plan_scene["references"]["actor_ids"] == ["ravena"],
                "prompt_corrections_applied": "sings with visible mouth movements" in prepared_request["prompt"],
                "request_created": result_path.is_file(),
                "consumer_received_expected_data": prepared_request["prompt"] == h3["prompt"],
                "render_completed": False,
                "output_path": result_path.as_posix(),
                "accepted_transcript_preserved": relay_phase["transcript_status"] == "accepted",
                "word_timing_preserved": [
                    (word["word"], word["start"], word["end"])
                    for word in relay_phase["word_timestamps"]
                ] == [
                    (
                        word["word"],
                        max(word["start"], bad_case["interval_seconds"][0]),
                        min(word["end"], bad_case["interval_seconds"][1]),
                    )
                    for word in evidence["word_timestamps"]
                ],
                "s1_binding_preserved": prepared_request["audio_subject_binding"] == {
                    "subject_id": "ravena", "speaker_id": "S1",
                },
                "vocal_guide_bound": (
                    audio_node["inputs"]["audio"] == "fixture/vocals.wav"
                    and trim_node["inputs"]["audio"] == [audio_node_id, 0]
                    and guide_node["inputs"]["audio"] == [trim_node_id, 0]
                ),
                "prompt_contradiction_absent": not any(
                    text in prepared_request["prompt"].casefold()
                    for text in (
                        "no sung vocal performance", "no vocal performance",
                        "mouth closed", "do not create lip-sync",
                    )
                ),
            }

            self.assertEqual(
                {
                    "cast_applied": True,
                    "prompt_corrections_applied": True,
                    "request_created": True,
                    "consumer_received_expected_data": True,
                    "render_completed": False,
                    "output_path": result_path.as_posix(),
                    "accepted_transcript_preserved": True,
                    "word_timing_preserved": True,
                    "s1_binding_preserved": True,
                    "vocal_guide_bound": True,
                    "prompt_contradiction_absent": True,
                },
                result,
            )
            self.assertTrue(report.groups["vocal_delivery"].passed, "\n".join(report.failures))
            self.assertFalse((render_output / "scene_0015" / "raw.mp4").exists())


if __name__ == "__main__":
    unittest.main()
