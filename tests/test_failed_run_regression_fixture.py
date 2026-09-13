from __future__ import annotations

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
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.h3_audio_delivery import load_h3_audio_delivery
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.prompting.concept_prompt_batcher import ConceptPromptBatcher
from feverslop.prompting.dspy_h3_models import MusicIntent, PlannedShot, ResolvedPromptPlan
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder

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


class _FixedH3Generator:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            plan=ResolvedPromptPlan(
                creative_intent="Ravena drinks from the silver cup.",
                style_opening="Live-action cinematic imagery uses cool grotto light.",
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
                "performance_intervals": [
                    {
                        "performance_phase": True,
                        "start_seconds": 0.0,
                        "end_seconds": 3.542,
                        "state": "singing",
                        "lyrics": "through the fire",
                        "word_timestamps": evidence["word_timestamps"],
                        "acoustically_verified": False,
                        "reason_codes": evidence["reason_codes"],
                        "vocal_events": [
                            {
                                "lyrics": "through the fire",
                                "subject_id": evidence["performer_id"],
                                "speaker_id": evidence["speaker_id"],
                                "word_timestamps": evidence["word_timestamps"],
                            }
                        ],
                    }
                ],
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
            self.assertEqual(
                "non_vocal_uncertain_evidence",
                generator.requests[0]["relay_segments"][0]["performance_fallback"],
            )

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
                "prompt_corrections_applied": "No sung vocal performance" in prepared_request["prompt"],
                "request_created": result_path.is_file(),
                "consumer_received_expected_data": prepared_request["prompt"] == h3["prompt"],
                "render_completed": False,
                "output_path": result_path.as_posix(),
            }

            self.assertEqual(
                {
                    "cast_applied": True,
                    "prompt_corrections_applied": True,
                    "request_created": True,
                    "consumer_received_expected_data": True,
                    "render_completed": False,
                    "output_path": result_path.as_posix(),
                },
                result,
            )
            self.assertFalse(report.groups["vocal_delivery"].passed)
            self.assertIn("scene_015.h3.prompt", "\n".join(report.failures))
            self.assertFalse((render_output / "scene_0015" / "raw.mp4").exists())


if __name__ == "__main__":
    unittest.main()
