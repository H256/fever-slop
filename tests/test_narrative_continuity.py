import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.comfyui_minimax_h3_r2v_backend import (
    ComfyUIMiniMaxH3R2VBackend,
)
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_video import (
    RenderVideoScenesRequest,
    RenderVideoScenesUseCase,
)
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.continuity import BoundaryFrameManifest
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.prompting.concept_prompt_batcher import (
    validate_and_annotate_concept_chronology,
)
from feverslop.prompting.dspy_h3_prompt_builder import (
    apply_narrative_continuity_to_h3,
)
from feverslop.prompting.scene_prompt_builder import normalize_scene_references


def _concept(
    *,
    location="fountain_grotto",
    cast_states=None,
    props=None,
    action="hold_cup",
    action_phase="holding",
    transition_events=None,
    transition_from_previous="cut",
):
    return {
        "concept": "A deterministic continuity fixture.",
        "references": {"actor_ids": ["ravena"], "location_id": location},
        "narrative": {
            "story_beat": action,
            "objective": "complete_the_rite",
            "action": action,
            "action_phase": action_phase,
            "milestones": [],
            "location": location,
            "cast_states": cast_states or {"ravena": "corporeal"},
            "props": props or {"silver_cup": "at_lips"},
            "transition_events": transition_events or [],
            "transition_from_previous": transition_from_previous,
        },
    }


class NarrativeContinuityTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "prop_state_order": {
                "silver_cup": [
                    "unseen", "acquired", "raised", "at_lips", "consumed", "lowered",
                ],
            },
            "actor_allowed_locations": {
                "well_guardian": ["fountain_grotto"],
            },
            "terminal_states": {
                "ravena": {
                    "milestone": "ascent_complete",
                    "state": "ascended_absent",
                    "reset_event": "ravena_returns",
                },
            },
        }

    def test_rejects_adjacent_prop_materialization_without_transition_event(self):
        previous = _concept(props={"silver_cup": "unseen"})
        current = _concept(props={"silver_cup": "at_lips"})

        with self.assertRaisesRegex(
            ValueError,
            r"segment_015\.incoming\.props\.silver_cup.*segment_014 outgoing state",
        ):
            validate_and_annotate_concept_chronology(
                {"segment_014": previous, "segment_015": current},
                self.contract,
            )

    def test_rejects_location_bound_actor_after_threshold_event(self):
        previous = _concept(cast_states={"ravena": "ascended_absent", "well_guardian": "present"})
        current = _concept(
            location="weeping_caves",
            cast_states={"ravena": "ascended_absent", "well_guardian": "present"},
            transition_events=["exit_grotto"],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"segment_038\.incoming\.cast_states\.well_guardian.*weeping_caves",
        ):
            validate_and_annotate_concept_chronology(
                {"segment_037": previous, "segment_038": current},
                self.contract,
            )

    def test_preserves_terminal_absence_when_next_scene_omits_actor_state(self):
        previous = _concept(cast_states={"ravena": "ascended_absent"})
        previous["narrative"]["milestones"] = ["ascent_complete"]
        current = _concept(cast_states={"varen": "present"}, action="kneel", action_phase="holding")

        annotated = validate_and_annotate_concept_chronology(
            {"segment_020": previous, "segment_021": current},
            self.contract,
        )

        continuity = annotated["segment_021"]["semantic_validation"]["continuity"]
        self.assertEqual("ascended_absent", continuity["incoming"]["cast_states"]["ravena"])
        self.assertEqual("ascended_absent", continuity["outgoing"]["cast_states"]["ravena"])
        self.assertEqual("segment_020", continuity["predecessor_id"])
        self.assertEqual("compatible", continuity["validation_result"])

    def test_rejects_terminal_actor_return_without_authored_event(self):
        previous = _concept(cast_states={"ravena": "ascended_absent"})
        previous["narrative"]["milestones"] = ["ascent_complete"]
        current = _concept(cast_states={"ravena": "corporeal"})

        with self.assertRaisesRegex(ValueError, r"ravena.*ascended_absent"):
            validate_and_annotate_concept_chronology(
                {"segment_020": previous, "segment_051": current},
                self.contract,
            )

    def test_marks_explicit_continuous_action_for_predecessor_handoff(self):
        previous = _concept(action="drink", action_phase="at_lips")
        current = _concept(
            action="drink",
            action_phase="at_lips",
            transition_from_previous="continuous",
        )

        annotated = validate_and_annotate_concept_chronology(
            {"segment_014": previous, "segment_015": current},
            self.contract,
        )

        continuity = annotated["segment_015"]["semantic_validation"]["continuity"]
        self.assertTrue(continuity["requires_continuation"])
        self.assertEqual("segment_014", continuity["predecessor_id"])
        self.assertEqual("drink", continuity["continuation_intent"])

    def test_rejects_continuous_action_phase_reset_at_boundary(self):
        previous = _concept(action="drink", action_phase="at_lips")
        current = _concept(
            action="drink",
            action_phase="completed",
            transition_from_previous="continuous",
        )
        current["narrative"]["incoming"] = {
            "location": "fountain_grotto",
            "cast_states": {"ravena": "corporeal"},
            "props": {"silver_cup": "at_lips"},
            "action": "drink",
            "action_phase": "starting",
        }

        with self.assertRaisesRegex(
            ValueError,
            r"segment_015\.incoming\.action_phase.*segment_014 outgoing state",
        ):
            validate_and_annotate_concept_chronology(
                {"segment_014": previous, "segment_015": current},
                self.contract,
            )

    def test_post_ascent_vocal_binding_does_not_restore_visual_ravena(self):
        references = normalize_scene_references(
            {"actor_ids": ["varen"]},
            {
                "actors": [{"id": "ravena"}, {"id": "varen"}],
                "audio_subject_bindings": {
                    "vocals": {"subject_id": "ravena", "speaker_id": "S1"},
                },
            },
            segment_type="vocals",
            narrative={"cast_states": {"ravena": "ascended_absent"}},
        )

        self.assertEqual(["varen"], references["actor_ids"])
        self.assertNotIn("audio_subject_bindings", references)
        self.assertEqual(
            {"vocals": {"subject_id": "ravena", "speaker_id": "S1"}},
            references["offscreen_audio_subject_bindings"],
        )

    def test_render_plan_carries_adjacent_continuation_to_predecessor_dependency(self):
        continuity = {
            "schema": "feverslop.narrative-continuity/v1",
            "scene_id": "segment_015",
            "predecessor_id": "segment_014",
            "transition": "continuous",
            "requires_continuation": True,
            "continuation_intent": "drink",
            "transition_events": [],
            "incoming": {
                "location": "fountain_grotto",
                "cast_states": {"ravena": "corporeal"},
                "props": {"silver_cup": "at_lips"},
                "action": "drink",
                "action_phase": "at_lips",
            },
            "outgoing": {
                "location": "fountain_grotto",
                "cast_states": {"ravena": "corporeal"},
                "props": {"silver_cup": "consumed"},
                "action": "drink",
                "action_phase": "completed",
            },
            "validation_result": "compatible",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "scenes.json"
            relay_path = root / "relay.json"
            h3_path = root / "h3.json"
            output_path = root / "render_plan.json"
            scenes = []
            for scene in (14, 15):
                scenes.append({
                    "scene": scene,
                    "segment_id": f"segment_{scene:03d}",
                    "type": "instrumental",
                    "start": float(scene - 14),
                    "end": float(scene - 13),
                    "duration": 1.0,
                    "zimage_prompt": "Ravena holds the cup.",
                    "ltx_base_prompt": "Ravena holds the cup.",
                })
            scene_path.write_text(json.dumps(scenes), encoding="utf-8")
            relay_path.write_text(json.dumps([
                {"scene": 14, "prompt_relay": []},
                {"scene": 15, "prompt_relay": []},
            ]), encoding="utf-8")
            h3_path.write_text(json.dumps([
                {"segment_id": "segment_014", "prompt": "Ravena holds the cup."},
                {
                    "segment_id": "segment_015",
                    "prompt": "Ravena drinks from the cup.",
                    "continuity_plan": continuity,
                    "continuation_intents": [{
                        "action_id": "drink",
                        "requires_continuation": True,
                    }],
                },
            ]), encoding="utf-8")

            build_render_plan(
                scene_path,
                relay_path,
                output_path,
                VideoSettings(fps=24, width=640, height=352),
                artifact_store=JsonArtifactStore(),
                h3_prompts_json=h3_path,
                project_dir=root,
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))[1]

        self.assertEqual("segment_014", result["continuation_predecessor_id"])
        self.assertEqual(continuity, result["continuity_plan"])
        self.assertEqual(continuity, result["metadata"]["continuity_plan"])
        self.assertEqual(
            "segment_014",
            result["narrative_boundary_manifest"]["predecessor_id"],
        )
        self.assertEqual(
            continuity["incoming"],
            result["narrative_boundary_manifest"]["incoming"],
        )

    def test_boundary_manifest_roundtrip_carries_narrative_state(self):
        continuity = {
            "schema": "feverslop.narrative-continuity/v1",
            "incoming": {"cast_states": {"ravena": "ascended_absent"}},
        }
        manifest = BoundaryFrameManifest.create(
            source_clip_path="output/scene_0020/final.mp4",
            source_clip_sha256="a" * 64,
            frame_index=47,
            extractor_revision="last-frame-v1",
            frame_path="output/keyframes/scene_0020_to_0021_start.png",
            frame_sha256="b" * 64,
            continuity_state=continuity,
        )

        restored = BoundaryFrameManifest.from_dict(manifest.to_dict())

        self.assertEqual(continuity, restored.continuity_state)
        instruction = ComfyUIMiniMaxH3R2VBackend._compile_continuity_start_state(
            restored,
            picture_slot=3,
        )
        self.assertIn("Ravena remains ascended and absent", instruction)

    def test_h3_result_carries_continuity_plan_and_terminal_absence(self):
        continuity = {
            "schema": "feverslop.narrative-continuity/v1",
            "scene_id": "segment_024",
            "predecessor_id": "segment_023",
            "transition": "cut",
            "requires_continuation": False,
            "continuation_intent": None,
            "transition_events": [],
            "incoming": {
                "location": "surface_passage",
                "cast_states": {"ravena": "ascended_absent"},
            },
            "outgoing": {
                "location": "surface_passage",
                "cast_states": {"ravena": "ascended_absent"},
            },
            "validation_result": "compatible",
        }

        result = apply_narrative_continuity_to_h3(
            {"prompt": "Varen and Silas continue toward the surface."},
            {"semantic_validation": {"continuity": continuity}},
        )

        self.assertEqual(continuity, result["continuity_plan"])
        self.assertEqual([], result["continuation_intents"])
        self.assertIn("Ravena remains ascended and absent", result["prompt"])
        self.assertIn("off-screen", result["prompt"])

    def test_final_h3_request_consumes_verified_boundary_and_manifest_state(self):
        class AssetUploader:
            def resolve_reference_image_name(self, path, **_kwargs):
                return f"fixture/{Path(path).name}"

            def resolve_reference_audio_name(self, path, **_kwargs):
                return f"fixture/{Path(path).name}"

            def resolve_reference_video_name(self, path, **_kwargs):
                return f"fixture/{Path(path).name}"

        class Queue:
            def __init__(self):
                self.workflows = []

            def queue_workflow_and_download_first_video(
                self, workflow, *, scene_number, output_path,
            ):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(f"scene {scene_number}".encode())
                self.workflows.append((scene_number, workflow))
                return output_path

        class Postprocessor:
            @staticmethod
            def extract_last_frame(source, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"verified predecessor frame")
                return destination

            @staticmethod
            def last_frame_index(_source):
                return 23

        continuity = {
            "schema": "feverslop.narrative-continuity/v1",
            "scene_id": "segment_015",
            "predecessor_id": "segment_014",
            "transition": "continuous",
            "requires_continuation": True,
            "continuation_intent": "drink",
            "transition_events": [],
            "incoming": {
                "location": "fountain_grotto",
                "cast_states": {"ravena": "ascended_absent"},
                "props": {"silver_cup": "at_lips"},
            },
            "outgoing": {
                "location": "fountain_grotto",
                "cast_states": {"ravena": "ascended_absent"},
                "props": {"silver_cup": "consumed"},
            },
            "validation_result": "compatible",
        }
        workflow_template = (
            Path(__file__).resolve().parents[1]
            / "workflows/video/minimax_h3/r2v_turbo_8s_v1.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actor = root / "varen.png"
            location = root / "grotto.png"
            audio = root / "song.wav"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            audio.write_bytes(b"audio")
            scene_path = root / "scenes.json"
            relay_path = root / "relay.json"
            h3_path = root / "h3.json"
            plan_path = root / "render_plan.json"
            common_refs = {
                "actor_ids": ["varen"],
                "actor_sheet_paths": [actor.name],
                "location_id": "fountain_grotto",
                "location_sheet_path": location.name,
            }
            scene_path.write_text(json.dumps([
                {
                    "scene": 14, "segment_id": "segment_014", "type": "instrumental",
                    "start": 0.0, "end": 1.0, "duration": 1.0,
                    "zimage_prompt": "Varen watches the empty light.",
                    "ltx_base_prompt": "Varen watches the empty light.",
                    "references": common_refs,
                },
                {
                    "scene": 15, "segment_id": "segment_015", "type": "instrumental",
                    "start": 1.0, "end": 2.0, "duration": 1.0,
                    "zimage_prompt": "Varen watches the empty light.",
                    "ltx_base_prompt": "Varen watches the empty light.",
                    "references": common_refs,
                },
            ]), encoding="utf-8")
            relay_path.write_text(json.dumps([
                {"scene": 14, "prompt_relay": []},
                {"scene": 15, "prompt_relay": []},
            ]), encoding="utf-8")
            h3_path.write_text(json.dumps([
                {"segment_id": "segment_014", "prompt": "Varen watches the empty light."},
                {
                    "segment_id": "segment_015",
                    "prompt": "Varen watches the empty light.",
                    "continuity_plan": continuity,
                    "continuation_intents": [{
                        "action_id": "drink", "requires_continuation": True,
                    }],
                },
            ]), encoding="utf-8")
            store = JsonArtifactStore()
            build_render_plan(
                scene_path, relay_path, plan_path,
                VideoSettings(fps=24, width=640, height=352, megapixels=0.2),
                artifact_store=store,
                h3_prompts_json=h3_path,
                project_dir=root,
            )
            queue = Queue()
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=object(),
                workflow_path=workflow_template,
                output_dir=root / "output/render",
                project_dir=root,
                asset_uploader=AssetUploader(),
                render_queue=queue,
                postprocessor=Postprocessor(),
                postprocess=False,
            )
            RenderVideoScenesUseCase(backend, store).execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=workflow_template,
                    audio_file=audio,
                    storyboard_dir=root / "storyboard",
                    output_dir=root / "output/render",
                    scene_numbers={14, 15},
                    skip_existing=False,
                ),
            )
            prompt_node = next(
                node for node in queue.workflows[1][1].values()
                if (node.get("_meta") or {}).get("title") == "#PROMPT"
            )
            manifest = json.loads(
                (root / "output/render/scene_0015/manifest.json").read_text(encoding="utf-8"),
            )

        final_prompt = prompt_node["inputs"]["value"]
        self.assertIn("verified predecessor boundary frame 23", final_prompt)
        self.assertIn("Ravena remains ascended and absent", final_prompt)
        self.assertIsNotNone(manifest["canonical_dependencies"])
        self.assertEqual(14, manifest["startframe_source_scene"])
        self.assertEqual("last_frame_from_previous", manifest["startframe_mode"])
        self.assertEqual(
            "ascended_absent",
            manifest["boundary_frame_manifest"]["continuity_state"]
            ["incoming"]["cast_states"]["ravena"],
        )
        self.assertEqual(
            "ascended_absent",
            manifest["narrative_boundary_manifest"]["incoming"]
            ["cast_states"]["ravena"],
        )


if __name__ == "__main__":
    unittest.main()
