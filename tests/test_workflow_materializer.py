from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from feverslop.adapters.comfyui_ingredients_video_backend import ComfyUIIngredientsVideoRenderBackend
from feverslop.adapters.prepared_workflow import (
    PreparedWorkflowRenderer,
    WorkflowMaterializationRequest,
    WorkflowMaterializer,
)
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.application.render_plan_ingredients_sheets import enrich_render_plan_with_ingredients_sheets
from feverslop.scene_artifacts import SceneArtifactLayout


class FakeUploader:
    def resolve_audio_name(self, path, **_kwargs):
        return "feverslop/audio/song-audiohash.wav"

    def resolve_reference_image_name(self, path, **_kwargs):
        return f"actual/{Path(path).name}"


class FakeResolver:
    def __init__(self):
        self.calls = 0

    def resolve_workflow_models(self, workflow, *, workflow_path):
        self.calls += 1
        return {**workflow, "model": str(workflow_path)}


class FakeBackend:
    def __init__(self, template: Path):
        self.workflow_path = template
        self.workflow_label = template
        self.asset_uploader = FakeUploader()
        self.model_resolver = FakeResolver()
        self.seed_offset = 100000
        self.randomize_seed = True
        self.build_calls = 0
        self.seed_calls = 0

    def _seed_for_scene(self, scene_number):
        self.seed_calls += 1
        return 9000 + scene_number

    def _rolling_spec(self, scene):
        return {"fps": scene["fps"], "render_frame_count": scene["frame_count"]}

    def build_workflow(self, scene, *, prompt, comfy_audio_name, rolling):
        self.build_calls += 1
        if scene.get("ingredients_scene_sheet"):
            reference_name = self.asset_uploader.resolve_reference_image_name(scene["ingredients_scene_sheet"])
        else:
            reference_name = None
        seed = self.seed_offset + scene["scene"]
        return {"scene": scene["scene"], "prompt": prompt, "audio": comfy_audio_name,
                "reference": reference_name, "frames": rolling["render_frame_count"], "seed": seed}


class FakeQueue:
    def __init__(self):
        self.workflows = []

    def queue_workflow_and_download_first_video(self, workflow, *, scene_number, output_path):
        self.workflows.append(workflow)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")
        return output_path


class FakePostprocessor:
    def __init__(self):
        self.specs = []

    def trim_clip(self, spec):
        self.specs.append(spec)
        spec.output_file.write_bytes(b"trimmed")
        return spec.output_file


class WorkflowMaterializerTests(unittest.TestCase):
    def test_vision_enriched_ingredients_prompt_reaches_materialized_workflow_unchanged(self):
        class VisionLLM:
            def complete_prompt_with_images(self, _system, _prompt, _paths):
                target = (
                    "The full-frame continuous shot opens on Mara beside the archive desk. "
                    "Mara turns toward the lens as the camera slowly pushes forward; her silver coat and black hair react to the moving air, while amber light travels across the shelves. "
                    + "She crosses the room with deliberate steps while dust drifts and the camera arcs around her, preserving a single uninterrupted composition. " * 14
                    + "The shot ends with Mara stopping at the desk, her hand resting on the ledger as the camera settles and the warm light fades."
                )
                return json.dumps({
                    "references": [
                        {"id": "mara", "type": "actor", "description": "Mara has a sharp black bob, grey eyes, and a silver coat."},
                        {"id": "archive", "type": "location", "description": "The archive has amber lamps, dark oak shelves, and a brass desk."},
                    ],
                    "target_description": target,
                })

        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            refs = project / "output" / "references"
            for kind, reference_id in (("actors", "mara"), ("locations", "archive")):
                root = refs / kind / reference_id
                root.mkdir(parents=True)
                image = root / "sheet.png"
                Image.new("RGB", (32, 32), "white").save(image)
                (root / "manifest.json").write_text(json.dumps({
                    "id": reference_id, "name": reference_id.title(), "sheet_path": image.relative_to(project).as_posix(),
                }))
            plan = project / "plan.json"
            plan.write_text(json.dumps([{
                "scene": 1, "fps": 24, "frame_count": 49, "width": 1280, "height": 704,
                "references": {"actor_ids": ["mara"], "location_id": "archive"},
                "ltx": {"i2v_prompt_from_t2i": "Mara approaches the ledger."},
            }]))
            enriched_path = enrich_render_plan_with_ingredients_sheets(
                plan, refs, project / "ingredients.json", llm=VisionLLM(),
            )
            scene = json.loads(enriched_path.read_text())[0]
            template = project / "workflow.json"
            original_negative = "bad anatomy, text, panels"
            template.write_text(json.dumps({
                "1": {"inputs": {"image": ""}, "_meta": {"title": "#INGREDIENTS"}},
                "2": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT_POSITIVE"}},
                "3": {"inputs": {"text": original_negative}, "_meta": {"title": "#PROMPT_NEGATIVE"}},
                "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
            }))
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=object(), workflow_path=template, output_dir=project / "out", project_dir=project,
                asset_uploader=FakeUploader(), model_resolver=FakeResolver(), postprocess=False,
            )
            prepared = WorkflowMaterializer(backend, SceneArtifactLayout(project)).prepare(
                WorkflowMaterializationRequest(
                    scene=scene, prompt="fallback", audio_file=None, render_plan_path=enriched_path,
                    pipeline="ltx_ingredients", seed=1,
                )
            )
            workflow = json.loads(prepared.workflow_path.read_text())
            positive = workflow["2"]["inputs"]["text"]
            self.assertEqual(1, positive.count("### Reference Sheet Description"))
            self.assertEqual(1, positive.count("### Target Description"))
            for expected in ("`mara`", "`archive`", "full-frame continuous shot", "camera slowly pushes forward", "do not reproduce their framing, composition, borders, panels, or layout"):
                self.assertIn(expected, positive)
            self.assertEqual(original_negative, workflow["3"]["inputs"]["text"])

    def test_manifest_reader_requires_assets_and_pipeline_roles(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            base = {
                "schema": "feverslop.scene-workflow/v1", "scene": 1,
                "pipeline": "ltx_ingredients", "workflow": {}, "template": {},
                "render_plan": {}, "seed": 1, "fps": 24, "frame_count": 9,
                "width": 64, "height": 64,
            }
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "requires an assets list"):
                SceneWorkflowManifest.read(path)
            base["assets"] = []
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "ingredients_sheet"):
                SceneWorkflowManifest.read(path)

    def test_prepare_selects_random_seed_once_when_request_does_not_supply_one(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            backend = FakeBackend(template)

            prepared = WorkflowMaterializer(backend, SceneArtifactLayout(project)).prepare(
                WorkflowMaterializationRequest(
                    scene={"scene": 3, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                    prompt="x", audio_file=None, render_plan_path=plan,
                    pipeline="test",
                )
            )

            self.assertEqual(1, backend.seed_calls)
            self.assertEqual(9003, SceneWorkflowManifest.read(prepared.manifest_path).seed)

    def test_prepare_writes_resolved_workflow_and_manifest_without_queueing(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "workflows" / "ingredients.json"
            plan = project / "output" / "render" / "plans" / "ingredients.json"
            audio = project / "song.wav"
            sheet = project / "output" / "references" / "ingredients_sheets" / "scene_0005.png"
            for path, data in ((template, b"{}"), (plan, b"{}"), (audio, b"audio"), (sheet, b"sheet")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            backend = FakeBackend(template)
            request = WorkflowMaterializationRequest(
                scene={"scene": 5, "fps": 24, "frame_count": 241, "width": 1536,
                       "height": 896, "ingredients_scene_sheet": str(sheet)},
                prompt="move", audio_file=audio, render_plan_path=plan,
                pipeline="ltx_ingredients", seed=7005,
            )

            prepared = WorkflowMaterializer(backend, SceneArtifactLayout(project)).prepare(request)

            workflow = json.loads(prepared.workflow_path.read_text())
            self.assertEqual(7005, workflow["seed"])
            self.assertEqual(1, backend.model_resolver.calls)
            manifest = SceneWorkflowManifest.read(prepared.manifest_path)
            self.assertEqual(7005, manifest.seed)
            self.assertEqual(241, manifest.frame_count)
            self.assertEqual(241, manifest.render_frame_count)
            self.assertEqual(0, manifest.trim_front_frames)
            self.assertEqual({"audio", "ingredients_sheet"}, {item.role for item in manifest.assets})
            self.assertIn("actual/scene_0005.png", {item.comfyui_name for item in manifest.assets})
            self.assertEqual([], manifest.verify(project))

    def test_renderer_reports_every_hash_mismatch_before_queue(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            layout = SceneArtifactLayout(project)
            template = project / "template.json"
            template.write_text("{}")
            plan = layout.ingredients_plan
            plan.parent.mkdir(parents=True)
            plan.write_text("{}")
            audio = project / "song.wav"
            audio.write_bytes(b"audio")
            backend = FakeBackend(template)
            prepared = WorkflowMaterializer(backend, layout).prepare(WorkflowMaterializationRequest(
                scene={"scene": 1, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                prompt="x", audio_file=audio, render_plan_path=plan, pipeline="test", seed=1))
            prepared.workflow_path.write_text('{"changed": true}')
            plan.write_text('{"changed": true}')
            queue = FakeQueue()

            with self.assertRaisesRegex(ValueError, "workflow.*render_plan"):
                PreparedWorkflowRenderer(
                    project_dir=project, render_queue=queue, postprocessor=FakePostprocessor(),
                    expected_pipeline="test",
                ).render(prepared.workflow_path)
            self.assertEqual([], queue.workflows)

    def test_renderer_queues_exact_stored_json_and_writes_scene_outputs(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            layout = SceneArtifactLayout(project)
            template = project / "template.json"
            template.write_text("{}")
            plan = layout.ingredients_plan
            plan.parent.mkdir(parents=True)
            plan.write_text("{}")
            backend = FakeBackend(template)
            prepared = WorkflowMaterializer(backend, layout).prepare(WorkflowMaterializationRequest(
                scene={"scene": 2, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                prompt="x", audio_file=None, render_plan_path=plan, pipeline="test", seed=2))
            expected = json.loads(prepared.workflow_path.read_text())
            queue = FakeQueue()

            postprocessor = FakePostprocessor()
            result = PreparedWorkflowRenderer(
                project_dir=project, render_queue=queue, postprocessor=postprocessor,
                expected_pipeline="test",
            ).render(prepared.workflow_path)

            self.assertEqual(expected, queue.workflows[0])
            self.assertEqual(layout.scene_final_video(2), result)
            self.assertEqual(b"video", layout.scene_raw_video(2).read_bytes())
            self.assertEqual(b"trimmed", result.read_bytes())
            self.assertEqual(9, postprocessor.specs[0].keep_frames)
            self.assertEqual(0, postprocessor.specs[0].trim_front_frames)

    def test_manifest_persists_render_count_and_trim_metadata(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            backend = FakeBackend(template)
            backend._rolling_spec = lambda _scene: {
                "fps": 24, "render_frame_count": 65, "trim_front_frames": 8,
            }
            prepared = WorkflowMaterializer(backend, SceneArtifactLayout(project)).prepare(
                WorkflowMaterializationRequest(
                    scene={"scene": 2, "fps": 24, "frame_count": 49, "width": 64, "height": 64},
                    prompt="x", audio_file=None, render_plan_path=plan, pipeline="test", seed=2,
                )
            )

            manifest = SceneWorkflowManifest.read(prepared.manifest_path)
            self.assertEqual(49, manifest.frame_count)
            self.assertEqual(65, manifest.render_frame_count)
            self.assertEqual(8, manifest.trim_front_frames)

    def test_renderer_rejects_unexpected_pipeline_before_queue(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            prepared = WorkflowMaterializer(FakeBackend(template), SceneArtifactLayout(project)).prepare(
                WorkflowMaterializationRequest(
                    scene={"scene": 2, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                    prompt="x", audio_file=None, render_plan_path=plan, pipeline="test", seed=2,
                )
            )
            queue = FakeQueue()

            with self.assertRaisesRegex(ValueError, "pipeline"):
                PreparedWorkflowRenderer(
                    project_dir=project, render_queue=queue,
                    postprocessor=FakePostprocessor(), expected_pipeline="other",
                ).render(prepared.workflow_path)
            self.assertEqual([], queue.workflows)

    def test_renderer_rejects_workflow_path_other_than_manifest_workflow(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            layout = SceneArtifactLayout(project)
            template = project / "template.json"
            template.write_text("{}")
            plan = layout.ingredients_plan
            plan.parent.mkdir(parents=True)
            plan.write_text("{}")
            prepared = WorkflowMaterializer(FakeBackend(template), layout).prepare(
                WorkflowMaterializationRequest(
                    scene={"scene": 2, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                    prompt="x", audio_file=None, render_plan_path=plan,
                    pipeline="test", seed=2,
                )
            )
            impostor = prepared.scene_dir / "impostor.json"
            impostor.write_text(prepared.workflow_path.read_text())

            with self.assertRaisesRegex(ValueError, "does not match manifest workflow"):
                PreparedWorkflowRenderer(
                    project_dir=project, render_queue=FakeQueue(), postprocessor=FakePostprocessor(),
                    expected_pipeline="test",
                ).render(impostor)

    def test_failed_reprepare_preserves_previous_manifest(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            layout = SceneArtifactLayout(project)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            backend = FakeBackend(template)
            request = WorkflowMaterializationRequest(
                scene={"scene": 4, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                prompt="x", audio_file=None, render_plan_path=plan,
                pipeline="ltx_ingredients", seed=4,
            )
            WorkflowMaterializer(backend, layout).prepare(request)
            backend.model_resolver.resolve_workflow_models = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad"))

            with self.assertRaises(RuntimeError):
                WorkflowMaterializer(backend, layout).prepare(request)
            self.assertTrue(layout.scene_manifest(4).exists())


if __name__ == "__main__":
    unittest.main()
