from __future__ import annotations

import json
import unittest
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from feverslop.adapters.comfyui_ingredients_video_backend import (
    ComfyUIIngredientsVideoRenderBackend,
)
from feverslop.adapters.prepared_workflow import (
    PreparedWorkflowRenderer,
    WorkflowMaterializationRequest,
    WorkflowMaterializer,
)
from feverslop.application.render_plan_ingredients_sheets import (
    enrich_render_plan_with_ingredients_sheets,
)
from feverslop.domain.effective_render_plan import CanonicalSceneDependencies
from feverslop.domain.prepared_workflow import SceneWorkflowManifest, sha256_file
from feverslop.domain.visual_consistency import (
    ReferenceAnchor,
    SceneConsistencyContract,
)
from feverslop.errors import FeverSlopValidationError
from feverslop.prompting.ingredients_signatures import IngredientsVisionResult
from feverslop.scene_artifacts import SceneArtifactLayout


class FakeUploader:
    def resolve_audio_name(self, path, **_kwargs):
        return "feverslop/audio/song-audiohash.wav"

    def resolve_reference_image_name(self, path, **_kwargs):
        return f"actual/{Path(path).name}"

    def resolve_startframe_name(self, path, **_kwargs):
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
        self.max_render_frames = None
        self.max_render_duration_seconds = None
        self.render_budget_workflow_path = None
        self.round_render_frames_to_8n1 = False

    def _seed_for_scene(self, scene_number):
        self.seed_calls += 1
        return 9000 + scene_number

    def _rolling_spec(self, scene):
        return {"fps": scene["fps"], "render_frame_count": scene["frame_count"]}

    def build_workflow(self, scene, *, prompt, comfy_audio_name, rolling):
        self.build_calls += 1
        ingredients_sheet = scene.get("ingredients_scene_sheet") or (scene.get("ingredients") or {}).get("sheet_path")
        if ingredients_sheet:
            reference_name = self.asset_uploader.resolve_reference_image_name(ingredients_sheet)
        else:
            reference_name = None
        startframe = (scene.get("keyframes") or {}).get("startframe_path")
        if startframe:
            self.asset_uploader.resolve_startframe_name(startframe)
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


class FakeCurrentServerClient:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.checked = []

    def input_file_exists(self, comfyui_name):
        self.checked.append(comfyui_name)
        return comfyui_name in self.existing


class FakeCurrentServerUploader:
    def __init__(self, existing=()):
        self.client = FakeCurrentServerClient(existing)
        self.audio_uploads = []
        self.reference_uploads = []

    def resolve_audio_name(self, path, **_kwargs):
        self.audio_uploads.append(Path(path))
        return "linux/audio/song.wav"

    def resolve_reference_image_name(self, path, **_kwargs):
        self.reference_uploads.append(Path(path))
        return "linux/references/sheet.png"


class FakeCurrentServerResolver:
    def __init__(self):
        self.workflow_paths = []

    def resolve_workflow_models(self, workflow, *, workflow_path):
        self.workflow_paths.append(workflow_path)
        resolved = json.loads(json.dumps(workflow))
        resolved["lora"]["inputs"]["lora_name"] = "LTXV2/model.safetensors"
        return resolved


class WorkflowMaterializerTests(unittest.TestCase):
    @staticmethod
    def _canonical_dependencies(
        *, workflow: str = "a", references: str = "b", revision: str = "c",
    ) -> CanonicalSceneDependencies:
        return CanonicalSceneDependencies(
            schema="feverslop.canonical-dependencies/v1",
            source="output/render/plans/base.json",
            source_revision=revision * 64,
            scene_id="canonical-scene-1",
            workflow_fingerprint=workflow * 64,
            reference_fingerprint=references * 64,
        )

    @staticmethod
    def _ingredients_contract(
        scene: int,
        actor_sha: str,
        location_sha: str,
        *,
        fingerprint: str | None = None,
    ) -> dict:
        contract = SceneConsistencyContract.create(
            scene=scene,
            mode="ingredients",
            workflow_profile="ingredients-v4",
            actors=(
                ReferenceAnchor(
                    id="hero",
                    kind="actor",
                    look_id="default",
                    asset_role="identity-reference",
                    asset_sha256=actor_sha,
                    prompt_anchor="hero",
                ),
            ),
            location=ReferenceAnchor(
                id="archive",
                kind="location",
                look_id="default",
                asset_role="environment-reference",
                asset_sha256=location_sha,
                prompt_anchor="archive",
            ),
            transition_from_previous="cut",
        ).to_dict()
        if fingerprint is not None:
            contract["fingerprint"] = fingerprint
        return contract

    def test_renderer_accepts_current_server_asset_and_model_adapters(self):
        parameters = signature(PreparedWorkflowRenderer).parameters

        self.assertIn("asset_uploader", parameters)
        self.assertIn("model_resolver", parameters)
        self.assertIn("model_workflow_path", parameters)

    def test_renderer_enforces_current_budget_for_legacy_manifest(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            prepared = WorkflowMaterializer(
                FakeBackend(template), SceneArtifactLayout(project),
            ).prepare(
                WorkflowMaterializationRequest(
                    scene={"scene": 1, "fps": 24, "frame_count": 50,
                           "width": 64, "height": 64},
                    prompt="x", audio_file=None, render_plan_path=plan,
                    pipeline="test", seed=1,
                ),
            )
            payload = json.loads(prepared.manifest_path.read_text())
            for key in (
                "max_render_frames", "max_render_duration_seconds",
                "render_budget_workflow_path", "round_render_frames_to_8n1",
            ):
                payload.pop(key, None)
            prepared.manifest_path.write_text(json.dumps(payload))
            queue = FakeQueue()

            with self.assertRaisesRegex(FeverSlopValidationError, "current.json"):
                PreparedWorkflowRenderer(
                    project_dir=project, render_queue=queue,
                    postprocessor=FakePostprocessor(), expected_pipeline="test",
                    max_render_frames=49, max_render_duration_seconds=2,
                    render_budget_workflow_path="current.json",
                ).render(prepared.workflow_path)

            self.assertEqual([], queue.workflows)

    def test_prepare_rejects_render_above_budget_before_workflow_build(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            backend = FakeBackend(template)
            backend.max_render_frames = 49
            backend.max_render_duration_seconds = 2
            backend.render_budget_workflow_path = "limited.json"

            with self.assertRaisesRegex(FeverSlopValidationError, "Scene 1 requires 50 render frames"):
                WorkflowMaterializer(backend, SceneArtifactLayout(project)).prepare(
                    WorkflowMaterializationRequest(
                        scene={"scene": 1, "fps": 24, "frame_count": 50,
                               "width": 64, "height": 64},
                        prompt="x", audio_file=None, render_plan_path=plan,
                        pipeline="test", seed=1,
                    ),
                )

            self.assertEqual(0, backend.build_calls)

    def test_renderer_rejects_prepared_workflow_above_persisted_budget(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            backend = FakeBackend(template)
            backend.max_render_frames = 49
            backend.max_render_duration_seconds = 2
            backend.render_budget_workflow_path = "limited.json"
            prepared = WorkflowMaterializer(backend, SceneArtifactLayout(project)).prepare(
                WorkflowMaterializationRequest(
                    scene={"scene": 1, "fps": 24, "frame_count": 49,
                           "width": 64, "height": 64},
                    prompt="x", audio_file=None, render_plan_path=plan,
                    pipeline="test", seed=1,
                ),
            )
            payload = json.loads(prepared.manifest_path.read_text())
            payload["render_frame_count"] = 50
            prepared.manifest_path.write_text(json.dumps(payload))
            queue = FakeQueue()

            with self.assertRaisesRegex(FeverSlopValidationError, "limited.json"):
                PreparedWorkflowRenderer(
                    project_dir=project, render_queue=queue,
                    postprocessor=FakePostprocessor(), expected_pipeline="test",
                ).render(prepared.workflow_path)

            self.assertEqual([], queue.workflows)

    def test_vision_enriched_ingredients_prompt_reaches_materialized_workflow_unchanged(self):
        class FakeIngredientsModule:
            def __init__(self, _llm, **_kwargs):
                pass

            def vision(self, _payload, _paths):
                return IngredientsVisionResult(
                    references=[
                        {"id": "mara", "type": "actor", "t2i_description": "Mara has a sharp black bob, grey eyes, and a silver coat."},
                        {"id": "archive", "type": "location", "t2i_description": "The archive has amber lamps, dark oak shelves, and a brass desk."},
                    ],
                    shot_invariants=" ".join(["stable full-frame composition and amber lighting"] * 12),
                )

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
                "ltx": {
                    "i2v_prompt_from_t2i": "Mara approaches the ledger.",
                    "msr_prompt_relay": [{
                        "frame_start": 0, "frame_end": 48, "state": "instrumental",
                        "prompt": "Mara approaches the ledger with her mouth closed.",
                    }],
                },
            }]))
            with patch("feverslop.application.ingredients_vision_prompt.IngredientsPromptModules", FakeIngredientsModule):
                enriched_path = enrich_render_plan_with_ingredients_sheets(
                    plan, refs, project / "ingredients.json", llm=object(),
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
                ),
            )
            workflow = json.loads(prepared.workflow_path.read_text())
            positive = workflow["2"]["inputs"]["text"]
            self.assertEqual(1, positive.count("### Reference Sheet Description"))
            self.assertEqual(1, positive.count("### Target Description"))
            for expected in ("`mara`", "`archive`", "stable full-frame composition", "do not reproduce their framing, composition, borders, panels, or layout", "mouths remain closed"):
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

    def test_prepare_records_nested_v4_ingredients_sheet_in_manifest(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "workflows" / "ingredients-v4.json"
            plan = project / "output" / "render" / "plans" / "ingredients.json"
            sheet = project / "output" / "references" / "ingredients_sheets" / "scene_0001.png"
            for path, data in ((template, b"{}"), (plan, b"{}"), (sheet, b"sheet")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            prepared = WorkflowMaterializer(FakeBackend(template), SceneArtifactLayout(project)).prepare(
                WorkflowMaterializationRequest(
                    scene={
                        "scene": 1, "fps": 24, "frame_count": 49, "width": 1536, "height": 896,
                        "ingredients": {"sheet_path": sheet.relative_to(project).as_posix()},
                    },
                    prompt="move", audio_file=None, render_plan_path=plan,
                    pipeline="ltx_ingredients", seed=1,
                ),
            )

            manifest = SceneWorkflowManifest.read(prepared.manifest_path)
            self.assertEqual({"ingredients_sheet"}, {asset.role for asset in manifest.assets})

    def test_prepare_records_exact_contract_and_source_provenance(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            plan = project / "plan.json"
            sheet = project / "sheet.png"
            actor = project / "actor.png"
            location = project / "location.png"
            for path, content in (
                (template, b"{}"),
                (plan, b"{}"),
                (sheet, b"sheet"),
                (actor, b"actor"),
                (location, b"location"),
            ):
                path.write_bytes(content)
            contract_payload = self._ingredients_contract(
                3, sha256_file(actor), sha256_file(location),
            )
            scene = {
                "scene": 3,
                "fps": 24,
                "frame_count": 49,
                "width": 1280,
                "height": 704,
                "ingredients": {
                    "sheet_path": sheet.relative_to(project).as_posix(),
                    "sheet_sha256": sha256_file(sheet),
                },
                "visual_consistency": contract_payload,
                "visual_consistency_sources": {
                    "actors": [
                        {
                            "id": "hero",
                            "path": actor.relative_to(project).as_posix(),
                        },
                    ],
                    "location": {
                        "id": "archive",
                        "path": location.relative_to(project).as_posix(),
                    },
                },
            }

            prepared = WorkflowMaterializer(
                FakeBackend(template), SceneArtifactLayout(project),
            ).prepare(
                WorkflowMaterializationRequest(
                    scene=scene,
                    prompt="x",
                    audio_file=None,
                    render_plan_path=plan,
                    pipeline="ltx_ingredients",
                    seed=100003,
                ),
            )

            manifest = SceneWorkflowManifest.read(prepared.manifest_path)
            self.assertEqual(
                SceneConsistencyContract.from_dict(contract_payload),
                manifest.consistency,
            )
            self.assertEqual(100003, manifest.seed)
            self.assertEqual(
                {"ingredients_sheet", "actor_sheet", "location_sheet"},
                {asset.role for asset in manifest.assets},
            )
            self.assertTrue(
                all(not Path(asset.path).is_absolute() for asset in manifest.assets),
            )
            self.assertEqual([], manifest.verify(project))

    def test_prepare_rejects_tampered_consistency_fingerprint_before_build(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            plan = project / "plan.json"
            actor = project / "actor.png"
            location = project / "location.png"
            for path in (template, plan, actor, location):
                path.write_bytes(path.name.encode())
            backend = FakeBackend(template)

            with self.assertRaisesRegex(
                ValueError, "fingerprint does not match canonical payload",
            ):
                WorkflowMaterializer(backend, SceneArtifactLayout(project)).prepare(
                    WorkflowMaterializationRequest(
                        scene={
                            "scene": 1,
                            "fps": 24,
                            "frame_count": 49,
                            "width": 1280,
                            "height": 704,
                            "visual_consistency": self._ingredients_contract(
                                1,
                                sha256_file(actor),
                                sha256_file(location),
                                fingerprint="0" * 64,
                            ),
                        },
                        prompt="x",
                        audio_file=None,
                        render_plan_path=plan,
                        pipeline="ltx_ingredients",
                        seed=100001,
                    ),
                )

            self.assertEqual(0, backend.build_calls)

    def test_prepare_rejects_invalid_consistency_provenance_before_manifest_replace(self):
        failures = (
            "source_hash",
            "mode",
            "missing_startframe",
            "startframe_hash",
            "missing_source_clip_hash",
        )
        for failure in failures:
            with self.subTest(failure=failure), TemporaryDirectory() as tmp:
                project = Path(tmp)
                template = project / "template.json"
                plan = project / "plan.json"
                sheet = project / "sheet.png"
                actor = project / "actor.png"
                wrong_actor = project / "wrong-actor.png"
                location = project / "location.png"
                for path, content in (
                    (template, b"{}"),
                    (plan, b"{}"),
                    (sheet, b"sheet"),
                    (actor, b"actor"),
                    (wrong_actor, b"wrong"),
                    (location, b"location"),
                ):
                    path.write_bytes(content)
                if failure in {
                    "missing_startframe",
                    "startframe_hash",
                    "missing_source_clip_hash",
                }:
                    contract = SceneConsistencyContract.create(
                        scene=2,
                        mode="msr",
                        workflow_profile="msr-startframe",
                        actors=(
                            ReferenceAnchor(
                                id="hero",
                                kind="actor",
                                look_id="default",
                                asset_role="identity-reference",
                                asset_sha256=sha256_file(actor),
                                prompt_anchor="hero",
                            ),
                        ),
                        location=ReferenceAnchor(
                            id="archive",
                            kind="location",
                            look_id="default",
                            asset_role="environment-reference",
                            asset_sha256=sha256_file(location),
                            prompt_anchor="archive",
                        ),
                        transition_from_previous="continuous",
                    )
                    pipeline = "ltx_msr"
                    scene_number = 2
                else:
                    contract = SceneConsistencyContract.from_dict(
                        self._ingredients_contract(
                            1, sha256_file(actor), sha256_file(location),
                        ),
                    )
                    pipeline = "ltx_msr" if failure == "mode" else "ltx_ingredients"
                    scene_number = 1
                actor_source = wrong_actor if failure == "source_hash" else actor
                scene = {
                    "scene": scene_number,
                    "fps": 24,
                    "frame_count": 49,
                    "width": 1280,
                    "height": 704,
                    "ingredients": {
                        "sheet_path": sheet.relative_to(project).as_posix(),
                        "sheet_sha256": sha256_file(sheet),
                    },
                    "visual_consistency": contract.to_dict(),
                    "visual_consistency_sources": {
                        "actors": [{
                            "id": "hero",
                            "path": actor_source.relative_to(project).as_posix(),
                        }],
                        "location": {
                            "id": "archive",
                            "path": location.relative_to(project).as_posix(),
                        },
                    },
                }
                if failure in {"startframe_hash", "missing_source_clip_hash"}:
                    source_clip = layout_source = (
                        project
                        / "output"
                        / "render"
                        / "scenes"
                        / "scene_0001"
                        / "final.mp4"
                    )
                    source_clip.parent.mkdir(parents=True, exist_ok=True)
                    source_clip.write_bytes(b"source clip")
                    startframe = project / "startframe.png"
                    startframe.write_bytes(b"extracted frame")
                    claimed_startframe_sha = sha256_file(startframe)
                    if failure == "startframe_hash":
                        startframe.write_bytes(b"tampered frame")
                    scene["keyframes"] = {
                        "startframe_path": startframe.relative_to(project).as_posix(),
                        "startframe_sha256": claimed_startframe_sha,
                        "startframe_mode": "last_frame_from_previous",
                        "startframe_source_scene": 1,
                        "startframe_source_clip_path": (
                            layout_source.relative_to(project).as_posix()
                        ),
                        "startframe_source_clip_sha256": sha256_file(source_clip),
                        "startframe_extractor": "last-frame-v1",
                    }
                    if failure == "missing_source_clip_hash":
                        scene["keyframes"].pop("startframe_source_clip_sha256")
                layout = SceneArtifactLayout(project)

                with self.assertRaisesRegex(ValueError, "consistency provenance"):
                    WorkflowMaterializer(FakeBackend(template), layout).prepare(
                        WorkflowMaterializationRequest(
                            scene=scene,
                            prompt="x",
                            audio_file=None,
                            render_plan_path=plan,
                            pipeline=pipeline,
                            seed=100000 + scene_number,
                        ),
                    )

                self.assertFalse(layout.scene_manifest(scene_number).exists())

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
                ),
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
            actor = project / "output" / "references" / "actors" / "singer" / "msr_sheet.png"
            location = project / "output" / "references" / "locations" / "stage" / "sheet.png"
            for path, data in (
                (template, b"{}"), (plan, b"{}"), (audio, b"audio"), (sheet, b"sheet"),
                (actor, b"actor"), (location, b"location"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            backend = FakeBackend(template)
            request = WorkflowMaterializationRequest(
                scene={"scene": 5, "fps": 24, "frame_count": 241, "width": 1536,
                       "height": 896, "ingredients_scene_sheet": str(sheet),
                       "references": {
                           "actor_msr_paths": [str(actor)],
                           "location_sheet_path": str(location),
                       }},
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

    def test_prepare_persists_canonical_scene_dependencies(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            plan = project / "plan.json"
            template.write_text("{}", encoding="utf-8")
            plan.write_text("[]", encoding="utf-8")
            dependencies = self._canonical_dependencies()

            prepared = WorkflowMaterializer(
                FakeBackend(template), SceneArtifactLayout(project),
            ).prepare(WorkflowMaterializationRequest(
                scene={"scene": 1, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                prompt="x", audio_file=None, render_plan_path=plan,
                pipeline="test", seed=1, canonical_dependencies=dependencies,
            ))

            self.assertEqual(
                dependencies,
                SceneWorkflowManifest.read(prepared.manifest_path).canonical_dependencies,
            )

    def test_renderer_rejects_stale_canonical_workflow_before_queueing(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            plan = project / "plan.json"
            template.write_text("{}", encoding="utf-8")
            plan.write_text("[]", encoding="utf-8")
            prepared = WorkflowMaterializer(
                FakeBackend(template), SceneArtifactLayout(project),
            ).prepare(WorkflowMaterializationRequest(
                scene={"scene": 1, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                prompt="x", audio_file=None, render_plan_path=plan,
                pipeline="test", seed=1,
                canonical_dependencies=self._canonical_dependencies(),
            ))
            queue = FakeQueue()

            with self.assertRaisesRegex(
                ValueError,
                r"output/render/plans/base.json.*scene 1.*workflow fingerprint changed.*--stage ltx_prepare_workflows",
            ):
                PreparedWorkflowRenderer(
                    project_dir=project, render_queue=queue,
                    postprocessor=FakePostprocessor(), expected_pipeline="test",
                ).render(
                    prepared.workflow_path,
                    canonical_dependencies=self._canonical_dependencies(workflow="d"),
                )

            self.assertEqual([], queue.workflows)

    def test_renderer_reuses_unchanged_scene_after_unrelated_plan_revision(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            plan = project / "plan.json"
            template.write_text("{}", encoding="utf-8")
            plan.write_text("[]", encoding="utf-8")
            prepared = WorkflowMaterializer(
                FakeBackend(template), SceneArtifactLayout(project),
            ).prepare(WorkflowMaterializationRequest(
                scene={"scene": 1, "fps": 24, "frame_count": 9, "width": 64, "height": 64},
                prompt="x", audio_file=None, render_plan_path=plan,
                pipeline="test", seed=1,
                canonical_dependencies=self._canonical_dependencies(),
            ))
            plan.write_text('[{"scene": 2, "changed": true}]', encoding="utf-8")
            queue = FakeQueue()

            PreparedWorkflowRenderer(
                project_dir=project, render_queue=queue,
                postprocessor=FakePostprocessor(), expected_pipeline="test",
            ).render(
                prepared.workflow_path,
                canonical_dependencies=self._canonical_dependencies(revision="f"),
            )

            self.assertEqual(1, len(queue.workflows))

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

    def test_renderer_preflights_assets_and_resolves_models_for_current_server(self):
        class PortableBackend(FakeBackend):
            def build_workflow(self, scene, *, prompt, comfy_audio_name, rolling):
                sheet_name = self.asset_uploader.resolve_reference_image_name(
                    scene["ingredients"]["sheet_path"],
                )
                return {
                    "image": {"inputs": {"image": sheet_name}},
                    "audio": {"inputs": {"audio": comfy_audio_name}},
                    "lora": {"inputs": {"lora_name": r"LTXV2\model.safetensors"}},
                }

        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            layout = SceneArtifactLayout(project)
            template = project / "template.json"
            plan = layout.ingredients_plan
            audio = project / "song.wav"
            sheet = project / "sheet.png"
            for path, content in (
                (template, b"{}"),
                (plan, b"{}"),
                (audio, b"audio"),
                (sheet, b"sheet"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            prepared = WorkflowMaterializer(
                PortableBackend(template), layout,
            ).prepare(
                WorkflowMaterializationRequest(
                    scene={
                        "scene": 2,
                        "fps": 24,
                        "frame_count": 9,
                        "width": 64,
                        "height": 64,
                        "ingredients": {"sheet_path": str(sheet)},
                    },
                    prompt="x",
                    audio_file=audio,
                    render_plan_path=plan,
                    pipeline="ltx_ingredients",
                    seed=2,
                ),
            )
            stored = json.loads(prepared.workflow_path.read_text())
            uploader = FakeCurrentServerUploader(
                existing={"feverslop/audio/song-audiohash.wav"},
            )
            resolver = FakeCurrentServerResolver()
            queue = FakeQueue()

            PreparedWorkflowRenderer(
                project_dir=project,
                render_queue=queue,
                postprocessor=FakePostprocessor(),
                expected_pipeline="ltx_ingredients",
                asset_uploader=uploader,
                model_resolver=resolver,
                model_workflow_path="current-linux.json",
            ).render(prepared.workflow_path)

            self.assertEqual(
                {"feverslop/audio/song-audiohash.wav", "actual/sheet.png"},
                set(uploader.client.checked),
            )
            self.assertEqual([], uploader.audio_uploads)
            self.assertEqual([sheet], uploader.reference_uploads)
            self.assertEqual("linux/references/sheet.png", queue.workflows[0]["image"]["inputs"]["image"])
            self.assertEqual("LTXV2/model.safetensors", queue.workflows[0]["lora"]["inputs"]["lora_name"])
            self.assertEqual(["current-linux.json"], resolver.workflow_paths)
            self.assertEqual(stored, json.loads(prepared.workflow_path.read_text()))

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
                ),
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
                ),
            )
            queue = FakeQueue()

            with self.assertRaisesRegex(ValueError, "pipeline"):
                PreparedWorkflowRenderer(
                    project_dir=project, render_queue=queue,
                    postprocessor=FakePostprocessor(), expected_pipeline="other",
                ).render(prepared.workflow_path)
            self.assertEqual([], queue.workflows)

    def test_renderer_rejects_active_consistency_profile_mismatch_before_upload(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            template = project / "template.json"
            template.write_text("{}")
            plan = project / "plan.json"
            plan.write_text("{}")
            prepared = WorkflowMaterializer(
                FakeBackend(template),
                SceneArtifactLayout(project),
            ).prepare(
                WorkflowMaterializationRequest(
                    scene={
                        "scene": 2,
                        "fps": 24,
                        "frame_count": 9,
                        "width": 64,
                        "height": 64,
                    },
                    prompt="x",
                    audio_file=None,
                    render_plan_path=plan,
                    pipeline="ltx_msr",
                    seed=2,
                ),
            )
            manifest_payload = json.loads(
                prepared.manifest_path.read_text(encoding="utf-8"),
            )
            manifest_payload["consistency"] = SceneConsistencyContract.create(
                scene=2,
                mode="msr",
                workflow_profile="prepared-profile",
                actors=(),
                location=None,
                transition_from_previous="cut",
            ).to_dict()
            prepared.manifest_path.write_text(
                json.dumps(manifest_payload),
                encoding="utf-8",
            )
            queue = FakeQueue()
            uploader = FakeCurrentServerUploader(existing=set())

            with self.assertRaisesRegex(ValueError, "workflow profile"):
                PreparedWorkflowRenderer(
                    project_dir=project,
                    render_queue=queue,
                    postprocessor=FakePostprocessor(),
                    expected_pipeline="ltx_msr",
                    expected_workflow_profile="active-profile",
                    asset_uploader=uploader,
                ).render(prepared.workflow_path)

            self.assertEqual([], queue.workflows)
            self.assertEqual([], uploader.client.checked)

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
                ),
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
