import json
import tempfile
import unittest
from pathlib import Path

from feverslop.domain.prepared_workflow import (
    PreparedSceneWorkflow,
    SceneWorkflowManifest,
    StoredArtifact,
    sha256_file,
)
from feverslop.domain.visual_consistency import ReferenceAnchor, SceneConsistencyContract

SCHEMA_V1 = "feverslop.scene-workflow/v1"
SCHEMA_V2 = "feverslop.scene-workflow/v2"


class PreparedWorkflowManifestTests(unittest.TestCase):
    @staticmethod
    def _contract(
        scene: int,
        actor_sha: str,
        location_sha: str,
        *,
        transition: str = "cut",
    ) -> SceneConsistencyContract:
        return SceneConsistencyContract.create(
            scene=scene,
            mode="msr",
            workflow_profile="msr-startframe",
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
            transition_from_previous=transition,
        )

    def test_round_trip_uses_project_relative_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "output" / "render" / "scenes" / "scene_0005" / "workflow.json"
            template = project / "workflows" / "ltx_ingredients.json"
            plan = project / "output" / "render" / "plans" / "ingredients.json"
            asset = project / "output" / "references" / "ingredients_sheets" / "scene_0005_ingredients.png"
            for path, content in ((workflow, b"{}"), (template, b"template"), (plan, b"[]"), (asset, b"png")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=5,
                pipeline="ltx_ingredients",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[("ingredients_sheet", asset, "feverslop/references/sheet.png")],
                seed=100005,
                fps=24,
                frame_count=241,
                width=1536,
                height=896,
            )
            path = workflow.with_name("manifest.json")
            manifest.write(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(SCHEMA_V2, payload["schema"])
            self.assertEqual("output/render/scenes/scene_0005/workflow.json", payload["workflow"]["path"])
            self.assertEqual(sha256_file(workflow), payload["workflow"]["sha256"])
            self.assertNotIn("external", payload["template"])
            self.assertEqual(manifest, SceneWorkflowManifest.read(path))

    def test_v2_round_trip_records_exact_consistency_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            actor = project / "actor.png"
            location = project / "location.png"
            for path, content in (
                (workflow, b"{}"),
                (template, b"template"),
                (plan, b"[]"),
                (actor, b"actor"),
                (location, b"location"),
            ):
                path.write_bytes(content)
            contract = self._contract(
                1, sha256_file(actor), sha256_file(location)
            )

            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=1,
                pipeline="ltx_msr",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[
                    ("actor_sheet", actor, "actor.png", "hero"),
                    ("location_sheet", location, "location.png", "archive"),
                ],
                seed=100001,
                fps=24,
                frame_count=49,
                width=1280,
                height=704,
                consistency=contract,
            )
            path = project / "manifest.json"
            manifest.write(path)

            restored = SceneWorkflowManifest.read(path)
            self.assertEqual(SCHEMA_V2, restored.schema)
            self.assertEqual(contract, restored.consistency)
            self.assertEqual(contract.to_dict(), json.loads(path.read_text())["consistency"])
            self.assertEqual([], restored.verify(project))

    def test_v1_manifest_reads_with_unknown_consistency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            actor = project / "actor.png"
            location = project / "location.png"
            for path in (workflow, template, plan, actor, location):
                path.write_bytes(path.name.encode())
            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=1,
                pipeline="ltx_msr",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[
                    ("actor_sheet", actor, "actor.png"),
                    ("location_sheet", location, "location.png"),
                ],
                seed=1,
                fps=24,
                frame_count=49,
                width=1280,
                height=704,
            )
            payload = manifest.to_dict()
            payload["schema"] = SCHEMA_V1
            payload.pop("consistency", None)
            path = project / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            restored = SceneWorkflowManifest.read(path)

            self.assertEqual(SCHEMA_V1, restored.schema)
            self.assertIsNone(restored.consistency)

    def test_verify_rejects_contract_asset_role_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            actor = project / "actor.png"
            location = project / "location.png"
            for path, content in (
                (workflow, b"{}"),
                (template, b"template"),
                (plan, b"[]"),
                (actor, b"actor"),
                (location, b"location"),
            ):
                path.write_bytes(content)
            contract = self._contract(
                1, sha256_file(actor), sha256_file(location)
            )
            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=1,
                pipeline="ltx_msr",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[
                    ("location_sheet", actor, "actor.png"),
                    ("actor_sheet", location, "location.png"),
                ],
                seed=1,
                fps=24,
                frame_count=49,
                width=1280,
                height=704,
                consistency=contract,
            )

            mismatches = manifest.verify(project)

            self.assertTrue(any("actor_sheet" in item for item in mismatches))
            self.assertTrue(any("location_sheet" in item for item in mismatches))

    def test_verify_requires_startframe_for_continuous_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            actor = project / "actor.png"
            location = project / "location.png"
            for path in (workflow, template, plan, actor, location):
                path.write_bytes(path.name.encode())
            contract = self._contract(
                2,
                sha256_file(actor),
                sha256_file(location),
                transition="continuous",
            )
            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=2,
                pipeline="ltx_msr",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[
                    ("actor_sheet", actor, "actor.png"),
                    ("location_sheet", location, "location.png"),
                ],
                seed=100002,
                fps=24,
                frame_count=49,
                width=1280,
                height=704,
                consistency=contract,
            )

            self.assertTrue(
                any("startframe" in item for item in manifest.verify(project))
            )

    def test_continuous_manifest_records_and_verifies_exact_startframe_lineage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            actor = project / "actor.png"
            location = project / "location.png"
            startframe = project / "startframe.png"
            source_clip = (
                project
                / "output"
                / "render"
                / "scenes"
                / "scene_0001"
                / "final.mp4"
            )
            for path in (
                workflow,
                template,
                plan,
                actor,
                location,
                startframe,
                source_clip,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode())
            contract = self._contract(
                2,
                sha256_file(actor),
                sha256_file(location),
                transition="continuous",
            )
            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=2,
                pipeline="ltx_msr",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[
                    ("actor_sheet", actor, "actor.png", "hero"),
                    ("location_sheet", location, "location.png", "archive"),
                    ("startframe", startframe, "startframe.png"),
                ],
                seed=100002,
                fps=24,
                frame_count=49,
                width=1280,
                height=704,
                consistency=contract,
                startframe_mode="last_frame_from_previous",
                startframe_source_scene=1,
                startframe_source_clip_path=source_clip,
                startframe_extractor="last-frame-v1",
                startframe_sha256=sha256_file(startframe),
            )
            path = project / "manifest.json"
            manifest.write(path)

            restored = SceneWorkflowManifest.read(path)
            self.assertEqual("last_frame_from_previous", restored.startframe_mode)
            self.assertEqual(1, restored.startframe_source_scene)
            self.assertEqual("last-frame-v1", restored.startframe_extractor)
            self.assertEqual(
                "output/render/scenes/scene_0001/final.mp4",
                restored.startframe_source_clip.path,
            )
            self.assertEqual([], restored.verify(project))

            for key, value in (
                ("startframe_mode", "storyboard"),
                ("startframe_source_scene", 7),
            ):
                payload = restored.to_dict()
                payload[key] = value
                path.write_text(json.dumps(payload), encoding="utf-8")
                tampered = SceneWorkflowManifest.read(path)
                self.assertTrue(
                    any("startframe" in item for item in tampered.verify(project))
                )

            payload = restored.to_dict()
            payload["startframe_source_clip"]["sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            tampered = SceneWorkflowManifest.read(path)
            self.assertTrue(
                any("source clip" in item for item in tampered.verify(project))
            )

            payload = restored.to_dict()
            payload["startframe_source_clip"]["path"] = "other/final.mp4"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tampered = SceneWorkflowManifest.read(path)
            self.assertTrue(
                any("source clip" in item for item in tampered.verify(project))
            )

    def test_verify_preserves_distinct_actor_bindings_for_shared_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            shared = project / "shared.png"
            for path in (workflow, template, plan, shared):
                path.write_bytes(path.name.encode())
            shared_sha = sha256_file(shared)
            contract = SceneConsistencyContract.create(
                scene=1,
                mode="msr",
                workflow_profile="msr",
                actors=tuple(
                    ReferenceAnchor(
                        id=actor_id,
                        kind="actor",
                        look_id="default",
                        asset_role="identity-reference",
                        asset_sha256=shared_sha,
                        prompt_anchor=actor_id,
                    )
                    for actor_id in ("hero", "double")
                ),
                location=None,
                transition_from_previous="cut",
            )
            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=1,
                pipeline="ltx_msr",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[
                    ("actor_sheet", shared, "shared.png", "double"),
                    ("actor_sheet", shared, "shared.png", "hero"),
                ],
                seed=100001,
                fps=24,
                frame_count=49,
                width=1280,
                height=704,
                consistency=contract,
            )

            self.assertEqual([], manifest.verify(project))
            self.assertEqual(
                ["double", "hero"],
                [asset.reference_id for asset in manifest.assets],
            )

            payload = manifest.to_dict()
            payload["assets"][1]["reference_id"] = "double"
            path = project / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tampered = SceneWorkflowManifest.read(path)
            self.assertTrue(
                any("actor_sheet" in item for item in tampered.verify(project))
            )

    def test_verify_rejects_msr_contract_for_i2v_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            actor = project / "actor.png"
            location = project / "location.png"
            for path in (workflow, template, plan, actor, location):
                path.write_bytes(path.name.encode())
            contract = self._contract(
                1, sha256_file(actor), sha256_file(location)
            )
            manifest = SceneWorkflowManifest.create(
                project_dir=project,
                scene=1,
                pipeline="ltx_i2v",
                workflow_path=workflow,
                template_path=template,
                render_plan_path=plan,
                assets=[
                    ("actor_sheet", actor, "actor.png"),
                    ("location_sheet", location, "location.png"),
                ],
                seed=100001,
                fps=24,
                frame_count=49,
                width=1280,
                height=704,
                consistency=contract,
            )

            self.assertTrue(
                any(
                    "mode does not match manifest pipeline" in item
                    for item in manifest.verify(project)
                )
            )

    def test_external_template_is_absolute_and_explicitly_marked(self):
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as external_dir:
            project = Path(temp_dir)
            external = Path(external_dir) / "template.json"
            external.write_text("{}", encoding="utf-8")
            artifact = StoredArtifact.from_path(external, project_dir=project, allow_external=True)

            self.assertTrue(artifact.external)
            self.assertEqual(str(external.resolve()), artifact.path)

    def test_non_template_external_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as external_dir:
            with self.assertRaisesRegex(ValueError, "outside project"):
                StoredArtifact.from_path(Path(external_dir) / "asset.png", project_dir=Path(temp_dir))

    def test_verify_reports_every_changed_or_missing_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = project / "workflow.json"
            template = project / "template.json"
            plan = project / "plan.json"
            asset = project / "asset.png"
            for path in (workflow, template, plan, asset):
                path.write_text(path.name, encoding="utf-8")
            manifest = SceneWorkflowManifest.create(
                project_dir=project, scene=1, pipeline="ltx_msr", workflow_path=workflow,
                template_path=template, render_plan_path=plan,
                assets=[("actor_sheet", asset, "feverslop/references/actor.png")],
                seed=1, fps=24, frame_count=25, width=1280, height=704,
            )
            workflow.write_text("changed", encoding="utf-8")
            plan.unlink()

            mismatches = manifest.verify(project)

            self.assertEqual(2, len(mismatches))
            self.assertTrue(any("workflow" in item and "sha256" in item for item in mismatches))
            self.assertTrue(any("render_plan" in item and "missing" in item for item in mismatches))

    def test_prepared_workflow_points_to_manifest_and_workflow(self):
        prepared = PreparedSceneWorkflow(
            scene=3,
            scene_dir=Path("scene_0003"),
            workflow_path=Path("scene_0003/workflow.json"),
            manifest_path=Path("scene_0003/manifest.json"),
        )
        self.assertEqual(3, prepared.scene)


if __name__ == "__main__":
    unittest.main()
