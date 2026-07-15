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


class PreparedWorkflowManifestTests(unittest.TestCase):
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

            self.assertEqual("feverslop.scene-workflow/v1", payload["schema"])
            self.assertEqual("output/render/scenes/scene_0005/workflow.json", payload["workflow"]["path"])
            self.assertEqual(sha256_file(workflow), payload["workflow"]["sha256"])
            self.assertNotIn("external", payload["template"])
            self.assertEqual(manifest, SceneWorkflowManifest.read(path))

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
