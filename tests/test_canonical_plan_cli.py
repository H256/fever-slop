from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

import main
from feverslop.cli.canonical_plan_cli import run_canonical_plan_command
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.domain.effective_render_plan import CanonicalSceneDependencies, project_effective_plan
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.scene_artifacts import SceneArtifactLayout


class CanonicalPlanCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.plans = self.project / "output/render/plans"
        self.plans.mkdir(parents=True)
        self.output = io.StringIO()
        self.console = Console(file=self.output, force_terminal=False, color_system=None, width=180)

    def _scene(self, *, override: bool = False) -> dict:
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.Z_IMAGE: "generated secret prompt"},
        )
        if override:
            canonical["roles"][PromptRole.Z_IMAGE]["override"] = {
                "value": "human secret prompt",
                "provenance": {"source": "human"},
            }
        return {"scene": 1, "canonical": canonical, "z_image": {"prompt": "generated secret prompt"}}

    def _write_base(self, *, override: bool = False) -> dict:
        scene = self._scene(override=override)
        (self.plans / "base.json").write_text(json.dumps([scene]), encoding="utf-8")
        return scene

    def _run(self, argv: list[str]) -> int:
        args = main.build_arg_parser().parse_args(argv)
        return run_canonical_plan_command(args, console=self.console)

    def _tree_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in self.project.rglob("*") if item.is_file()):
            digest.update(path.relative_to(self.project).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def test_parser_exposes_plan_commands_and_status(self):
        cases = (
            (["plan", "path", "project"], ("plan", "path")),
            (["plan", "validate", "project"], ("plan", "validate")),
            (["plan", "show", "project", "--scene", "2"], ("plan", "show")),
            (["plan", "overrides", "project", "--orphans"], ("plan", "overrides")),
            (["status", "project"], ("status", None)),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                args = main.build_arg_parser().parse_args(argv)
                self.assertEqual(expected, (args.command, getattr(args, "plan_command", None)))

    def test_plan_path_labels_base_editable_and_derived_as_caches_without_writing(self):
        self._write_base()
        before = self._tree_hash()

        exit_code = self._run(["plan", "path", str(self.project)])

        rendered = self.output.getvalue()
        self.assertEqual(0, exit_code)
        self.assertIn("base.json", rendered)
        self.assertIn("SOLE EDITABLE PLAN", rendered)
        self.assertIn("references.json", rendered)
        self.assertIn("derived cache", rendered)
        self.assertEqual(before, self._tree_hash())

    def test_plan_show_is_the_only_command_that_prints_prompt_bodies(self):
        self._write_base(override=True)

        self.assertEqual(0, self._run(["plan", "show", str(self.project), "--scene", "1"]))

        rendered = self.output.getvalue()
        self.assertIn("generated secret prompt", rendered)
        self.assertIn("human secret prompt", rendered)
        self.assertIn("Effective", rendered)
        self.assertIn("human", rendered)

    def test_validate_distinguishes_valid_action_required_and_corrupt(self):
        scene = self._write_base()
        self.assertEqual(0, self._run(["plan", "validate", str(self.project)]))
        self.output.seek(0)
        self.output.truncate(0)
        (self.plans / "references.json").write_text(json.dumps([
            scene,
            {**self._scene(), "scene": 9, "canonical": build_canonical_scene(segment_id="orphan", generated_roles={})},
        ]), encoding="utf-8")
        self.assertEqual(2, self._run(["plan", "validate", str(self.project)]))
        self.output.seek(0)
        self.output.truncate(0)
        (self.plans / "base.json").write_text("{", encoding="utf-8")
        self.assertEqual(1, self._run(["plan", "validate", str(self.project)]))

    def test_validate_rejects_malformed_override_provenance_contract(self):
        scene = self._scene(override=True)
        scene["canonical"]["roles"][PromptRole.Z_IMAGE]["override"] = {}
        (self.plans / "base.json").write_text(json.dumps([scene]), encoding="utf-8")

        self.assertEqual(1, self._run(["plan", "validate", str(self.project)]))
        self.assertIn("override", self.output.getvalue())

    def test_validate_reports_stale_projection_identity_as_action_required(self):
        scene = self._write_base()
        derived = project_effective_plan([scene], [scene])
        derived[0]["canonical_projection"]["scene_id"] = "stale-scene-id"
        (self.plans / "references.json").write_text(json.dumps(derived), encoding="utf-8")

        self.assertEqual(2, self._run(["plan", "validate", str(self.project)]))
        rendered = self.output.getvalue()
        self.assertIn("stale projection identity", rendered)
        self.assertIn("scene 1", rendered)

    def test_status_reports_missing_without_exposing_prompt_and_is_read_only(self):
        self._write_base(override=True)
        before = self._tree_hash()

        exit_code = self._run(["status", str(self.project)])

        rendered = self.output.getvalue()
        self.assertEqual(2, exit_code)
        self.assertIn("MISSING", rendered)
        self.assertIn("required next phase", rendered)
        self.assertNotIn("generated secret prompt", rendered)
        self.assertNotIn("human secret prompt", rendered)
        self.assertEqual(before, self._tree_hash())

    def test_status_reports_ready_then_stale_workflow_without_printing_prompt(self):
        scene = self._write_base()
        layout = SceneArtifactLayout(self.project)
        projected = project_effective_plan([scene], [scene])[0]
        dependencies = CanonicalSceneDependencies.from_dict(
            projected["canonical_projection"]["dependencies"],
        )
        workflow = layout.scene_workflow(1)
        template = self.project / "workflow-template.json"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("{}", encoding="utf-8")
        template.write_text("{}", encoding="utf-8")
        SceneWorkflowManifest.create(
            project_dir=self.project,
            scene=1,
            pipeline="ltx_i2v",
            workflow_path=workflow,
            template_path=template,
            render_plan_path=layout.base_plan,
            assets=[],
            seed=1,
            fps=24,
            frame_count=25,
            width=1280,
            height=704,
            canonical_dependencies=dependencies,
        ).write(layout.scene_manifest(1))
        layout.scene_h3_prompt(1).write_text(json.dumps({
            "status": "good",
            "input_fingerprint": "sha256:fresh",
        }), encoding="utf-8")

        self.assertEqual(0, self._run(["status", str(self.project)]))
        self.assertIn("READY", self.output.getvalue())
        self.output.seek(0)
        self.output.truncate(0)
        scene["canonical"]["roles"][PromptRole.Z_IMAGE]["override"] = {
            "value": "changed private prompt",
            "provenance": {"source": "human"},
        }
        layout.base_plan.write_text(json.dumps([scene]), encoding="utf-8")

        self.assertEqual(2, self._run(["status", str(self.project)]))
        rendered = self.output.getvalue()
        self.assertIn("STALE", rendered)
        self.assertIn("workflow fingerprint changed", rendered)
        self.assertNotIn("changed private prompt", rendered)

    def test_every_inspection_command_preserves_complete_project_tree(self):
        self._write_base(override=True)
        commands = (
            ["plan", "path", str(self.project)],
            ["plan", "validate", str(self.project)],
            ["plan", "show", str(self.project), "--scene", "1"],
            ["plan", "overrides", str(self.project)],
            ["plan", "overrides", str(self.project), "--orphans"],
            ["status", str(self.project)],
        )
        for argv in commands:
            with self.subTest(argv=argv):
                before = self._tree_hash()
                self._run(argv)
                self.assertEqual(before, self._tree_hash())


if __name__ == "__main__":
    unittest.main()
