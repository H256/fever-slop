"""Tests for the ``feverslop artifact prune`` CLI (issue #1388, piece 4a).

Covers parser registration, safe/apply exit codes, no-write guarantees,
archive-first deletion, the timestamped report file, per-candidate manifest
problems as non-fatal ineligibility, and a two-scene end-to-end smoke run.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rich.console import Console

import main
from feverslop.cli.artifact_cli import run_artifact_prune_command
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.domain.effective_render_plan import CanonicalSceneDependencies, project_effective_plan
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.scene_artifacts import SceneArtifactLayout


class ArtifactCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project"
        self.project.mkdir()
        self.layout = SceneArtifactLayout(self.project)
        self.output = io.StringIO()
        self.console = Console(file=self.output, force_terminal=False, color_system=None, width=180)

    def _scene_entry(self, scene_number: int) -> dict:
        canonical = build_canonical_scene(
            segment_id=f"segment-{scene_number}",
            generated_roles={PromptRole.Z_IMAGE: f"generated prompt {scene_number}"},
        )
        return {
            "scene": scene_number,
            "canonical": canonical,
            "z_image": {"prompt": f"generated prompt {scene_number}"},
        }

    def _write_base(self, scene_numbers: tuple[int, ...]) -> dict[int, dict]:
        scenes = {number: self._scene_entry(number) for number in scene_numbers}
        self.layout.plans_dir.mkdir(parents=True, exist_ok=True)
        self.layout.base_plan.write_text(
            json.dumps([scenes[number] for number in scene_numbers]), encoding="utf-8"
        )
        return scenes

    def _write_scene_manifest(self, scene_number: int, scene: dict) -> None:
        scene_dir = self.layout.scene_dir(scene_number)
        scene_dir.mkdir(parents=True, exist_ok=True)
        workflow = self.layout.scene_workflow(scene_number)
        workflow.write_text("{}", encoding="utf-8")
        template = self.project / "workflow-template.json"
        template.write_text("{}", encoding="utf-8")
        projected = project_effective_plan([scene])[0]
        dependencies = CanonicalSceneDependencies.from_dict(
            projected["canonical_projection"]["dependencies"]
        )
        SceneWorkflowManifest.create(
            project_dir=self.project,
            scene=scene_number,
            pipeline="ltx_i2v",
            workflow_path=workflow,
            template_path=template,
            render_plan_path=self.layout.base_plan,
            assets=[],
            seed=1,
            fps=24,
            frame_count=25,
            width=1280,
            height=704,
            canonical_dependencies=dependencies,
        ).write(self.layout.scene_manifest(scene_number))

    def _write_complete_scene(self, scene_number: int, scene: dict) -> None:
        self._write_scene_manifest(scene_number, scene)
        (self.layout.scene_dir(scene_number) / "raw.mp4").write_bytes(b"raw-video-bytes")
        self.layout.scene_final_video(scene_number).write_bytes(b"final-video-bytes")

    def _write_blocked_scene(self, scene_number: int) -> None:
        scene_dir = self.layout.scene_dir(scene_number)
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "workflow.json").write_text("{}", encoding="utf-8")
        (scene_dir / "raw.mp4").write_bytes(b"raw-video-bytes")

    def _write_config(self, *, upscale_enabled: bool = False) -> None:
        (self.project / "config.json").write_text(
            json.dumps({"upscale": {"enabled": upscale_enabled}}), encoding="utf-8"
        )

    def _build_two_scene_project(self) -> None:
        scenes = self._write_base((1, 2))
        self._write_complete_scene(1, scenes[1])
        self._write_blocked_scene(2)
        self._write_config()
        derived = project_effective_plan([scenes[1]], [scenes[1]])
        self.layout.references_plan.write_text(json.dumps(derived), encoding="utf-8")

    def _run(self, argv: list[str]) -> int:
        args = main.build_arg_parser().parse_args(argv)
        return run_artifact_prune_command(args, console=self.console)

    def _rendered(self) -> str:
        return self.output.getvalue()

    def _reset_output(self) -> None:
        self.output.seek(0)
        self.output.truncate(0)

    def _tree_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in self.project.rglob("*") if item.is_file()):
            digest.update(path.relative_to(self.project).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _archive_path(self, name: str = "prune.zip") -> Path:
        return Path(self.temp.name) / name


class ArtifactCliParserTests(ArtifactCliTests):
    def test_parser_exposes_artifact_prune_with_safe_apply_and_archive(self):
        safe_args = main.build_arg_parser().parse_args(
            ["artifact", "prune", str(self.project), "--safe"]
        )
        self.assertEqual("artifact", safe_args.command)
        self.assertEqual("prune", safe_args.artifact_command)
        self.assertTrue(safe_args.safe)
        self.assertFalse(safe_args.apply)
        self.assertIsNone(safe_args.archive)

        apply_args = main.build_arg_parser().parse_args(
            ["artifact", "prune", str(self.project), "--apply", "--archive", "a.zip"]
        )
        self.assertTrue(apply_args.apply)
        self.assertFalse(apply_args.safe)
        self.assertEqual("a.zip", apply_args.archive)

        with self.assertRaises(SystemExit):
            main.build_arg_parser().parse_args(
                ["artifact", "prune", str(self.project), "--safe", "--apply"]
            )

    def test_main_facade_is_the_production_entry_point(self):
        # ``main`` is a thin re-export of ``feverslop.cli.app``; the facade
        # parser must be the same object so the test exercises the real
        # production entry point.
        from feverslop.cli.app import build_arg_parser as app_build_arg_parser

        self.assertIs(app_build_arg_parser, main.build_arg_parser)
        args = app_build_arg_parser().parse_args(
            ["artifact", "prune", str(self.project), "--safe"]
        )
        self.assertEqual("artifact", args.command)
        self.assertEqual("prune", args.artifact_command)


class ArtifactCliSafeTests(ArtifactCliTests):
    def test_safe_run_prints_report_and_performs_no_writes(self):
        self._build_two_scene_project()
        before = self._tree_hash()

        exit_code = self._run(["artifact", "prune", str(self.project), "--safe"])

        rendered = self._rendered()
        self.assertEqual(2, exit_code)
        self.assertIn("feverslop.artifact-prune/v1", rendered)
        self.assertIn("output/render/scenes/scene_0001/workflow.json", rendered)
        self.assertIn("output/render/plans/references.json", rendered)
        self.assertIn("manifest_missing", rendered)
        self.assertIn("ACTION REQUIRED", rendered)
        self.assertEqual(before, self._tree_hash())

    def test_safe_run_without_eligible_candidates_exits_0(self):
        self._write_base((2,))
        self._write_blocked_scene(2)
        self._write_config()

        exit_code = self._run(["artifact", "prune", str(self.project), "--safe"])

        rendered = self._rendered()
        self.assertEqual(0, exit_code)
        self.assertIn("feverslop.artifact-prune/v1", rendered)
        self.assertIn("no eligible candidates", rendered)
        self.assertNotIn("ACTION REQUIRED", rendered)

    def test_safe_run_with_missing_project_exits_1(self):
        exit_code = self._run(
            ["artifact", "prune", str(self.project / "missing"), "--safe"]
        )

        self.assertEqual(1, exit_code)
        self.assertIn("Artifact prune failed", self._rendered())

    def test_corrupt_scene_manifest_blocks_candidate_without_failing(self):
        scenes = self._write_base((1,))
        self._write_complete_scene(1, scenes[1])
        self._write_config()
        self.layout.scene_manifest(1).write_text("{", encoding="utf-8")

        exit_code = self._run(["artifact", "prune", str(self.project), "--safe"])

        rendered = self._rendered()
        self.assertEqual(0, exit_code)
        self.assertIn("manifest_invalid", rendered)
        self.assertNotIn("Artifact prune failed", rendered)

    def test_incomplete_facefix_and_upscale_chains_block_deletion(self):
        scenes = self._write_base((1,))
        self._write_complete_scene(1, scenes[1])
        self._write_config(upscale_enabled=True)
        (self.layout.scene_dir(1) / "workflow_facefix.json").write_text("{}", encoding="utf-8")

        exit_code = self._run(["artifact", "prune", str(self.project), "--safe"])

        rendered = self._rendered()
        self.assertEqual(0, exit_code)
        self.assertIn("facefix_pending", rendered)
        self.assertIn("upscale_pending", rendered)
        self.assertIn("no eligible candidates", rendered)


class ArtifactCliApplyTests(ArtifactCliTests):
    def test_apply_without_archive_exits_1_and_writes_nothing(self):
        self._build_two_scene_project()
        before = self._tree_hash()

        exit_code = self._run(["artifact", "prune", str(self.project), "--apply"])

        rendered = self._rendered()
        self.assertEqual(1, exit_code)
        self.assertIn("--apply requires --archive PATH", rendered)
        self.assertEqual(before, self._tree_hash())

    def test_apply_with_archive_deletes_eligible_and_writes_report_file(self):
        self._build_two_scene_project()
        archive = self._archive_path()

        exit_code = self._run(
            [
                "artifact",
                "prune",
                str(self.project),
                "--apply",
                "--archive",
                str(archive),
            ]
        )

        rendered = self._rendered()
        self.assertEqual(0, exit_code)
        self.assertIn("OK pruned 3 file(s)", rendered)
        # Eligible candidates are gone.
        self.assertFalse(self.layout.scene_workflow(1).exists())
        self.assertFalse(self.layout.scene_raw_video(1).exists())
        self.assertFalse(self.layout.references_plan.exists())
        # Ineligible candidates and protected files survive.
        self.assertTrue(self.layout.scene_workflow(2).is_file())
        self.assertTrue(self.layout.scene_raw_video(2).is_file())
        self.assertTrue(self.layout.base_plan.is_file())
        self.assertTrue(self.layout.scene_manifest(1).is_file())
        self.assertTrue(self.layout.scene_final_video(1).is_file())
        self.assertTrue((self.project / "config.json").is_file())
        # The archive contains exactly the deleted files plus the manifest.
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as zip_file:
            self.assertEqual(
                {
                    "archive_manifest.json",
                    "output/render/plans/references.json",
                    "output/render/scenes/scene_0001/workflow.json",
                    "output/render/scenes/scene_0001/raw.mp4",
                },
                set(zip_file.namelist()),
            )
        # The timestamped machine-readable report file is written under output/.
        report_files = list((self.project / "output").glob("prune_report_*.json"))
        self.assertEqual(1, len(report_files))
        payload = json.loads(report_files[0].read_text(encoding="utf-8"))
        self.assertEqual("feverslop.artifact-prune/v1", payload["schema_version"])
        self.assertEqual("apply", payload["mode"])
        self.assertEqual(str(archive), payload["archive_path"])
        self.assertEqual(
            {
                "output/render/plans/references.json",
                "output/render/scenes/scene_0001/workflow.json",
                "output/render/scenes/scene_0001/raw.mp4",
            },
            {entry["path"] for entry in payload["deleted"]},
        )

    def test_apply_with_no_eligible_candidates_deletes_nothing(self):
        self._write_base((2,))
        self._write_blocked_scene(2)
        self._write_config()
        archive = self._archive_path()

        exit_code = self._run(
            [
                "artifact",
                "prune",
                str(self.project),
                "--apply",
                "--archive",
                str(archive),
            ]
        )

        self.assertEqual(0, exit_code)
        self.assertIn("OK pruned 0 file(s)", self._rendered())
        self.assertTrue(archive.is_file())
        self.assertTrue(self.layout.scene_workflow(2).is_file())
        self.assertTrue(self.layout.scene_raw_video(2).is_file())

    def test_exit_code_matrix(self):
        self._build_two_scene_project()
        cases = (
            (["--safe"], 2),
            (["--apply"], 1),
            (["--apply", "--archive", str(self._archive_path())], 0),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                self._reset_output()
                exit_code = self._run(["artifact", "prune", str(self.project), *argv])
                self.assertEqual(expected, exit_code)
        # After the apply deleted the eligible candidates, safe mode finds no
        # eligible candidates remaining and exits 0.
        self._reset_output()
        self.assertEqual(0, self._run(["artifact", "prune", str(self.project), "--safe"]))
        self.assertIn("no eligible candidates", self._rendered())


class ArtifactCliSmokeTests(ArtifactCliTests):
    def test_two_scene_smoke_end_to_end(self):
        self._build_two_scene_project()
        before = self._tree_hash()
        archive = self._archive_path()

        safe_code = self._run(["artifact", "prune", str(self.project), "--safe"])
        safe_rendered = self._rendered()
        self.assertEqual(2, safe_code)
        self.assertIn("feverslop.artifact-prune/v1", safe_rendered)
        self.assertIn("ACTION REQUIRED", safe_rendered)
        self.assertEqual(before, self._tree_hash())

        self._reset_output()
        apply_code = self._run(
            [
                "artifact",
                "prune",
                str(self.project),
                "--apply",
                "--archive",
                str(archive),
            ]
        )
        apply_rendered = self._rendered()
        self.assertEqual(0, apply_code)
        self.assertIn("OK pruned 3 file(s)", apply_rendered)
        self.assertFalse(self.layout.scene_workflow(1).exists())
        self.assertFalse(self.layout.scene_raw_video(1).exists())
        self.assertFalse(self.layout.references_plan.exists())
        self.assertTrue(self.layout.scene_workflow(2).is_file())
        self.assertTrue(self.layout.scene_manifest(1).is_file())
        self.assertTrue(archive.is_file())
        report_files = list((self.project / "output").glob("prune_report_*.json"))
        self.assertEqual(1, len(report_files))
        payload = json.loads(report_files[0].read_text(encoding="utf-8"))
        self.assertEqual("apply", payload["mode"])
        self.assertEqual(3, len(payload["deleted"]))


if __name__ == "__main__":
    unittest.main()
