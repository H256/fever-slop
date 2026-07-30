import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from feverslop.adapters.pipeline_runner_options import build_runner_argv
from feverslop.composition.arg_parser import build_arg_parser as build_pipeline_parser
from feverslop.domain.visual_consistency import PreflightMode
from feverslop.tools.visual_consistency_preflight import (
    build_arg_parser,
    main,
    run,
)


class VisualConsistencyCliTests(unittest.TestCase):
    @staticmethod
    def _snapshot(project: Path) -> dict[str, bytes]:
        return {
            path.relative_to(project).as_posix(): path.read_bytes()
            for path in project.rglob("*")
            if path.is_file()
        }

    def _project(self, root: Path) -> Path:
        project = root / "demo"
        project.mkdir()
        (project / "config.json").write_text(
            json.dumps({"input_audio": "song.mp3", "actors": [], "locations": []}),
            encoding="utf-8",
        )
        plan = project / "output" / "render" / "plans" / "ingredients.json"
        plan.parent.mkdir(parents=True)
        plan.write_text(
            json.dumps([{"scene": 1, "references": {"actor_ids": ["missing"]}}]),
            encoding="utf-8",
        )
        return project

    def _valid_project(self, root: Path) -> Path:
        project = root / "valid"
        project.mkdir()
        (project / "config.json").write_text(
            json.dumps({"input_audio": "song.mp3"}),
            encoding="utf-8",
        )
        for kind, semantic_id in (("actor", "hero"), ("location", "stage")):
            directory = (
                project
                / "output"
                / "references"
                / f"{kind}s"
                / semantic_id
            )
            directory.mkdir(parents=True)
            asset = directory / "sheet.png"
            asset.write_bytes(kind.encode())
            (directory / "manifest.json").write_text(
                json.dumps({
                    "id": semantic_id,
                    "sheet_path": asset.relative_to(project).as_posix(),
                    "visual_description": f"{semantic_id} reference",
                }),
                encoding="utf-8",
            )
        sheet = project / "ingredients.png"
        sheet.write_bytes(b"sheet")
        plan = project / "output" / "render" / "plans" / "ingredients.json"
        plan.parent.mkdir(parents=True)
        plan.write_text(json.dumps([{
            "scene": 1,
            "references": {"actor_ids": ["hero"], "location_id": "stage"},
            "ingredients": {
                "sheet_path": "ingredients.png",
                "anchors": [{"id": "hero"}, {"id": "stage"}],
            },
        }]), encoding="utf-8")
        return project

    def test_json_validation_issues_exit_two(self):
        with TemporaryDirectory() as tmp:
            project = self._project(Path(tmp))
            before = self._snapshot(project)
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([
                    str(project),
                    "--plan",
                    "output/render/plans/ingredients.json",
                    "--mode",
                    "ingredients",
                    "--preflight-mode",
                    "strict",
                    "--json",
                ])
            after = self._snapshot(project)

        payload = json.loads(output.getvalue())
        self.assertEqual(2, status)
        self.assertFalse(payload["renderable"])
        self.assertEqual("missing_actor_reference", payload["issues"][0]["code"])
        self.assertEqual([], payload["contracts"])
        self.assertEqual(before, after)

    def test_strict_valid_plan_exits_zero_with_contract(self):
        with TemporaryDirectory() as tmp:
            project = self._valid_project(Path(tmp))
            before = self._snapshot(project)
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([
                    str(project),
                    "--plan",
                    "output/render/plans/ingredients.json",
                    "--mode",
                    "ingredients",
                    "--preflight-mode",
                    "strict",
                    "--json",
                ])
            after = self._snapshot(project)

        payload = json.loads(output.getvalue())
        self.assertEqual(0, status)
        self.assertTrue(payload["renderable"])
        self.assertEqual(1, len(payload["contracts"]))
        self.assertEqual([], payload["issues"])
        self.assertEqual(before, after)

    def test_strict_reports_missing_project_artifact_file(self):
        with TemporaryDirectory() as tmp:
            project = self._valid_project(Path(tmp))
            (project / "ingredients.png").unlink()
            before = self._snapshot(project)
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([
                    str(project),
                    "--preflight-mode",
                    "strict",
                    "--json",
                ])
            after = self._snapshot(project)

        payload = json.loads(output.getvalue())
        self.assertEqual(2, status)
        self.assertIn(
            "missing_ingredients_sheet_file",
            [issue["code"] for issue in payload["issues"]],
        )
        self.assertEqual(before, after)

    def test_strict_rejects_absolute_project_artifact_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._valid_project(root)
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            plan = project / "output" / "render" / "plans" / "ingredients.json"
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload[0]["ingredients"]["sheet_path"] = str(outside)
            plan.write_text(json.dumps(payload), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([
                    str(project),
                    "--preflight-mode",
                    "strict",
                    "--json",
                ])

        result = json.loads(output.getvalue())
        self.assertEqual(2, status)
        self.assertIn(
            "invalid_ingredients_sheet_path",
            [issue["code"] for issue in result["issues"]],
        )

    def test_rejects_plan_outside_project_as_unreadable_input(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._project(root)
            outside = root / "outside.json"
            outside.write_text("[]", encoding="utf-8")
            before = self._snapshot(project)
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([str(project), "--plan", str(outside), "--json"])
            after = self._snapshot(project)

        self.assertEqual(1, status)
        self.assertIn("inside the project", json.loads(output.getvalue())["error"])
        self.assertEqual(before, after)

    def test_malformed_project_config_exits_one(self):
        with TemporaryDirectory() as tmp:
            project = self._project(Path(tmp))
            (project / "config.json").write_text("{}", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([str(project), "--json"])

        self.assertEqual(1, status)
        self.assertIn("input_audio", json.loads(output.getvalue())["error"])

    def test_argparse_boundaries_return_typed_preflight_modes(self):
        cli_args = build_arg_parser().parse_args(["demo", "--preflight-mode", "strict"])
        pipeline_default = build_pipeline_parser().parse_args([])
        pipeline_strict = build_pipeline_parser().parse_args(
            ["--visual-consistency-preflight", "strict"]
        )

        self.assertIs(PreflightMode.STRICT, cli_args.preflight_mode)
        self.assertIs(PreflightMode.WARN, pipeline_default.visual_consistency_preflight)
        self.assertIs(PreflightMode.STRICT, pipeline_strict.visual_consistency_preflight)

    def test_help_renders_mode_values_without_enum_reprs(self):
        for help_text in (
            build_arg_parser().format_help(),
            build_pipeline_parser().format_help(),
        ):
            with self.subTest(help_text=help_text):
                self.assertIn("strict", help_text)
                self.assertIn("warn", help_text)
                self.assertIn("off", help_text)
                self.assertNotIn("PreflightMode.", help_text)

    def test_runner_argv_serializes_typed_mode_value(self):
        argv = build_runner_argv(
            Path("demo/config.json"),
            {"visual_consistency_preflight": PreflightMode.STRICT},
        )

        self.assertIn("--visual-consistency-preflight", argv)
        self.assertEqual("strict", argv[argv.index("--visual-consistency-preflight") + 1])

    def test_exact_module_entrypoint_invocation_returns_json_success(self):
        with TemporaryDirectory() as tmp:
            project = self._project(Path(tmp))
            before = self._snapshot(project)
            output = io.StringIO()
            argv = [
                "visual_consistency_preflight",
                str(project),
                "--plan",
                "output/render/plans/ingredients.json",
                "--mode",
                "ingredients",
                "--json",
            ]
            with patch("sys.argv", argv), redirect_stdout(output), self.assertRaises(
                SystemExit
            ) as exited:
                main()
            after = self._snapshot(project)

        payload = json.loads(output.getvalue())
        self.assertEqual(0, exited.exception.code)
        self.assertEqual({"renderable", "contracts", "issues"}, set(payload))
        self.assertTrue(payload["renderable"])
        self.assertEqual("missing_actor_reference", payload["issues"][0]["code"])
        self.assertEqual(before, after)

    def test_off_bypasses_unreadable_config_and_manifest_contract_inputs(self):
        with TemporaryDirectory() as tmp:
            project = self._project(Path(tmp))
            (project / "config.json").write_text("{}", encoding="utf-8")
            manifest = project / "output" / "references" / "actors" / "bad" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{bad", encoding="utf-8")
            before = self._snapshot(project)
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([
                    str(project),
                    "--preflight-mode",
                    "off",
                    "--json",
                ])
            after = self._snapshot(project)

        payload = json.loads(output.getvalue())
        self.assertEqual(0, status)
        self.assertEqual(
            {"renderable": True, "contracts": [], "issues": []},
            payload,
        )
        self.assertEqual(before, after)

    def test_human_output_reports_warn_mode_issues(self):
        with TemporaryDirectory() as tmp:
            project = self._project(Path(tmp))
            output = io.StringIO()
            with redirect_stdout(output):
                status = run([
                    str(project),
                    "--preflight-mode",
                    "warn",
                ])

        self.assertEqual(0, status)
        self.assertIn("renderable", output.getvalue())
        self.assertIn("missing_actor_reference", output.getvalue())

    def test_real_module_subprocess_returns_json_without_mutating_project(self):
        with TemporaryDirectory() as tmp:
            project = self._valid_project(Path(tmp))
            before = self._snapshot(project)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "feverslop.tools.visual_consistency_preflight",
                    str(project),
                    "--plan",
                    "output/render/plans/ingredients.json",
                    "--mode",
                    "ingredients",
                    "--preflight-mode",
                    "strict",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            after = self._snapshot(project)

        payload = json.loads(completed.stdout)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertTrue(payload["renderable"])
        self.assertEqual(1, len(payload["contracts"]))
        self.assertEqual([], payload["issues"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
