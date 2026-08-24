from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

import main
from feverslop.cli.run_cli import run_project_command
from feverslop.domain.execution_plan import ExecutionPlan, ExecutionPlanItem, PlanAction


class RunCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3",
            "video_pipeline": "ltx_msr",
            "private_prompt": "private prompt value",
        }), encoding="utf-8")
        (self.project / "song.mp3").write_bytes(b"audio")
        self.stream = io.StringIO()
        self.console = Console(file=self.stream, force_terminal=False, color_system=None, width=180)

    def _args(self, *extra: str) -> Namespace:
        return main.build_arg_parser().parse_args(["run", str(self.project), *extra])

    def _hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in self.project.rglob("*") if item.is_file()):
            digest.update(path.relative_to(self.project).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _plan(self, *, blocked: bool = False) -> ExecutionPlan:
        action = PlanAction.BLOCKED if blocked else PlanAction.RUN
        stage = None if blocked else "ltx_render_scenes"
        return ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("render", action, "workflow fingerprint changed" if not blocked else "run plan-migrate", 2, stage),
        ))

    def test_parser_exposes_dry_run_resume_and_advanced_stage(self):
        dry = self._args("--dry-run")
        resume = self._args("--resume", "--scenes", "2,4")
        advanced = self._args("--dry-run", "--stage", "anchor_fix")

        self.assertTrue(dry.dry_run)
        self.assertTrue(resume.resume)
        self.assertEqual("2,4", resume.scenes)
        self.assertEqual(["anchor_fix"], advanced.stages)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_dry_run_is_read_only_and_never_executes(self, pipeline_run):
        before = self._hash()
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=self._plan()):
            exit_code = run_project_command(self._args("--dry-run"), console=self.console)

        self.assertEqual(0, exit_code)
        pipeline_run.assert_not_called()
        self.assertEqual(before, self._hash())
        rendered = self.stream.getvalue()
        self.assertIn("RUN", rendered)
        self.assertNotIn("private prompt value", rendered)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_resume_executes_exact_dry_run_plan_and_scene_union(self, pipeline_run):
        plan = self._plan()
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan) as planner:
            exit_code = run_project_command(self._args("--resume"), console=self.console)

        self.assertEqual(0, exit_code)
        planner.assert_called_once_with(self.project.resolve(), video_pipeline="ltx_msr", selected_scenes=None)
        executed_args = pipeline_run.call_args.args[0]
        self.assertEqual(["ltx_render_scenes"], executed_args.stages)
        self.assertEqual("2", executed_args.scenes)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_unchanged_dry_run_and_resume_render_the_same_plan(self, pipeline_run):
        plan = self._plan()
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan), patch(
            "feverslop.cli.run_cli._render_plan",
        ) as render:
            self.assertEqual(0, run_project_command(self._args("--dry-run"), console=self.console))
            self.assertEqual(0, run_project_command(self._args("--resume"), console=self.console))

        self.assertEqual([plan, plan], [call.args[0] for call in render.call_args_list])

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_resume_executes_each_stage_with_its_own_scene_selection(self, pipeline_run):
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("references", PlanAction.RUN, "binding", 2, "msr_references"),
            ExecutionPlanItem("projection", PlanAction.RUN, "prompt", 1, "msr_prompt_enrich"),
        ))
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            self.assertEqual(0, run_project_command(self._args("--resume"), console=self.console))

        calls = pipeline_run.call_args_list
        self.assertEqual(2, len(calls))
        self.assertEqual((["msr_references"], "2"), (calls[0].args[0].stages, calls[0].args[0].scenes))
        self.assertEqual((["msr_prompt_enrich"], "1"), (calls[1].args[0].stages, calls[1].args[0].scenes))

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_blocked_plan_returns_two_without_execution(self, pipeline_run):
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=self._plan(blocked=True)):
            exit_code = run_project_command(self._args("--resume"), console=self.console)

        self.assertEqual(2, exit_code)
        pipeline_run.assert_not_called()
        self.assertIn("plan-migrate", self.stream.getvalue())

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_failure_reports_last_completed_stage_and_exact_resume_command(self, pipeline_run):
        def fail(_args, *, on_stage_complete):
            on_stage_complete("ltx_prepare_workflows")
            raise RuntimeError("render failed")

        pipeline_run.side_effect = fail
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("prepare", PlanAction.RUN, "stale", 2, "ltx_prepare_workflows"),
            ExecutionPlanItem("render", PlanAction.RUN, "missing", 2, "ltx_render_scenes"),
        ))
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            exit_code = run_project_command(self._args("--resume"), console=self.console)

        rendered = self.stream.getvalue()
        self.assertEqual(1, exit_code)
        self.assertIn("ltx_prepare_workflows", rendered)
        self.assertIn(f"uv run python main.py run {self.project.resolve()} --resume", rendered)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_advanced_stage_is_translated_to_execution_plan(self, pipeline_run):
        exit_code = run_project_command(
            self._args("--dry-run", "--stage", "anchor_fix"),
            console=self.console,
        )

        self.assertEqual(0, exit_code)
        pipeline_run.assert_not_called()
        self.assertIn("advanced stage", self.stream.getvalue())
        self.assertIn("anchor_fix", self.stream.getvalue())

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_explicit_runner_override_uses_compatibility_plan(self, pipeline_run):
        exit_code = run_project_command(
            self._args(
                "--dry-run",
                "--single-prompt-workflow",
                "workflows/custom.json",
            ),
            console=self.console,
        )

        self.assertEqual(0, exit_code)
        pipeline_run.assert_not_called()
        self.assertIn("advanced stage", self.stream.getvalue())


if __name__ == "__main__":
    unittest.main()
