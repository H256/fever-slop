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

    def _app_config(self, handoff: str) -> Path:
        path = self.project / "app_config.json"
        path.write_text(
            json.dumps({"execution": {"vram_handoff": handoff}}),
            encoding="utf-8",
        )
        return path

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

    def test_parser_does_not_expose_internal_settings_sync_stage(self):
        with self.assertRaises(SystemExit):
            self._args("--dry-run", "--stage", "sync_project_settings")

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
        call = planner.call_args
        self.assertEqual(self.project.resolve(), call.args[0])
        self.assertEqual("ltx_msr", call.kwargs["video_pipeline"])
        self.assertIsNotNone(call.kwargs["render_settings"])
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
        app_config = self._app_config("continuous")
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("references", PlanAction.RUN, "binding", 2, "msr_references"),
            ExecutionPlanItem("projection", PlanAction.RUN, "prompt", 1, "msr_prompt_enrich"),
        ))
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            self.assertEqual(
                0,
                run_project_command(
                    self._args("--resume", "--app-config", str(app_config)),
                    console=self.console,
                ),
            )

        calls = pipeline_run.call_args_list
        self.assertEqual(2, len(calls))
        self.assertEqual((["msr_references"], "2"), (calls[0].args[0].stages, calls[0].args[0].scenes))
        self.assertEqual((["msr_prompt_enrich"], "1"), (calls[1].args[0].stages, calls[1].args[0].scenes))

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_manual_handoff_executes_only_first_safe_resource_phase(self, pipeline_run):
        app_config = self._app_config("manual")
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("references", PlanAction.RUN, "missing", 2, "msr_references"),
            ExecutionPlanItem("bindings", PlanAction.RUN, "missing", 2, "msr_reference_sheets"),
            ExecutionPlanItem("H3 prompts", PlanAction.RUN, "missing", 2, "h3_prompts"),
            ExecutionPlanItem("render plan", PlanAction.RUN, "stale", 2, "render_plan"),
            ExecutionPlanItem("render", PlanAction.RUN, "missing", 2, "ltx_render_scenes"),
        ))
        next_plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("prompts", PlanAction.RUN, "missing", 2, "h3_prompts"),
        ))

        with patch(
            "feverslop.cli.run_cli.build_resume_plan",
            side_effect=[plan, next_plan],
        ):
            exit_code = run_project_command(
                self._args("--resume", "--app-config", str(app_config)),
                console=self.console,
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [["msr_references"], ["msr_reference_sheets"]],
            [call.args[0].stages for call in pipeline_run.call_args_list],
        )
        rendered = self.stream.getvalue()
        self.assertIn("Manual VRAM handoff", rendered)
        self.assertIn("unload ComfyUI", rendered)
        self.assertIn("load the LLM", rendered)
        self.assertIn("--app-config", rendered)
        self.assertIn(str(app_config), rendered)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_manual_handoff_resume_command_preserves_explicit_pipeline(self, pipeline_run):
        app_config = self._app_config("manual")
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("prompts", PlanAction.RUN, "missing", 1, "h3_prompts"),
            ExecutionPlanItem("render", PlanAction.RUN, "missing", 1, "ltx_render_scenes"),
        ))

        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            exit_code = run_project_command(
                self._args(
                    "--resume",
                    "--app-config",
                    str(app_config),
                    "--video-pipeline",
                    "minimax-h3-r2v",
                ),
                console=self.console,
            )

        self.assertEqual(0, exit_code)
        self.assertIn("--video-pipeline minimax-h3-r2v", self.stream.getvalue())

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_manual_handoff_dry_run_shows_phase_without_execution(self, pipeline_run):
        app_config = self._app_config("manual")
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("prompts", PlanAction.RUN, "missing", 1, "h3_prompts"),
            ExecutionPlanItem("plan", PlanAction.RUN, "stale", 1, "render_plan"),
            ExecutionPlanItem("render", PlanAction.RUN, "missing", 1, "ltx_render_scenes"),
        ))

        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            exit_code = run_project_command(
                self._args("--dry-run", "--app-config", str(app_config)),
                console=self.console,
            )

        self.assertEqual(0, exit_code)
        pipeline_run.assert_not_called()
        rendered = self.stream.getvalue()
        self.assertIn("Next manual execution phase: LLM", rendered)
        self.assertIn("Next required resource after that: ComfyUI", rendered)
        self.assertIn("Dry run: no project artifacts were changed", rendered)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_manual_handoff_replans_after_completed_phase(self, pipeline_run):
        app_config = self._app_config("manual")
        initial_plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("main", PlanAction.RUN, "missing", 1, "main_pipeline"),
        ))
        next_plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("references", PlanAction.RUN, "missing", 2, "msr_references"),
        ))

        with patch(
            "feverslop.cli.run_cli.build_resume_plan",
            side_effect=[initial_plan, next_plan],
        ):
            exit_code = run_project_command(
                self._args("--resume", "--app-config", str(app_config)),
                console=self.console,
            )

        self.assertEqual(0, exit_code)
        self.assertEqual([["main_pipeline"]], [call.args[0].stages for call in pipeline_run.call_args_list])
        rendered = self.stream.getvalue()
        self.assertIn("Manual VRAM handoff required", rendered)
        self.assertIn("unload the LLM and load ComfyUI", rendered)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_manual_handoff_is_not_printed_when_replan_is_complete(self, pipeline_run):
        app_config = self._app_config("manual")
        initial_plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("prompts", PlanAction.RUN, "missing", 1, "h3_prompts"),
            ExecutionPlanItem("plan", PlanAction.RUN, "stale", 1, "render_plan"),
            ExecutionPlanItem("render", PlanAction.RUN, "missing", 1, "ltx_render_scenes"),
        ))
        completed_plan = ExecutionPlan(self.project, "resume", ())

        with patch(
            "feverslop.cli.run_cli.build_resume_plan",
            side_effect=[initial_plan, completed_plan],
        ):
            exit_code = run_project_command(
                self._args("--resume", "--app-config", str(app_config)),
                console=self.console,
            )

        self.assertEqual(0, exit_code)
        self.assertNotIn("Manual VRAM handoff", self.stream.getvalue())

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_continuous_handoff_keeps_executing_all_safe_stages(self, pipeline_run):
        app_config = self._app_config("continuous")
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("references", PlanAction.RUN, "missing", 1, "msr_references"),
            ExecutionPlanItem("prompts", PlanAction.RUN, "missing", 1, "h3_prompts"),
        ))

        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            exit_code = run_project_command(
                self._args("--resume", "--app-config", str(app_config)),
                console=self.console,
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [["msr_references"], ["h3_prompts"]],
            [call.args[0].stages for call in pipeline_run.call_args_list],
        )
        self.assertNotIn("Manual VRAM handoff", self.stream.getvalue())

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_manual_handoff_does_not_slice_explicit_compatibility_stages(self, pipeline_run):
        app_config = self._app_config("manual")

        exit_code = run_project_command(
            self._args(
                "--resume",
                "--app-config",
                str(app_config),
                "--stage",
                "msr_references",
                "--stage",
                "h3_prompts",
            ),
            console=self.console,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, pipeline_run.call_count)
        self.assertEqual(
            ["msr_references", "h3_prompts"],
            pipeline_run.call_args.args[0].stages,
        )
        self.assertNotIn("Manual VRAM handoff", self.stream.getvalue())

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
    def test_llm_loading_failure_explains_required_resource(self, pipeline_run):
        app_config = self._app_config("manual")
        pipeline_run.side_effect = RuntimeError(
            "DSPy H3 generation failed: ServiceUnavailableError: model is still loading",
        )
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("H3 prompts", PlanAction.RUN, "missing", 7, "h3_prompts"),
        ))
        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            exit_code = run_project_command(
                self._args("--resume", "--app-config", str(app_config)),
                console=self.console,
            )

        self.assertEqual(1, exit_code)
        rendered = self.stream.getvalue()
        self.assertIn("LLM", rendered)
        self.assertIn("noch nicht bereit", rendered)
        self.assertIn("laden", rendered)

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
    def test_advanced_stage_preserves_explicit_scene_selection(self, pipeline_run):
        exit_code = run_project_command(
            self._args("--resume", "--stage", "ltx_render_scenes", "--scenes", "1"),
            console=self.console,
        )

        self.assertEqual(0, exit_code)
        pipeline_run.assert_called_once()
        self.assertEqual("1", pipeline_run.call_args.args[0].scenes)

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

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_explicit_workflow_override_keeps_other_project_workflows(self, pipeline_run):
        hero = Path.cwd() / "workflows" / "test_project_precedence_hero.json"
        edit = Path.cwd() / "workflows" / "test_project_precedence_edit.json"
        for path in (hero, edit):
            path.write_text("{}", encoding="utf-8")
            self.addCleanup(path.unlink, missing_ok=True)
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3",
            "video_pipeline": "ltx_msr",
            "workflows": {
                "reference_hero": "workflows/test_project_precedence_hero.json",
                "reference_edit": "workflows/test_project_precedence_edit.json",
            },
        }), encoding="utf-8")
        args = self._args(
            "--dry-run",
            "--msr-workflow",
            "workflows/one-off-video.json",
        )

        exit_code = run_project_command(args, console=self.console)

        self.assertEqual(0, exit_code)
        pipeline_run.assert_not_called()
        self.assertEqual("workflows/one-off-video.json", args.msr_workflow)
        self.assertTrue(args.reference_hero_workflow.endswith("test_project_precedence_hero.json"))
        self.assertTrue(args.reference_edit_workflow.endswith("test_project_precedence_edit.json"))

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_explicit_workflow_override_shadows_missing_config_workflow(self, pipeline_run):
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3",
            "video_pipeline": "ltx_msr",
            "workflows": {"video": "workflows/missing-on-this-machine.json"},
        }), encoding="utf-8")

        exit_code = run_project_command(
            self._args(
                "--dry-run",
                "--msr-workflow",
                "workflows/one-off-video.json",
            ),
            console=self.console,
        )

        self.assertEqual(0, exit_code)
        pipeline_run.assert_not_called()
        self.assertNotIn("Invalid/corrupt project", self.stream.getvalue())

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_plain_safe_run_resolves_project_dimensions_and_video_workflow(self, pipeline_run):
        workflow = Path.cwd() / "workflows" / "test_project_selected_video.json"
        workflow.write_text('{"steps": 8}', encoding="utf-8")
        self.addCleanup(workflow.unlink, missing_ok=True)
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3",
            "video_pipeline": "minimax-h3-r2v",
            "video": {"width": 1024, "height": 576},
            "workflows": {"video": "workflows/test_project_selected_video.json"},
        }), encoding="utf-8")

        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=self._plan()) as planner:
            exit_code = run_project_command(self._args("--dry-run"), console=self.console)

        self.assertEqual(0, exit_code)
        pipeline_run.assert_not_called()
        call = planner.call_args
        settings = call.kwargs["render_settings"]
        self.assertEqual((1024, 576), (settings.width, settings.height))
        self.assertEqual("workflows/test_project_selected_video.json", settings.video_workflow.path)

    @patch("feverslop.cli.run_cli.pipeline_run")
    def test_project_workflows_are_passed_to_unchanged_resume_execution(self, pipeline_run):
        video = Path.cwd() / "workflows" / "test_project_video.json"
        hero = Path.cwd() / "workflows" / "test_project_hero.json"
        edit = Path.cwd() / "workflows" / "test_project_edit.json"
        for path in (video, hero, edit):
            path.write_text("{}", encoding="utf-8")
            self.addCleanup(path.unlink, missing_ok=True)
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3",
            "video_pipeline": "minimax-h3-r2v",
            "workflows": {
                "video": "workflows/test_project_video.json",
                "reference_hero": "workflows/test_project_hero.json",
                "reference_edit": "workflows/test_project_edit.json",
            },
        }), encoding="utf-8")
        plan = ExecutionPlan(self.project, "resume", (
            ExecutionPlanItem("sync", PlanAction.RUN, "settings", None, "sync_project_settings"),
            ExecutionPlanItem("render", PlanAction.RUN, "workflow", 1, "ltx_render_scenes"),
        ))

        with patch("feverslop.cli.run_cli.build_resume_plan", return_value=plan):
            exit_code = run_project_command(self._args("--resume"), console=self.console)

        self.assertEqual(0, exit_code)
        self.assertEqual(2, pipeline_run.call_count)
        sync_args = pipeline_run.call_args_list[0].args[0]
        render_args = pipeline_run.call_args_list[1].args[0]
        self.assertIsNotNone(sync_args.project_render_settings)
        self.assertTrue(render_args.single_prompt_workflow.endswith("test_project_video.json"))
        self.assertTrue(render_args.reference_hero_workflow.endswith("test_project_hero.json"))
        self.assertTrue(render_args.reference_edit_workflow.endswith("test_project_edit.json"))


if __name__ == "__main__":
    unittest.main()
