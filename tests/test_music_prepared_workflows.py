import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from feverslop.composition.arg_parser import PipelineStage
from feverslop.composition.config_loader import PipelineRunState, build_run_context
from feverslop.composition.stage_runners import (
    STAGE_RUNNERS,
    _run_ltx_prepare_workflows_stage,
    _run_ltx_render_scenes_stage,
    resolve_pipeline_stages,
)


class MusicPreparedWorkflowStageTests(unittest.TestCase):
    def _state(self, project: Path, *, pipeline: str, scenes: str = "") -> PipelineRunState:
        config = project / "config.json"
        config.write_text(json.dumps({"project_name": "Song", "input_audio": "song.mp3"}), encoding="utf-8")
        args = argparse.Namespace(
            project_config=str(config), project_root=None, video_pipeline=pipeline,
            render_mode="single_prompt", smoke_only=False, scenes=scenes, smoke_scene=1,
            no_skip_existing=False, randomize_seed=False, rolling_frame_profile="off",
            video_character_lora_strength=None, video_lora_1_strength_model=None,
            video_lora_1_strength_clip=None, lora_split_enabled=False,
            single_prompt_title="#PROMPT", single_prompt_input="text", relay_workflow="",
        )
        context = build_run_context(args)
        return PipelineRunState(
            args=args, context=context, app_config_path=project / "app.json",
            storyboard_workflow=project / "storyboard.json",
            reference_hero_workflow=project / "hero.json", reference_edit_workflow=project / "edit.json",
            msr_workflow=project / "msr.json", ingredients_workflow=project / "ingredients.json",
            relay_workflow=Path(""), single_prompt_workflow=project / "i2v.json",
            plan_for_next_step=context.ingredients_plan if pipeline == "ltx_ingredients" else context.reference_plan,
        )

    def test_default_specialized_pipeline_prepares_before_render(self):
        args = argparse.Namespace(
            stages=None, skip_tests=True, skip_main_pipeline=True, skip_relay_compact=True,
            skip_anchor_fix=True, video_pipeline="ltx_msr", skip_msr_reference_render=True,
            skip_msr_prompt_enrichment=True, skip_ingredients_sheets=True, skip_ltx=False,
            skip_final_concat=True, render_mode="single_prompt", skip_storyboard=True,
            skip_storyboard_page=True, diagnostic_original_audio_mux=False,
            no_original_audio_mux=False,
        )

        stages = resolve_pipeline_stages(args)

        self.assertLess(stages.index(PipelineStage.LTX_PREPARE_WORKFLOWS), stages.index(PipelineStage.LTX_RENDER_SCENES))
        self.assertIn(PipelineStage.LTX_PREPARE_WORKFLOWS, STAGE_RUNNERS)

    def test_prepare_uses_same_scene_selection_and_never_queues(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1,3")
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([
                {"scene": number, "ingredients": {
                    "sheet_path": f"sheet{number}.png", "anchors": [], "global_prompt": f"scene {number}",
                }, "ltx": {"base_prompt": f"scene {number}", "static_prompt": f"scene {number}", "prompt_relay": [
                    {"frame_start": 0, "frame_end": 48, "state": "instrumental", "prompt": "mouth closed"},
                ]}}
                for number in (1, 2, 3)
            ]), encoding="utf-8")
            for number in (1, 2, 3):
                (project / f"sheet{number}.png").write_bytes(b"sheet")
            use_case = Mock()
            backend = use_case.backend
            materializer = Mock()
            with patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case), \
                 patch("feverslop.composition.stage_runners.WorkflowMaterializer", return_value=materializer):
                _run_ltx_prepare_workflows_stage(state)

        self.assertEqual([1, 3], [call.args[0].scene["scene"] for call in materializer.prepare.call_args_list])
        backend.render_queue.queue_workflow_and_download_first_video.assert_not_called()

    def test_render_requires_prepared_scene_and_names_prepare_stage(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr", scenes="5")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([{"scene": 5}]), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "--stage ltx_prepare_workflows"):
                _run_ltx_render_scenes_stage(state)

    def test_prepare_aggregates_missing_plan_audio_and_template(self):
        with TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), pipeline="ltx_ingredients")

            with self.assertRaises(FileNotFoundError) as raised:
                _run_ltx_prepare_workflows_stage(state)

            message = str(raised.exception)
            self.assertIn("render plan", message)
            self.assertIn("audio", message)
            self.assertIn("workflow template", message)

    def test_prepare_failure_rolls_back_manifests_created_in_same_invocation(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1,2")
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([
                {"scene": 1, "ingredients_scene_sheet": "sheet1.png"},
                {"scene": 2, "ingredients_scene_sheet": "sheet2.png"},
            ]), encoding="utf-8")
            for number in (1, 2):
                (project / f"sheet{number}.png").write_bytes(b"sheet")
            use_case = Mock()
            materializer = Mock()

            def prepare(request):
                layout = state.context.artifact_layout
                layout.scene_workflow(request.scene["scene"]).parent.mkdir(parents=True, exist_ok=True)
                layout.scene_workflow(request.scene["scene"]).write_text("{}")
                layout.scene_manifest(request.scene["scene"]).write_text("{}")
                if request.scene["scene"] == 2:
                    raise RuntimeError("failed")

            materializer.prepare.side_effect = prepare
            with patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case), \
                 patch("feverslop.composition.stage_runners.WorkflowMaterializer", return_value=materializer), \
                 self.assertRaisesRegex(RuntimeError, "failed"):
                _run_ltx_prepare_workflows_stage(state)

            self.assertFalse(state.context.artifact_layout.scene_manifest(1).exists())
            self.assertFalse(state.context.artifact_layout.scene_manifest(2).exists())


if __name__ == "__main__":
    unittest.main()
