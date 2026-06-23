import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from rich.progress import TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

import run_pipeline


class RunPipelinePathTests(unittest.TestCase):
    def test_run_pipeline_defaults_do_not_embed_windows_only_relative_prefixes(self):
        args = run_pipeline.build_arg_parser().parse_args([])

        self.assertEqual("app_config.json", args.app_config)
        self.assertNotIn(".\\", args.storyboard_workflow)
        self.assertNotIn(".\\", args.single_prompt_workflow)

    def test_runner_path_accepts_windows_relative_cli_paths(self):
        self.assertEqual(
            run_pipeline.runner_root() / "app_config.json",
            run_pipeline.resolve_runner_path(".\\app_config.json"),
        )

    def test_build_run_context_resolves_project_paths_like_test_ps1(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            input_dir = project_dir / "input"
            input_dir.mkdir(parents=True)
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "My Song: Final!",
                        "input_audio": "input/song demo.mp3",
                    }
                ),
                encoding="utf-8",
            )

            args = run_pipeline.build_arg_parser().parse_args([str(project_dir), "--smoke-only"])
            context = run_pipeline.build_run_context(args)

        self.assertEqual(config_path.resolve(), context.project_config_path)
        self.assertEqual((input_dir / "song demo.mp3").resolve(), context.input_audio)
        self.assertEqual("song demo", context.song_id)
        self.assertEqual(project_dir / "output" / "render" / "render_plan_song demo.json", context.render_plan)
        self.assertEqual(project_dir / "output" / "render" / "render_plan_song demo__compact.json", context.compact_plan)
        self.assertEqual(project_dir / "output" / "render" / "render_plan_song demo__compact_anchored.json", context.anchored_plan)
        self.assertEqual(project_dir / "output" / "render" / "storyboard" / "index.html", context.storyboard_page)
        self.assertEqual(project_dir / "output" / "render" / "ltx_single_prompt_smoke", context.ltx_dir)
        self.assertEqual(context.ltx_dir / "My_Song_Final_video_only.mp4", context.final_concat_video)
        self.assertEqual(context.ltx_dir / "My_Song_Final.mp4", context.final_concat)
        self.assertEqual(context.ltx_dir / "My_Song_Final_scene_audio_debug.mp4", context.final_concat_scene_audio_debug)


class RunPipelineOrchestrationTests(unittest.TestCase):
    def test_skip_flags_suppress_pipeline_steps(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-ltx",
                    "--skip-final-concat",
                ]
            )

            with patch("feverslop.composition.pipeline_runner.run_unittest_suite") as tests, \
                patch("feverslop.composition.pipeline_runner.build_generate_render_plan_use_case") as main_builder, \
                patch("feverslop.composition.pipeline_runner.OpenAICompatibleLLMClient") as llm, \
                patch("feverslop.composition.pipeline_runner.LTXPromptAnchorFixer") as fixer, \
                patch("feverslop.composition.pipeline_runner.build_render_storyboard_use_case") as storyboard_builder, \
                patch("feverslop.composition.pipeline_runner.generate_storyboard_page") as storyboard_page, \
                patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case") as video_builder, \
                patch("feverslop.composition.pipeline_runner.VideoPostProcessor") as postprocessor:
                result = run_pipeline.run(args)

        self.assertEqual(context_path(config_path), result.render_plan_path)
        tests.assert_not_called()
        main_builder.assert_not_called()
        llm.assert_not_called()
        fixer.assert_not_called()
        storyboard_builder.assert_not_called()
        storyboard_page.assert_not_called()
        video_builder.assert_not_called()
        postprocessor.assert_not_called()

    def test_runner_render_progress_has_percent_elapsed_and_eta_columns(self):
        progress = run_pipeline.RenderProgressReporter("Rendering", total=3)

        column_types = [type(column) for column in progress.progress.columns]

        self.assertIn(TaskProgressColumn, column_types)
        self.assertIn(TimeElapsedColumn, column_types)
        self.assertIn(TimeRemainingColumn, column_types)

    def test_smoke_ltx_uses_selected_scene_and_forces_rerender(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 7}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-final-concat",
                    "--smoke-only",
                    "--smoke-scene",
                    "7",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = [project_dir / "output" / "render" / "ltx_single_prompt_smoke" / "final" / "scene_0007.mp4"]
            with patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case", return_value=use_case):
                run_pipeline.run(args)

        request = use_case.execute.call_args.args[0]
        self.assertEqual({7}, request.scene_numbers)
        self.assertFalse(request.skip_existing)

    def test_storyboard_runner_wires_progress_callback(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 1}, {"scene": 2}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard-page",
                    "--skip-ltx",
                    "--skip-final-concat",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = []
            with patch("feverslop.composition.pipeline_runner.build_render_storyboard_use_case", return_value=use_case):
                run_pipeline.run(args)

        request = use_case.execute.call_args.args[0]
        self.assertIsNotNone(request.on_frame_complete)
        self.assertEqual(2, request.on_frame_complete.__self__.total)

    def test_ltx_runner_wires_progress_callback(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 1}, {"scene": 2}, {"scene": 3}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-final-concat",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = []
            with patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case", return_value=use_case):
                run_pipeline.run(args)

        request = use_case.execute.call_args.args[0]
        self.assertIsNotNone(request.on_scene_complete)
        self.assertEqual(3, request.on_scene_complete.__self__.total)

    def test_ltx_resume_rewrites_concat_list_from_all_returned_clips_before_final_concat(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            plan_path = render_dir / "render_plan_song.json"
            plan_path.write_text("[]", encoding="utf-8")
            clip_1 = render_dir / "ltx_single_prompt" / "final" / "scene_0001.mp4"
            clip_2 = render_dir / "ltx_single_prompt" / "final" / "scene_0002.mp4"
            clip_1.parent.mkdir(parents=True)
            clip_1.write_bytes(b"clip 1")
            clip_2.write_bytes(b"clip 2")
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = [clip_1, clip_2]
            postprocessor = Mock()
            postprocessor.concat_clips.return_value = render_dir / "ltx_single_prompt" / "Song_video_only.mp4"
            postprocessor.mux_original_audio.return_value = render_dir / "ltx_single_prompt" / "Song.mp4"
            with patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case", return_value=use_case), \
                patch("feverslop.composition.pipeline_runner.VideoPostProcessor", return_value=postprocessor):
                run_pipeline.run(args)

            concat_list = render_dir / "ltx_single_prompt" / "concat_list.txt"
            self.assertEqual(
                [
                    f"file '{clip_1.resolve().as_posix()}'",
                    f"file '{clip_2.resolve().as_posix()}'",
                ],
                concat_list.read_text(encoding="utf-8").splitlines(),
            )


def context_path(config_path: Path) -> Path:
    return config_path.parent / "output" / "render" / "render_plan_song.json"


if __name__ == "__main__":
    unittest.main()
