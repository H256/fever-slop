import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import run_pipeline


class RunPipelinePathTests(unittest.TestCase):
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

            with patch("run_pipeline.run_unittest_suite") as tests, \
                patch("run_pipeline.build_generate_render_plan_use_case") as main_builder, \
                patch("run_pipeline.OpenAICompatibleLLMClient") as llm, \
                patch("run_pipeline.LTXPromptAnchorFixer") as fixer, \
                patch("run_pipeline.build_render_storyboard_use_case") as storyboard_builder, \
                patch("run_pipeline.generate_storyboard_page") as storyboard_page, \
                patch("run_pipeline.build_render_video_scenes_use_case") as video_builder, \
                patch("run_pipeline.VideoPostProcessor") as postprocessor:
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

    def test_smoke_ltx_uses_selected_scene_and_forces_rerender(self):
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
                    "--skip-final-concat",
                    "--smoke-only",
                    "--smoke-scene",
                    "7",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = [project_dir / "output" / "render" / "ltx_single_prompt_smoke" / "final" / "scene_0007.mp4"]
            with patch("run_pipeline.build_render_video_scenes_use_case", return_value=use_case):
                run_pipeline.run(args)

        request = use_case.execute.call_args.args[0]
        self.assertEqual({7}, request.scene_numbers)
        self.assertFalse(request.skip_existing)


def context_path(config_path: Path) -> Path:
    return config_path.parent / "output" / "render" / "render_plan_song.json"


if __name__ == "__main__":
    unittest.main()
