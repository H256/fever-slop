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
    def test_pipeline_stage_arg_executes_only_anchor_fix(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            prompts_dir = project_dir / "output" / "prompts"
            render_dir = project_dir / "output" / "render"
            prompts_dir.mkdir(parents=True)
            render_dir.mkdir(parents=True)
            (prompts_dir / "resolved_context_song.json").write_text(json.dumps({"subject": "Singer"}), encoding="utf-8")
            (render_dir / "render_plan_song.json").write_text(json.dumps([{"scene": 1, "prompt": "Singer sings"}]), encoding="utf-8")
            args = run_pipeline.build_arg_parser().parse_args([str(config_path), "--stage", "anchor_fix"])

            fixer = Mock()
            anchored_plan = render_dir / "render_plan_song__compact_anchored.json"

            def fix_file(*, input_render_plan, output_render_plan):
                output_render_plan.write_text(Path(input_render_plan).read_text(encoding="utf-8"), encoding="utf-8")
                return output_render_plan

            fixer.fix_file.side_effect = fix_file
            with patch("feverslop.composition.pipeline_runner.run_unittest_suite") as tests, \
                patch("feverslop.composition.pipeline_runner.LTXPromptAnchorFixer", return_value=fixer) as fixer_class, \
                patch("feverslop.composition.pipeline_runner.build_generate_render_plan_use_case") as main_builder, \
                patch("feverslop.composition.pipeline_runner.build_render_storyboard_use_case") as storyboard_builder, \
                patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case") as video_builder, \
                patch("feverslop.composition.pipeline_runner.VideoPostProcessor") as postprocessor:
                result = run_pipeline.run(args)

        tests.assert_not_called()
        main_builder.assert_not_called()
        storyboard_builder.assert_not_called()
        video_builder.assert_not_called()
        postprocessor.assert_not_called()
        fixer_class.assert_called_once_with(subject_anchor="Singer")
        fixer.fix_file.assert_called_once()
        self.assertEqual(anchored_plan, result.render_plan_path)

    def test_pipeline_stage_error_names_failed_stage(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args([str(config_path), "--stage", "mux_original_audio"])

            with self.assertRaisesRegex(RuntimeError, "Mux original audio failed"):
                run_pipeline.run(args)

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

    def test_ltx_msr_runner_does_not_require_relay_workflow_in_auto_mode(self):
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
                json.dumps([{"scene": 1}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--video-pipeline",
                    "ltx_msr",
                    "--render-mode",
                    "auto",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-msr-reference-render",
                    "--skip-final-concat",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = []
            with patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case", return_value=use_case) as builder:
                run_pipeline.run(args)

        options = builder.call_args.args[0]
        self.assertEqual("ltx_msr", options.video_pipeline)
        self.assertTrue(str(options.output_dir).endswith("ltx_msr"))

    def test_ltx_msr_runner_builds_references_enriches_msr_prompts_and_skips_storyboard(self):
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
            base_plan = render_dir / "render_plan_song.json"
            refs_plan = render_dir / "render_plan_song_refs.json"
            msr_plan = refs_plan
            base_plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--video-pipeline",
                    "ltx_msr",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-final-concat",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = []

            def enrich(input_plan, references_dir, output_plan, on_scene_complete=None):
                self.assertIsNotNone(on_scene_complete)
                refs_plan.write_text(json.dumps([{"scene": 1, "references": {}}]), encoding="utf-8")
                on_scene_complete(1, 1, 1)
                return Path(output_plan)

            def enrich_msr(input_plan, output_plan, *, llm, on_scene_complete=None):
                self.assertIsNotNone(llm)
                self.assertIsNotNone(on_scene_complete)
                msr_plan.write_text(json.dumps([{"scene": 1, "ltx": {"msr_prompt_relay": []}}]), encoding="utf-8")
                on_scene_complete(1, 1, 1)
                return Path(output_plan)

            with patch("feverslop.composition.pipeline_runner.build_render_storyboard_use_case") as storyboard_builder, \
                patch("feverslop.composition.pipeline_runner.generate_storyboard_page") as storyboard_page, \
                patch("feverslop.composition.pipeline_runner.render_reference_bible") as reference_bible, \
                patch("feverslop.composition.pipeline_runner.enrich_render_plan_with_reference_sheets", side_effect=enrich) as enrich_refs, \
                patch("feverslop.composition.pipeline_runner.enrich_render_plan_with_msr_prompts", side_effect=enrich_msr) as enrich_msr_prompts, \
                patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case", return_value=use_case) as video_builder:
                result = run_pipeline.run(args)

        storyboard_builder.assert_not_called()
        storyboard_page.assert_not_called()
        reference_bible.assert_called_once()
        self.assertEqual(base_plan, enrich_refs.call_args.args[0])
        self.assertEqual(project_dir / "output" / "references", enrich_refs.call_args.args[1])
        self.assertEqual(refs_plan, enrich_refs.call_args.args[2])
        self.assertIn("on_scene_complete", enrich_refs.call_args.kwargs)
        enrich_msr_prompts.assert_called_once()
        self.assertEqual(refs_plan, enrich_msr_prompts.call_args.args[0])
        self.assertEqual(refs_plan, enrich_msr_prompts.call_args.args[1])
        self.assertIn("on_scene_complete", enrich_msr_prompts.call_args.kwargs)
        self.assertEqual(refs_plan, result.render_plan_path)
        options = video_builder.call_args.args[0]
        request = use_case.execute.call_args.args[0]
        self.assertEqual("ltx_msr", options.video_pipeline)
        self.assertEqual(refs_plan, options.render_plan_path)
        self.assertEqual(refs_plan, request.render_plan_path)

    def test_ltx_msr_runner_can_resume_selected_scenes_without_prompt_enrichment(self):
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
            refs_plan = render_dir / "render_plan_song_refs.json"
            refs_plan.write_text(json.dumps([{"scene": 14}, {"scene": 15}, {"scene": 16}]), encoding="utf-8")
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 14}, {"scene": 15}, {"scene": 16}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--video-pipeline",
                    "ltx_msr",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-anchor-fix",
                    "--skip-msr-reference-render",
                    "--skip-msr-prompt-enrichment",
                    "--skip-final-concat",
                    "--scenes",
                    "15-16",
                ]
            )

            use_case = Mock()
            use_case.execute.return_value = []

            def enrich_refs(input_plan, references_dir, output_plan, on_scene_complete=None):
                return Path(output_plan)

            with patch("feverslop.composition.pipeline_runner.enrich_render_plan_with_reference_sheets", side_effect=enrich_refs), \
                patch("feverslop.composition.pipeline_runner.enrich_render_plan_with_msr_prompts") as enrich_msr_prompts, \
                patch("feverslop.composition.pipeline_runner.build_render_video_scenes_use_case", return_value=use_case):
                run_pipeline.run(args)

            enrich_msr_prompts.assert_not_called()
            request = use_case.execute.call_args.args[0]
            self.assertEqual({15, 16}, request.scene_numbers)

    def test_ltx_resume_rewrites_concat_list_from_full_render_plan_before_final_concat(self):
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
            plan_path.write_text(
                json.dumps([{"scene": 1}, {"scene": 2}, {"scene": 3}]),
                encoding="utf-8",
            )
            clip_1 = render_dir / "ltx_single_prompt" / "final" / "scene_0001.mp4"
            clip_2 = render_dir / "ltx_single_prompt" / "final" / "scene_0002.mp4"
            clip_3 = render_dir / "ltx_single_prompt" / "final" / "scene_0003.mp4"
            clip_1.parent.mkdir(parents=True)
            clip_1.write_bytes(b"clip 1")
            clip_2.write_bytes(b"clip 2")
            clip_3.write_bytes(b"clip 3")
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
            use_case.execute.return_value = [clip_2]
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
                    f"file '{clip_3.resolve().as_posix()}'",
                ],
                concat_list.read_text(encoding="utf-8").splitlines(),
            )

    def test_ltx_msr_skip_ltx_rewrites_concat_list_from_existing_scene_clips(self):
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
            plan_path = render_dir / "render_plan_song_refs.json"
            plan_path.write_text(
                json.dumps([{"scene": 1}, {"scene": 2}, {"scene": 3}]),
                encoding="utf-8",
            )
            base_plan = render_dir / "render_plan_song.json"
            base_plan.write_text(json.dumps([{"scene": 1}, {"scene": 2}, {"scene": 3}]), encoding="utf-8")
            clip_1 = render_dir / "ltx_msr" / "scene_0001.mp4"
            clip_2 = render_dir / "ltx_msr" / "scene_0002.mp4"
            clip_3 = render_dir / "ltx_msr" / "scene_0003.mp4"
            clip_1.parent.mkdir(parents=True)
            clip_1.write_bytes(b"clip 1")
            clip_2.write_bytes(b"clip 2")
            clip_3.write_bytes(b"clip 3")
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--video-pipeline",
                    "ltx_msr",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-msr-reference-render",
                    "--skip-ltx",
                ]
            )

            def enrich(_input_plan, _references_dir, output_plan, on_scene_complete=None):
                if on_scene_complete is not None:
                    on_scene_complete(1, 1, 3)
                    on_scene_complete(2, 2, 3)
                    on_scene_complete(3, 3, 3)
                return Path(output_plan)

            postprocessor = Mock()
            postprocessor.concat_clips.return_value = render_dir / "ltx_msr" / "Song_video_only.mp4"
            postprocessor.mux_original_audio.return_value = render_dir / "ltx_msr" / "Song.mp4"
            with patch("feverslop.composition.pipeline_runner.enrich_render_plan_with_reference_sheets", side_effect=enrich), \
                patch("feverslop.composition.pipeline_runner.VideoPostProcessor", return_value=postprocessor):
                run_pipeline.run(args)

            concat_list = render_dir / "ltx_msr" / "concat_list.txt"
            self.assertEqual(
                [
                    f"file '{clip_1.resolve().as_posix()}'",
                    f"file '{clip_2.resolve().as_posix()}'",
                    f"file '{clip_3.resolve().as_posix()}'",
                ],
                concat_list.read_text(encoding="utf-8").splitlines(),
            )


def context_path(config_path: Path) -> Path:
    return config_path.parent / "output" / "render" / "render_plan_song.json"


if __name__ == "__main__":
    unittest.main()
