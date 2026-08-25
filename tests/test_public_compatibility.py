import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compact_relay_prompts
import fix_ltx_prompt_anchors
import full_auto
import ltx_video_renderer
import main
import render_ltx
import render_storyboard
import run_pipeline
import storyboard_renderer
import workflow_patcher
from feverslop.cli import render_ltx as canonical_render_ltx
from feverslop.cli.movie_cli import build_movie_arg_parser
from feverslop.composition.generate_render_plan import (
    build_generate_render_plan_execution_request,
)


class PublicCompatibilityTests(unittest.TestCase):
    def test_unified_render_forms_share_argument_defaults(self):
        options = [
            "--project", "config.json",
            "--app-config", "global.json",
            "--render-storyboard",
            "--zimage-workflow", "zimage.json",
            "--concept-batch-size", "3",
            "--video-workflow", "video.json",
            "--rolling-frame-profile", "safe",
        ]
        subcommand = main.build_arg_parser().parse_args(["render", *options])
        legacy = main.build_arg_parser().parse_args(options)

        for name in (
            "project",
            "app_config",
            "render_storyboard",
            "zimage_workflow",
            "concept_batch_size",
            "video_workflow",
            "rolling_frame_profile",
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(subcommand, name), getattr(legacy, name))

    def test_unified_movie_parser_matches_canonical_movie_parser(self):
        options = [
            "projects/demo",
            "--stage", "openshot_export",
            "--skip-openshot-export",
            "--app-config", "global.json",
            "--reference-backend", "local",
            "--reference-generation", "sequence_sheet",
            "--render-backend", "local",
            "--hero-workflow", "hero.json",
            "--edit-workflow", "edit.json",
            "--director-workflow", "director.json",
            "--startframe-director-backend", "krea2",
            "--mask-workflow", "mask.json",
            "--identity-repair-workflow", "identity.json",
            "--detail-workflow", "detail.json",
            "--startframe-comfyui-base-url", "http://comfy",
            "--startframe-validator-base-url", "http://validator",
            "--startframe-validator-model", "model",
            "--msr-workflow", "msr.json",
            "--msr-i2v-workflow", "msr-i2v.json",
            "--i2v-workflow", "i2v.json",
            "--r2v-workflow", "r2v.json",
            "--sequence-to-sheet-workflow", "sheet.json",
            "--t2v-workflow", "t2v.json",
            "--ingredients-workflow", "ingredients.json",
            "--skip-movie-bible",
            "--force-movie-bible",
            "--movie-planner-backend", "deterministic",
            "--skip-movie-story-design",
            "--force-movie-story-design",
            "--skip-movie-screenplay",
            "--force-movie-screenplay",
            "--skip-movie-narrative",
            "--skip-movie-scene-cards",
            "--skip-movie-shot-cards",
            "--skip-movie-continuity",
            "--skip-movie-plan",
            "--skip-movie-references",
            "--skip-movie-msr-enrich",
            "--skip-movie-ingredients-sheets",
            "--skip-movie-render",
            "--force-movie-references",
            "--keyframe-mode", "start-end",
            "--movie-video-workflow", "minimax-h3-r2v",
            "--continuity-keyframes", "last-to-start",
            "--scenes", "1,3",
            "--write-debug-workflows",
            "--debug-workflows-dir", "debug",
        ]
        canonical = build_movie_arg_parser().parse_args(options)
        unified = main.build_arg_parser().parse_args(["movie", *options])

        self.assertEqual(
            vars(canonical),
            {key: getattr(unified, key) for key in vars(canonical)},
        )

    def test_legacy_renderer_imports_remain_available(self):
        self.assertTrue(hasattr(ltx_video_renderer, "LTXVideoRenderer"))
        self.assertTrue(hasattr(storyboard_renderer, "StoryboardRenderer"))
        self.assertTrue(hasattr(workflow_patcher, "WorkflowPatcher"))

    def test_public_cli_modules_expose_testable_arg_parsers(self):
        modules = [
            main,
            run_pipeline,
            render_ltx,
            render_storyboard,
            compact_relay_prompts,
            fix_ltx_prompt_anchors,
            full_auto,
        ]

        for module in modules:
            with self.subTest(module=module.__name__):
                parser = module.build_arg_parser()
                self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_root_compatibility_modules_document_new_import_paths(self):
        self.assertIn("adapters.comfyui_video_backend", inspect.getdoc(ltx_video_renderer) or "")
        self.assertIn("adapters.storyboard_renderer", inspect.getdoc(storyboard_renderer) or "")

    def test_storyboard_renderer_root_file_is_only_compatibility_facade(self):
        text = Path("storyboard_renderer.py").read_text(encoding="utf-8")

        self.assertIn(
            "from feverslop.adapters.storyboard_renderer import StoryboardRenderer",
            text,
        )
        self.assertNotIn("class StoryboardRenderer", text)

    def test_storyboard_page_package_module_exists(self):
        from feverslop.tools import storyboard_page

        self.assertIn(
            "src/feverslop/tools/storyboard_page.py",
            Path(storyboard_page.__file__).as_posix(),
        )
        self.assertTrue(hasattr(storyboard_page, "generate_storyboard_page"))

    def test_render_ltx_uses_importable_composition_root(self):
        source = inspect.getsource(canonical_render_ltx.main)

        self.assertIn("build_render_video_scenes_use_case", source)
        self.assertIn("namespace_to_options(args)", source)

    def test_cli_parser_defaults_stay_compatible(self):
        args = main.build_arg_parser().parse_args(["--project", "config.json"])
        self.assertEqual("config.json", args.project)
        self.assertEqual("app_config.json", args.app_config)
        self.assertEqual(0, args.concept_batch_size)
        self.assertEqual([], args.video_workflow)
        self.assertEqual("original", args.rolling_frame_profile)

        ltx_args = render_ltx.build_arg_parser().parse_args(
            [
                "--render-plan",
                "render_plan.json",
                "--workflow",
                "workflow.json",
                "--audio",
                "song.wav",
                "--storyboard-dir",
                "storyboard",
                "--output-dir",
                "ltx",
            ],
        )
        self.assertEqual("single_prompt", ltx_args.render_mode)
        self.assertEqual("original", ltx_args.rolling_frame_profile)

        storyboard_args = render_storyboard.build_arg_parser().parse_args(
            [
                "--render-plan",
                "render_plan.json",
                "--workflow",
                "workflow.json",
                "--output-dir",
                "storyboard",
            ],
        )
        self.assertEqual("#PROMPT_POSITIVE", storyboard_args.positive_title)

    def test_main_cli_forwards_repeatable_video_workflows_and_rolling_profile(self):
        argv = [
            "main.py",
            "--project",
            "config.json",
            "--video-workflow",
            "relay.json",
            "--video-workflow",
            "single.json",
            "--rolling-frame-profile",
            "safe",
        ]

        with patch("sys.argv", argv), patch.object(main, "execute_generate_render_plan") as execute:
            main.main()

        request = execute.call_args.args[0]
        self.assertEqual((Path("relay.json"), Path("single.json")), request.video_workflow_paths)
        self.assertEqual("safe", request.rolling_frame_profile)

    def test_main_cli_empty_workflows_do_not_override_valid_workflow_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project = temp / "config.json"
            project.write_text(
                json.dumps(
                    {
                        "project_name": "demo",
                        "input_audio": "song.wav",
                        "video": {"fps": 24},
                        "scene_generation": {"min_duration": 2.0, "max_duration": 30.0},
                    },
                ),
                encoding="utf-8",
            )
            app_config = temp / "app.json"
            app_config.write_text(
                json.dumps(
                    {
                        "comfyui": {
                            "default_max_render_duration_seconds": 12.0,
                            "video_workflow_limits": [
                                {
                                    "workflow": "optimized.json",
                                    "max_render_duration_seconds": 24.0,
                                },
                            ],
                        },
                    },
                ),
                encoding="utf-8",
            )
            argv = [
                "main.py",
                "--project",
                str(project),
                "--app-config",
                str(app_config),
                "--video-workflow",
                "",
                "--video-workflow",
                ".",
                "--video-workflow",
                "   ",
                "--video-workflow",
                "optimized.json",
            ]
            execution_requests = []

            def resolve(request, **_kwargs):
                execution_requests.append(build_generate_render_plan_execution_request(request))

            with patch("sys.argv", argv), patch.object(
                main,
                "execute_generate_render_plan",
                side_effect=resolve,
            ):
                main.main()

        policy = execution_requests[0].scene_duration_policy
        self.assertEqual(24.0, policy.max_render_duration_seconds)
        self.assertEqual("optimized.json", policy.limiting_workflow)

    def test_storyboard_render_plan_subset_matches_scene_filter_and_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            render_plan = Path(temp_dir) / "render_plan.json"
            render_plan.write_text(
                json.dumps([
                    {"scene": 1},
                    {"scene": 2},
                    {"scene": 3},
                ]),
                encoding="utf-8",
            )

            subset = render_storyboard.load_render_plan_subset(
                render_plan,
                scene_numbers={2, 3},
                limit=1,
            )

            self.assertEqual([2], [scene["scene"] for scene in subset])

    def test_root_python_files_are_only_public_cli_or_explicit_facades(self):
        allowed = {
            "main.py",
            "run_pipeline.py",
            "render_ltx.py",
            "render_storyboard.py",
            "compact_relay_prompts.py",
            "fix_ltx_prompt_anchors.py",
            "full_auto.py",
            "movie_pipeline.py",
            "storyboard_page.py",
            "normalize_render_plan.py",
            "repair_scene_srt.py",
            "scaffold_movie.py",
            "trim_existing_ltx_clips.py",
            "ltx_video_renderer.py",
            "storyboard_renderer.py",
            "workflow_patcher.py",
            "debug_facefix.py",
        }
        actual = {path.name for path in Path().glob("*.py")}
        unexpected = sorted(actual - allowed)

        self.assertEqual([], unexpected)

    def test_root_compatibility_facades_do_not_use_star_imports(self):
        for name in ["ltx_video_renderer.py", "storyboard_renderer.py", "workflow_patcher.py"]:
            with self.subTest(name=name):
                self.assertNotIn("import *", Path(name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
