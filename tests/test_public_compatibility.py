import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import compact_relay_prompts
import fix_ltx_prompt_anchors
import full_auto
import ltx_video_renderer
import main
import run_pipeline
import render_ltx
import render_storyboard
import storyboard_renderer
import workflow_patcher


class PublicCompatibilityTests(unittest.TestCase):
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
        import feverslop.tools.storyboard_page as storyboard_page

        self.assertIn(
            "src/feverslop/tools/storyboard_page.py",
            Path(storyboard_page.__file__).as_posix(),
        )
        self.assertTrue(hasattr(storyboard_page, "generate_storyboard_page"))

    def test_render_ltx_uses_importable_composition_root(self):
        source = inspect.getsource(render_ltx.main)

        self.assertIn("build_render_video_scenes_use_case", source)
        self.assertIn("namespace_to_options(args)", source)

    def test_cli_parser_defaults_stay_compatible(self):
        args = main.build_arg_parser().parse_args(["--project", "config.json"])
        self.assertEqual("config.json", args.project)
        self.assertEqual("app_config.json", args.app_config)
        self.assertEqual(0, args.concept_batch_size)

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
            ]
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
            ]
        )
        self.assertEqual("#PROMPT_POSITIVE", storyboard_args.positive_title)

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
        }
        actual = {path.name for path in Path(".").glob("*.py")}
        unexpected = sorted(actual - allowed)

        self.assertEqual([], unexpected)

    def test_root_compatibility_facades_do_not_use_star_imports(self):
        for name in ["ltx_video_renderer.py", "storyboard_renderer.py", "workflow_patcher.py"]:
            with self.subTest(name=name):
                self.assertNotIn("import *", Path(name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
