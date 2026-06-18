import argparse
import inspect
import unittest
from pathlib import Path

import compact_relay_prompts
import fix_ltx_prompt_anchors
import ltx_video_renderer
import main
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
            render_ltx,
            render_storyboard,
            compact_relay_prompts,
            fix_ltx_prompt_anchors,
        ]

        for module in modules:
            with self.subTest(module=module.__name__):
                parser = module.build_arg_parser()
                self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_root_compatibility_modules_document_new_import_paths(self):
        self.assertIn("adapters.comfyui_video_backend", inspect.getdoc(ltx_video_renderer) or "")
        self.assertIn("adapters.comfyui_rendering", inspect.getdoc(storyboard_renderer) or "")

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

    def test_root_python_files_are_only_cli_or_compatibility_facades(self):
        allowed = {
            "main.py",
            "render_ltx.py",
            "render_storyboard.py",
            "compact_relay_prompts.py",
            "fix_ltx_prompt_anchors.py",
            "storyboard_page.py",
            "normalize_render_plan.py",
            "repair_scene_srt.py",
            "trim_existing_ltx_clips.py",
            "ltx_video_renderer.py",
            "storyboard_renderer.py",
            "workflow_patcher.py",
        }
        actual = {path.name for path in Path(".").glob("*.py")}
        unexpected = sorted(actual - allowed)

        self.assertEqual([], unexpected)


if __name__ == "__main__":
    unittest.main()
