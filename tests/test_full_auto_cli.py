import unittest
from pathlib import Path
from unittest.mock import patch

import full_auto
from feverslop.cli import full_auto as canonical_full_auto
from feverslop.domain.visual_consistency import PreflightMode


class FullAutoCliTests(unittest.TestCase):
    def test_root_cli_is_a_compatibility_facade_for_package_cli(self):
        self.assertIs(full_auto.build_arg_parser, canonical_full_auto.build_arg_parser)
        self.assertIs(full_auto.parse_optional_bool, canonical_full_auto.parse_optional_bool)
        self.assertIs(full_auto.request_from_args, canonical_full_auto.request_from_args)

    @patch("feverslop.cli.full_auto.build_full_auto_use_case")
    def test_package_cli_runs_full_auto_request(self, build_use_case):
        args = canonical_full_auto.build_arg_parser().parse_args(
            ["--idea", "friendship", "--style", "bright pop"],
        )

        canonical_full_auto.run_full_auto_command(args)

        build_use_case.assert_called_once()
        build_use_case.return_value.execute.assert_called_once()

    def test_parser_accepts_required_idea_style_and_runner_passthrough(self):
        args = full_auto.build_arg_parser().parse_args(
            [
                "--idea",
                "friendship",
                "--style",
                "bright pop",
                "--project-name",
                "Joy Demo",
                "--duration-seconds",
                "90.5",
                "--width",
                "1024",
                "--height",
                "576",
                "--fps",
                "50",
                "--language",
                "en",
                "--bpm",
                "123",
                "--keyscale",
                "D major",
                "--seed",
                "42",
                "--run-video-pipeline",
                "--concept-batch-size",
                "5",
                "--storyboard-workflow",
                "storyboard.json",
                "--relay-workflow",
                "relay.json",
                "--single-prompt-workflow",
                "single.json",
                "--render-mode",
                "auto",
                "--single-prompt-title",
                "#PROMPT_POSITIVE",
                "--single-prompt-input",
                "text",
                "--skip-tests",
                "--smoke-only",
                "--smoke-scene",
                "3",
                "--rolling-frame-profile",
                "safe",
                "--storyboard-lora-strength",
                "0.4",
                "--video-character-lora-strength",
                "0.8",
                "--video-lora-1-strength-model",
                "0.7",
                "--video-lora-1-strength-clip",
                "0.6",
                "--lora-split-enabled",
                "--no-skip-existing",
                "--skip-main-pipeline",
                "--skip-relay-compact",
                "--skip-anchor-fix",
                "--skip-storyboard",
                "--skip-storyboard-page",
                "--skip-ltx",
                "--skip-final-concat",
                "--diagnostic-original-audio-mux",
                "--no-original-audio-mux",
                "--silent-mode",
            ],
        )

        self.assertEqual("friendship", args.idea)
        self.assertEqual("bright pop", args.style)
        self.assertEqual("Joy Demo", args.project_name)
        self.assertEqual(90.5, args.duration_seconds)
        self.assertEqual(1024, args.width)
        self.assertEqual(576, args.height)
        self.assertEqual(50, args.fps)
        self.assertEqual("en", args.language)
        self.assertEqual(123, args.bpm)
        self.assertEqual("D major", args.keyscale)
        self.assertEqual(42, args.seed)
        self.assertTrue(args.run_video_pipeline)
        self.assertEqual(5, args.concept_batch_size)
        self.assertEqual("storyboard.json", args.storyboard_workflow)
        self.assertEqual("relay.json", args.relay_workflow)
        self.assertEqual("single.json", args.single_prompt_workflow)
        self.assertEqual("auto", args.render_mode)
        self.assertEqual("#PROMPT_POSITIVE", args.single_prompt_title)
        self.assertTrue(args.skip_tests)
        self.assertTrue(args.smoke_only)
        self.assertEqual(3, args.smoke_scene)
        self.assertEqual("safe", args.rolling_frame_profile)
        self.assertEqual(0.4, args.storyboard_lora_strength)
        self.assertEqual(0.8, args.video_character_lora_strength)
        self.assertEqual(0.7, args.video_lora_1_strength_model)
        self.assertEqual(0.6, args.video_lora_1_strength_clip)
        self.assertTrue(args.lora_split_enabled)
        self.assertTrue(args.no_skip_existing)
        self.assertTrue(args.skip_main_pipeline)
        self.assertTrue(args.skip_relay_compact)
        self.assertTrue(args.skip_anchor_fix)
        self.assertTrue(args.skip_storyboard)
        self.assertTrue(args.skip_storyboard_page)
        self.assertTrue(args.skip_ltx)
        self.assertTrue(args.skip_final_concat)
        self.assertTrue(args.diagnostic_original_audio_mux)
        self.assertTrue(args.no_original_audio_mux)
        self.assertTrue(args.silent_mode)

    def test_parser_accepts_explicit_silent_mode_false(self):
        args = full_auto.build_arg_parser().parse_args(
            [
                "--idea",
                "friendship",
                "--style",
                "bright pop",
                "--silent-mode",
                "false",
            ],
        )

        self.assertFalse(args.silent_mode)

    def test_request_maps_explicit_music_style(self):
        args = full_auto.build_arg_parser().parse_args(
            [
                "--idea",
                "water beings fight fire",
                "--style",
                "dark gothic visual fantasy",
                "--music-style",
                "epic fantasy power metal",
            ],
        )

        request = full_auto.request_from_args(args)

        self.assertEqual("dark gothic visual fantasy", request.style)
        self.assertEqual("epic fantasy power metal", request.music_style)

    def test_request_from_args_maps_runner_options(self):
        args = full_auto.build_arg_parser().parse_args(
            [
                "--idea",
                "friendship",
                "--style",
                "bright pop",
                "--projects-dir",
                "projects_out",
                "--width",
                "1024",
                "--height",
                "576",
                "--run-video-pipeline",
                "--concept-batch-size",
                "5",
                "--storyboard-workflow",
                "storyboard.json",
                "--relay-workflow",
                "relay.json",
                "--single-prompt-workflow",
                "single.json",
                "--render-mode",
                "auto",
                "--skip-tests",
                "--smoke-only",
                "--smoke-scene",
                "2",
                "--rolling-frame-profile",
                "off",
                "--video-lora-1-strength-model",
                "0.7",
                "--no-lora-split-enabled",
                "--skip-ltx",
                "--silent-mode",
            ],
        )

        request = full_auto.request_from_args(args)

        self.assertEqual(Path("projects_out"), request.projects_dir)
        self.assertEqual(1024, request.width)
        self.assertEqual(576, request.height)
        self.assertEqual(24, request.fps)
        self.assertTrue(request.run_video_pipeline)
        self.assertTrue(request.silent_mode)
        expected_options = {
                "app_config": "app_config.json",
                "resolution": None,
                "concept_batch_size": 5,
                "storyboard_workflow": "storyboard.json",
                "reference_hero_workflow": str(Path("workflows/image/image-model") / "image_t2i_startframe_krea_v1.json"),
                "reference_edit_workflow": str(Path("workflows/image/image-model") / "image_edit_flux2_klein_1ref_v1.json"),
                "msr_workflow": str(Path("workflows") / "video" / "ltx_25" / "msr" / "msr_draft.json"),
                "ingredients_workflow": str(Path("workflows") / "video" / "ltx_25" / "ingredients" / "ingredients_draft.json"),
                "relay_workflow": "relay.json",
                "single_prompt_workflow": "single.json",
                "video_pipeline": "ltx_i2v",
                "render_mode": "auto",
                "single_prompt_title": "#PROMPT",
                "single_prompt_input": "text",
                "storyboard_lora_strength": None,
                "video_character_lora_strength": None,
                "video_lora_1_strength_model": 0.7,
                "video_lora_1_strength_clip": None,
                "lora_split_enabled": False,
                "randomize_seed": False,
                "scenes": None,
                "skip_tests": True,
                "smoke_only": True,
                "smoke_scene": 2,
                "rolling_frame_profile": "off",
                "visual_consistency_preflight": PreflightMode.WARN,
                "no_skip_existing": False,
                "skip_main_pipeline": False,
                "skip_relay_compact": False,
                "skip_anchor_fix": False,
                "skip_storyboard": False,
                "skip_storyboard_page": False,
                "skip_msr_reference_render": False,
                "skip_msr_prompt_enrichment": False,
                "skip_ingredients_sheets": False,
                "skip_ltx": True,
                "skip_final_concat": False,
                "skip_facefix": True,
                "facefix_debug": False,
                "facefix_workflow": str(Path("workflows") / "video_ltxv_facefix_v1.json"),
                "diagnostic_original_audio_mux": False,
                "no_original_audio_mux": False,
        }
        for key, value in expected_options.items():
            with self.subTest(key=key):
                self.assertEqual(value, request.runner_options[key])


if __name__ == "__main__":
    unittest.main()
