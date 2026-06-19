import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from tools.repair_scene_srt import main as repair_scene_srt_main
import run_pipeline


class RunnerScriptTests(unittest.TestCase):
    def test_os_specific_runner_scripts_are_removed(self):
        self.assertFalse(Path("test.ps1").exists())
        self.assertFalse(Path("test.bat").exists())

    def test_run_pipeline_parser_defaults_match_python_runner_contract(self):
        args = run_pipeline.build_arg_parser().parse_args([])

        self.assertIsNone(args.project_root)
        self.assertIsNone(args.project_config)
        self.assertEqual(".\\app_config.json", args.app_config)
        self.assertEqual(10, args.concept_batch_size)
        self.assertEqual(".\\workflows\\image_t2i_startframe_v1.json", args.storyboard_workflow)
        self.assertEqual("", args.relay_workflow)
        self.assertEqual(".\\workflows\\video_ltxv_i2v_v1.json", args.single_prompt_workflow)
        self.assertEqual("single_prompt", args.render_mode)
        self.assertEqual("#PROMPT", args.single_prompt_title)
        self.assertEqual("text", args.single_prompt_input)
        self.assertEqual("original", args.rolling_frame_profile)
        self.assertEqual(16, args.smoke_scene)
        self.assertFalse(args.skip_main_pipeline)

    def test_run_pipeline_parser_accepts_powershell_parity_flags(self):
        args = run_pipeline.build_arg_parser().parse_args(
            [
                "projects/song",
                "--render-mode",
                "auto",
                "--relay-workflow",
                "relay.json",
                "--single-prompt-workflow",
                "single.json",
                "--storyboard-lora-strength",
                "0.4",
                "--video-character-lora-strength",
                "0.8",
                "--video-lora-1-strength-model",
                "0.7",
                "--video-lora-1-strength-clip",
                "0.6",
                "--lora-split-enabled",
                "--smoke-only",
                "--no-skip-existing",
                "--skip-tests",
                "--skip-main-pipeline",
                "--skip-relay-compact",
                "--skip-anchor-fix",
                "--skip-storyboard",
                "--skip-storyboard-page",
                "--skip-ltx",
                "--skip-final-concat",
                "--diagnostic-original-audio-mux",
                "--no-original-audio-mux",
            ]
        )

        self.assertEqual("projects/song", args.project_root)
        self.assertEqual("auto", args.render_mode)
        self.assertEqual(0.4, args.storyboard_lora_strength)
        self.assertEqual(0.8, args.video_character_lora_strength)
        self.assertEqual(0.7, args.video_lora_1_strength_model)
        self.assertEqual(0.6, args.video_lora_1_strength_clip)
        self.assertTrue(args.lora_split_enabled)
        self.assertTrue(args.smoke_only)
        self.assertTrue(args.no_skip_existing)
        self.assertTrue(args.skip_tests)
        self.assertTrue(args.skip_main_pipeline)
        self.assertTrue(args.skip_relay_compact)
        self.assertTrue(args.skip_anchor_fix)
        self.assertTrue(args.skip_storyboard)
        self.assertTrue(args.skip_storyboard_page)
        self.assertTrue(args.skip_ltx)
        self.assertTrue(args.skip_final_concat)
        self.assertTrue(args.diagnostic_original_audio_mux)
        self.assertTrue(args.no_original_audio_mux)

    def test_repair_scene_srt_cli_writes_repaired_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_srt = temp / "input.srt"
            output_srt = temp / "output.srt"
            input_srt.write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nScene 1\n\n"
                "2\n00:00:00,500 --> 00:00:02,000\nScene 2\n",
                encoding="utf-8",
            )

            argv = [
                "repair_scene_srt.py",
                "--input-srt",
                str(input_srt),
                "--output-srt",
                str(output_srt),
                "--min-duration",
                "1.0",
                "--max-duration",
                "3.0",
            ]
            with patch.object(sys, "argv", argv):
                repair_scene_srt_main()

            self.assertTrue(output_srt.exists())
            self.assertIn("00:00:00,000 --> 00:00:02,000", output_srt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
