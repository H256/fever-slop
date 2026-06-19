import unittest
from pathlib import Path

import run_pipeline


class RunnerScriptTests(unittest.TestCase):
    def test_run_pipeline_parser_defaults_match_test_ps1(self):
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

    def test_test_ps1_forwards_project_config_to_ltx(self):
        script = Path("test.ps1").read_text(encoding="utf-8")

        self.assertIn('"--project-config", $projectConfigPath', script)
        self.assertNotIn('"--min-duration", "$sceneMinDuration"', script)
        self.assertNotIn('"--max-duration", "$sceneMaxDuration"', script)

    def test_test_ps1_forwards_optional_lora_strengths_when_set(self):
        script = Path("test.ps1").read_text(encoding="utf-8")

        self.assertIn("[Nullable[double]]$StoryboardLoraStrength", script)
        self.assertIn("[Nullable[double]]$VideoCharacterLoraStrength", script)
        self.assertIn("[Nullable[double]]$VideoLora1StrengthModel", script)
        self.assertIn("[Nullable[double]]$VideoLora1StrengthClip", script)
        self.assertIn("[Nullable[bool]]$LoraSplitEnabled", script)

        self.assertIn('if ($null -ne $StoryboardLoraStrength)', script)
        self.assertIn(
            '$storyboardArgs += @("--character-lora-strength", (Convert-ToInvariantString $StoryboardLoraStrength))',
            script,
        )
        self.assertIn('if ($null -ne $VideoCharacterLoraStrength)', script)
        self.assertIn(
            '$ltxArgs += @("--character-lora-strength", (Convert-ToInvariantString $VideoCharacterLoraStrength))',
            script,
        )
        self.assertIn('if ($null -ne $VideoLora1StrengthModel)', script)
        self.assertIn(
            '$ltxArgs += @("--lora-1-strength-model", (Convert-ToInvariantString $VideoLora1StrengthModel))',
            script,
        )
        self.assertIn('if ($null -ne $VideoLora1StrengthClip)', script)
        self.assertIn(
            '$ltxArgs += @("--lora-1-strength-clip", (Convert-ToInvariantString $VideoLora1StrengthClip))',
            script,
        )
        self.assertIn('if ($null -ne $LoraSplitEnabled)', script)
        self.assertIn('$ltxArgs += @(if ($LoraSplitEnabled) { "--lora-split-enabled" } else { "--no-lora-split-enabled" })', script)

    def test_test_ps1_can_skip_main_pipeline(self):
        script = Path("test.ps1").read_text(encoding="utf-8")

        self.assertIn("[switch]$SkipMainPipeline", script)
        self.assertIn("if (-not $SkipMainPipeline)", script)
        self.assertIn('Write-Step "Running main pipeline"', script)
        self.assertIn('"Skipping main pipeline; using existing timeline, prompts, and render plan."', script)

    def test_test_ps1_uses_sanitized_project_name_for_final_video(self):
        script = Path("test.ps1").read_text(encoding="utf-8")

        self.assertIn("function Convert-ToSafeFileStem", script)
        self.assertIn('$projectFileStem = Convert-ToSafeFileStem $projectConfigJson.project_name $songId', script)
        self.assertIn('$finalConcatVideo = Join-Path $ltxDir "${projectFileStem}_video_only.mp4"', script)
        self.assertIn('$finalConcat = Join-Path $ltxDir "${projectFileStem}.mp4"', script)
        self.assertIn('$finalConcatSceneAudioDebug = Join-Path $ltxDir "${projectFileStem}_scene_audio_debug.mp4"', script)

    def test_test_ps1_generates_storyboard_page_unless_skipped(self):
        script = Path("test.ps1").read_text(encoding="utf-8")

        self.assertIn("[switch]$SkipStoryboardPage", script)
        self.assertIn("$storyboardPage = Join-Path $storyboardDir \"index.html\"", script)
        self.assertIn("if (-not $SkipStoryboardPage)", script)
        self.assertIn('Write-Step "Generating storyboard page"', script)
        self.assertIn('Invoke-UvPython -Script "storyboard_page.py" -Arguments @(', script)
        self.assertIn('"--render-plan", $planForNextStep', script)
        self.assertIn('"--storyboard-dir", $storyboardDir', script)
        self.assertIn('"--output-html", $storyboardPage', script)


if __name__ == "__main__":
    unittest.main()
