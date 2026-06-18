import unittest
from pathlib import Path


class RunnerScriptTests(unittest.TestCase):
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
