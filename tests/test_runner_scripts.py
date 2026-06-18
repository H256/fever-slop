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


if __name__ == "__main__":
    unittest.main()
