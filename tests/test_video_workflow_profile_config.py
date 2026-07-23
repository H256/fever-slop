import json
import tempfile
import unittest
from pathlib import Path

from feverslop.config.app_config import AppConfig


class VideoWorkflowProfileConfigTests(unittest.TestCase):
    def _load(self, profiles):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "app_config.json"
            path.write_text(
                json.dumps({"video_workflow_profiles": profiles}),
                encoding="utf-8",
            )
            return AppConfig.load(path)

    def _profile(self, **overrides):
        values = {
            "name": "ingredients-final",
            "pipeline": "ltx_ingredients",
            "workflow": "workflows/video_ltxv_ingredients_audio_2stage_v6.json",
            "purpose": "final",
            "stages": 2,
            "output_scale": 1.0,
            "supports_per_pass_loras": True,
        }
        values.update(overrides)
        return values

    def test_loads_profile_shape_and_capabilities(self):
        config = self._load([self._profile(default=True)])

        [profile] = config.video_workflow_profiles
        self.assertEqual("ingredients-final", profile.name)
        self.assertEqual("workflows/video_ltxv_ingredients_audio_2stage_v6.json", profile.workflow_path)
        self.assertEqual(2, profile.stages)
        self.assertEqual(1.0, profile.output_scale)
        self.assertTrue(profile.supports_per_pass_loras)
        self.assertTrue(profile.satisfies_final_output)
        self.assertIs(
            profile,
            config.resolve_video_workflow_profile(
                pipeline="ltx_ingredients", purpose="final"
            ),
        )

    def test_named_resolution_is_independent_of_default(self):
        config = self._load([
            self._profile(name="first", default=True),
            self._profile(name="second", workflow="workflows/second.json"),
        ])

        selected = config.resolve_video_workflow_profile(
            pipeline="ltx_ingredients", purpose="final", name="second"
        )

        self.assertEqual("second", selected.name)

    def test_returns_none_when_pipeline_purpose_has_no_default(self):
        config = self._load([self._profile()])

        self.assertIsNone(
            config.resolve_video_workflow_profile(
                pipeline="ltx_ingredients", purpose="final"
            )
        )

    def test_absent_section_preserves_backward_compatibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "app_config.json"
            path.write_text("{}", encoding="utf-8")
            config = AppConfig.load(path)

        self.assertEqual((), config.video_workflow_profiles)
        self.assertIsNone(
            config.resolve_video_workflow_profile(
                pipeline="ltx_ingredients", purpose="final"
            )
        )

    def test_rejects_duplicate_profile_names(self):
        with self.assertRaisesRegex(ValueError, "Duplicate video workflow profile name"):
            self._load([self._profile(), self._profile(workflow="workflows/other.json")])

    def test_rejects_multiple_defaults_for_pipeline_and_purpose(self):
        with self.assertRaisesRegex(ValueError, "Multiple default video workflow profiles"):
            self._load([
                self._profile(name="first", default=True),
                self._profile(name="second", workflow="workflows/second.json", default=True),
            ])

    def test_rejects_unknown_profile_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown video workflow profile fields: surprise"):
            self._load([self._profile(surprise=True)])

    def test_rejects_invalid_profile_path(self):
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            self._load([self._profile(workflow="../outside.json")])

    def test_rejects_non_boolean_default(self):
        with self.assertRaisesRegex(ValueError, "default must be a boolean"):
            self._load([self._profile(default="yes")])

    def test_rejects_named_profile_for_other_pipeline(self):
        config = self._load([self._profile()])

        with self.assertRaisesRegex(ValueError, "does not match pipeline/purpose"):
            config.resolve_video_workflow_profile(
                pipeline="ltx_msr", purpose="final", name="ingredients-final"
            )


if __name__ == "__main__":
    unittest.main()
