import unittest

from feverslop.domain.video_workflow_profile import VideoWorkflowProfile


class VideoWorkflowProfileTests(unittest.TestCase):
    def create_profile(self, **overrides):
        values = {
            "name": "ingredients-final",
            "pipeline": "ltx_ingredients",
            "workflow_path": "workflows/video_ltxv_ingredients_audio_2stage_v5.json",
            "purpose": "final",
            "stages": 2,
            "output_scale": 1.0,
            "supports_per_pass_loras": True,
        }
        values.update(overrides)
        return VideoWorkflowProfile.create(**values)

    def test_creates_final_profile_with_normalized_values(self):
        profile = self.create_profile(
            name="  ingredients-final  ",
            pipeline="  ltx_ingredients  ",
            workflow_path="  workflows\\video.json  ",
            purpose="  FINAL  ",
        )

        self.assertEqual("ingredients-final", profile.name)
        self.assertEqual("ltx_ingredients", profile.pipeline)
        self.assertEqual("workflows/video.json", profile.workflow_path)
        self.assertEqual("final", profile.purpose)
        self.assertEqual(2, profile.stages)
        self.assertEqual(1.0, profile.output_scale)
        self.assertTrue(profile.supports_per_pass_loras)
        self.assertTrue(profile.satisfies_final_output)

    def test_creates_stage1_preview_profile(self):
        profile = self.create_profile(
            name="ingredients-preview",
            workflow_path="workflows/video_ltxv_ingredients_audio_stage1_preview_v1.json",
            purpose="preview",
            stages=1,
            output_scale=0.5,
            supports_per_pass_loras=False,
        )

        self.assertEqual("preview", profile.purpose)
        self.assertEqual(1, profile.stages)
        self.assertEqual(0.5, profile.output_scale)
        self.assertFalse(profile.supports_per_pass_loras)
        self.assertFalse(profile.satisfies_final_output)

    def test_rejects_invalid_purpose(self):
        with self.assertRaisesRegex(ValueError, "purpose must be preview or final"):
            self.create_profile(purpose="draft")

    def test_rejects_blank_required_strings(self):
        for field in ("name", "pipeline", "workflow_path"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "name, pipeline, and path are required"):
                    self.create_profile(**{field: "  "})

    def test_rejects_absolute_and_parent_traversal_paths(self):
        for workflow_path in (
            "/workflows/video.json",
            "C:\\workflows\\video.json",
            "\\\\server\\share\\video.json",
            "workflows/../video.json",
            "workflows\\..\\video.json",
        ):
            with self.subTest(workflow_path=workflow_path):
                with self.assertRaisesRegex(ValueError, "path must be repository-relative"):
                    self.create_profile(workflow_path=workflow_path)

    def test_normalizes_mixed_path_separators_to_posix(self):
        profile = self.create_profile(workflow_path="workflows\\nested/video.json")

        self.assertEqual("workflows/nested/video.json", profile.workflow_path)

    def test_rejects_invalid_stage_count(self):
        for stages in (0, 3, 1.5, True):
            with self.subTest(stages=stages):
                with self.assertRaisesRegex(ValueError, "stages must be 1 or 2"):
                    self.create_profile(stages=stages)

    def test_rejects_non_positive_output_scale(self):
        for output_scale in (0, -0.5):
            with self.subTest(output_scale=output_scale):
                with self.assertRaisesRegex(ValueError, "output_scale must be greater than zero"):
                    self.create_profile(output_scale=output_scale)

    def test_rejects_non_finite_and_non_real_output_scale(self):
        for output_scale in (True, float("nan"), float("inf"), float("-inf"), "large", []):
            with self.subTest(output_scale=output_scale):
                with self.assertRaisesRegex(ValueError, "output_scale must be greater than zero"):
                    self.create_profile(output_scale=output_scale)

    def test_rejects_non_boolean_per_pass_lora_support(self):
        for supports_per_pass_loras in ("false", 1, []):
            with self.subTest(supports_per_pass_loras=supports_per_pass_loras):
                with self.assertRaisesRegex(ValueError, "supports_per_pass_loras must be a boolean"):
                    self.create_profile(supports_per_pass_loras=supports_per_pass_loras)

    def test_rejects_non_boolean_explicit_final_output_flag(self):
        for satisfies_final_output in ("false", 0, []):
            with self.subTest(satisfies_final_output=satisfies_final_output):
                with self.assertRaisesRegex(ValueError, "satisfies_final_output must be a boolean or None"):
                    self.create_profile(satisfies_final_output=satisfies_final_output)

    def test_rejects_preview_profile_with_final_output_flag(self):
        with self.assertRaisesRegex(ValueError, "preview profile cannot satisfy final output"):
            self.create_profile(
                purpose="preview",
                stages=1,
                output_scale=0.5,
                supports_per_pass_loras=False,
                satisfies_final_output=True,
            )


if __name__ == "__main__":
    unittest.main()
