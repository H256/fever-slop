import unittest

from feverslop.domain.h3_two_pass import (
    H3TwoPassSchemaError,
    H3TwoPassSpec,
    apply_h3_two_pass_patch,
)


class H3TwoPassTests(unittest.TestCase):
    def make_spec(self, **overrides):
        values = {
            "model_assets": ["minimax_h3", "vae"],
            "pass1_sampler": "euler",
            "pass1_scheduler": "normal",
            "pass1_steps": 20,
            "pass1_denoise": 1.0,
            "pass2_sampler": "euler_cfg1a",
            "pass2_scheduler": "sgm_uniform",
            "pass2_steps": 8,
            "pass2_denoise": 0.35,
            "preserve_audio_latent": True,
            "required_anchors": ["#PROMPT", "#FRAMECOUNT", "#PASS1", "#PASS2", "#AUDIO_LATENT"],
        }
        values.update(overrides)
        return H3TwoPassSpec.create(**values)

    def test_normalizes_and_serializes_two_pass_contract(self):
        spec = self.make_spec(pass1_sampler=" EULER ", model_assets=["vae", "minimax_h3", "vae"])

        self.assertEqual(("minimax_h3", "vae"), spec.model_assets)
        self.assertEqual("euler", spec.pass1_sampler)
        self.assertTrue(spec.preserve_audio_latent)
        self.assertEqual(spec, H3TwoPassSpec.from_dict(spec.to_dict()))

    def test_patches_sampler_scheduler_steps_and_denoise_anchors(self):
        spec = self.make_spec(required_anchors=["#PASS1", "#PASS2"])
        workflow = {
            "1": {"class_type": "KSampler", "inputs": {"sampler_name": "old", "scheduler": "old", "steps": 1, "denoise": 1}, "_meta": {"title": "#PASS1"}},
            "2": {"class_type": "KSampler", "inputs": {"sampler_name": "old", "scheduler": "old", "steps": 1, "denoise": 1}, "_meta": {"title": "#PASS2"}},
        }
        patched = apply_h3_two_pass_patch(workflow, spec)
        self.assertEqual("euler", patched["1"]["inputs"]["sampler_name"])
        self.assertEqual(20, patched["1"]["inputs"]["steps"])
        self.assertEqual(0.35, patched["2"]["inputs"]["denoise"])

    def test_rejects_invalid_pass_parameters_and_three_pass_shape(self):
        for overrides in (
            {"pass1_steps": 0},
            {"pass2_denoise": 1.1},
            {"required_anchors": ["#PROMPT", "#PASS3"]},
            {"preserve_audio_latent": "yes"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(H3TwoPassSchemaError):
                    self.make_spec(**overrides)

    def test_validates_required_workflow_anchors(self):
        spec = self.make_spec()

        spec.validate_workflow_anchors({"#PROMPT", "#FRAMECOUNT", "#PASS1", "#PASS2", "#AUDIO_LATENT"})
        with self.assertRaises(H3TwoPassSchemaError):
            spec.validate_workflow_anchors({"#PROMPT", "#FRAMECOUNT", "#PASS1"})


if __name__ == "__main__":
    unittest.main()
