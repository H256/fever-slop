import unittest

from feverslop.domain.h3_two_pass import H3TwoPassSchemaError, H3TwoPassSpec


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
