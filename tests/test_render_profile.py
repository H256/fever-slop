import unittest

from feverslop.domain.render_profile import (
    PostprocessStrategy,
    QualityProfile,
    RenderMode,
    RenderProfile,
    RenderProfileSchemaError,
    RenderPassStrategy,
)


class RenderProfileTests(unittest.TestCase):
    def make_profile(self, **overrides):
        values = {
            "profile_id": "minimax-h3-r2v-draft-2pass",
            "model_family": "minimax_h3",
            "mode": "r2v",
            "quality": "draft",
            "pass_strategy": "two_pass",
            "postprocess": "none",
            "capabilities": ["audio", "continuation", "reference_images"],
        }
        values.update(overrides)
        return RenderProfile.create(**values)

    def test_normalizes_and_freezes_profile_axes(self):
        profile = self.make_profile(
            profile_id="  MiniMax-H3-R2V-Draft-2Pass ",
            model_family=" MiniMax_H3 ",
            mode=" R2V ",
            quality=" DRAFT ",
            pass_strategy=" TWO_PASS ",
            postprocess=" NONE ",
            capabilities=["reference_images", "audio", "audio"],
        )

        self.assertEqual(1, profile.schema_version)
        self.assertEqual("minimax-h3-r2v-draft-2pass", profile.profile_id)
        self.assertEqual("minimax_h3", profile.model_family)
        self.assertIs(QualityProfile.DRAFT, profile.quality)
        self.assertIs(RenderMode.R2V, profile.mode)
        self.assertIs(RenderPassStrategy.TWO_PASS, profile.pass_strategy)
        self.assertIs(PostprocessStrategy.NONE, profile.postprocess)
        self.assertEqual(("audio", "reference_images"), profile.capabilities)

    def test_round_trips_as_stable_mapping(self):
        profile = self.make_profile(max_duration_seconds=12.0)

        restored = RenderProfile.from_dict(profile.to_dict())

        self.assertEqual(profile, restored)
        self.assertEqual(
            {
                "schema_version": 1,
                "profile_id": "minimax-h3-r2v-draft-2pass",
                "model_family": "minimax_h3",
                "mode": "r2v",
                "quality": "draft",
                "pass_strategy": "two_pass",
                "postprocess": "none",
                "capabilities": ["audio", "continuation", "reference_images"],
                "max_duration_seconds": 12.0,
            },
            profile.to_dict(),
        )

    def test_rejects_unknown_axes_and_three_pass(self):
        for field, value in (
            ("quality", "preview"),
            ("pass_strategy", "three_pass"),
            ("postprocess", "unknown"),
            ("mode", "unknown"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(RenderProfileSchemaError):
                    self.make_profile(**{field: value})

    def test_rejects_invalid_identity_capabilities_and_duration(self):
        invalid = (
            {"profile_id": ""},
            {"model_family": ""},
            {"capabilities": ["", "audio"]},
            {"max_duration_seconds": 0},
            {"max_duration_seconds": -1},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(RenderProfileSchemaError):
                    self.make_profile(**overrides)


if __name__ == "__main__":
    unittest.main()
