import unittest

from feverslop.domain.render_profile import (
    PostprocessStrategy,
    QualityProfile,
    RenderMode,
    RenderProfile,
    RenderProfileSchemaError,
    RenderPassStrategy,
    RenderProfileRegistry,
    RegisteredRenderProfile,
    RenderProfileResolution,
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

    def test_registry_resolves_exact_profile_and_capabilities(self):
        profile = self.make_profile()
        registry = RenderProfileRegistry(
            [RegisteredRenderProfile(profile=profile, workflow_path="workflows/video/minimax_h3/r2v.json")]
        )

        resolved = registry.resolve(
            profile_id=profile.profile_id,
            required_capabilities={"audio", "continuation"},
        )

        self.assertEqual(profile, resolved.profile)
        self.assertEqual("workflows/video/minimax_h3/r2v.json", resolved.workflow_path)

    def test_registry_rejects_duplicates_and_unsupported_capabilities(self):
        profile = self.make_profile()
        entry = RegisteredRenderProfile(profile=profile, workflow_path="workflows/video/minimax_h3/r2v.json")
        with self.assertRaises(RenderProfileSchemaError):
            RenderProfileRegistry([entry, entry])

        registry = RenderProfileRegistry([entry])
        with self.assertRaises(RenderProfileSchemaError):
            registry.resolve(profile_id=profile.profile_id, required_capabilities={"does_not_exist"})

    def test_resolution_has_stable_semantic_fingerprint_and_provenance(self):
        profile = self.make_profile()
        entry = RegisteredRenderProfile(profile=profile, workflow_path="workflows/video/minimax_h3/r2v.json")
        resolution = RenderProfileResolution.create(
            requested_profile_id=" MiniMax-H3-R2V-Draft-2Pass ",
            entry=entry,
            workflow_sha256="a" * 64,
            model_assets=["minimax-h3.safetensors", "clip.safetensors"],
        )

        payload = resolution.to_dict()

        self.assertEqual(profile.profile_id, payload["requested_profile_id"])
        self.assertEqual("a" * 64, payload["workflow_sha256"])
        self.assertEqual(["clip.safetensors", "minimax-h3.safetensors"], payload["model_assets"])
        self.assertEqual(profile.capabilities, tuple(payload["capabilities"]))
        self.assertEqual(64, len(payload["fingerprint"]))
        self.assertEqual(resolution.fingerprint, payload["fingerprint"])

        changed = RenderProfileResolution.create(
            requested_profile_id=profile.profile_id,
            entry=entry,
            workflow_sha256="b" * 64,
            model_assets=payload["model_assets"],
        )
        self.assertNotEqual(resolution.fingerprint, changed.fingerprint)


if __name__ == "__main__":
    unittest.main()
