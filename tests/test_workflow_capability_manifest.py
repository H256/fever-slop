import unittest

from feverslop.domain.workflow_capability_manifest import WorkflowCapabilityManifest


class WorkflowCapabilityManifestTests(unittest.TestCase):
    def test_resolves_complete_inventory(self):
        manifest = WorkflowCapabilityManifest.create(
            manifest_id="ltx-25", model_family="ltx", model_version="2.5",
            required_models=["transformer.safetensors"], required_nodes=["LTXVLatentUpsampler"],
        )
        result = manifest.validate_inventory(
            models=["transformer.safetensors"], nodes=["LTXVLatentUpsampler"],
        )
        self.assertTrue(result.ok)

    def test_reports_missing_without_legacy_fallback(self):
        manifest = WorkflowCapabilityManifest.create(
            manifest_id="ltx-25", model_family="ltx", model_version="2.5",
            required_models=["transformer.safetensors"], required_nodes=["LTXVLatentUpsampler"],
        )
        result = manifest.validate_inventory(models=[], nodes=[])
        self.assertFalse(result.ok)
        self.assertIn("transformer.safetensors", result.missing_models)
        self.assertIn("LTXVLatentUpsampler", result.missing_nodes)
        self.assertFalse(result.legacy_fallback_allowed)

    def test_ltx25_manifest_is_versioned_and_not_legacy(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "workflows/video/ltx_25/capabilities.json").read_text(encoding="utf-8"))
        manifest = WorkflowCapabilityManifest.create(**payload)
        self.assertEqual("2.5", manifest.model_version)
        self.assertTrue(all("2.3" not in name.lower() for name in manifest.required_models))

    def test_ltx25_profile_matrix_covers_modes_and_quality(self):
        import json
        from pathlib import Path
        from feverslop.domain.render_profile import RenderProfile

        root = Path(__file__).resolve().parents[1]
        entries = json.loads((root / "workflows/video/ltx_25/profile-matrix.json").read_text(encoding="utf-8"))
        profiles = [RenderProfile.create(model_family="ltx-2.5", **entry) for entry in entries]
        self.assertEqual(12, len(profiles))
        self.assertTrue(all(profile.pass_strategy.value == "two_pass" for profile in profiles))
        self.assertEqual({"t2v", "i2v", "msr", "ingredients"}, {profile.mode.value for profile in profiles})

    def test_ltx25_t2v_profiles_use_only_25_model_assets(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        paths = sorted(
            path for path in (root / "workflows/video/ltx_25/t2v").glob("t2v_*.json")
            if not path.name.endswith(".profile.json")
        )
        self.assertEqual(3, len(paths))
        for path in paths:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            serialized = json.dumps(workflow).lower()
            self.assertNotIn("2.3", serialized)
            self.assertIn("ltx-2.5", serialized)
