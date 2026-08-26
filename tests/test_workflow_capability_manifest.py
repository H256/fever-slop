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
