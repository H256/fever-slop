import json
import unittest
from pathlib import Path


WORKFLOW_ROOT = Path("workflows/video/ltx_25")


class LTX25WorkflowHygieneTests(unittest.TestCase):
    def test_active_workflows_do_not_reference_retired_ltx23_assets(self):
        for path in sorted(WORKFLOW_ROOT.glob("**/*.json")):
            if path.name.endswith(".profile.json") or path.name in {"capabilities.json", "profile-matrix.json"}:
                continue
            with self.subTest(workflow=path):
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                serialized = json.dumps(payload, ensure_ascii=False).lower()
                self.assertNotIn("ltx23", serialized)
                self.assertNotIn("ltx-2.3", serialized)

    def test_active_workflows_use_ltx25_core_model_names(self):
        required = {
            "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            "ltx-2.5-video-vae-conv-bf16.safetensors",
            "ltx-2.5-audio-vae-bf16.safetensors",
        }
        for path in sorted(WORKFLOW_ROOT.glob("**/*.json")):
            if path.name.endswith(".profile.json") or path.name in {"capabilities.json", "profile-matrix.json"}:
                continue
            serialized = json.dumps(json.loads(path.read_text(encoding="utf-8-sig")))
            with self.subTest(workflow=path):
                for model_name in required:
                    self.assertIn(model_name, serialized)
