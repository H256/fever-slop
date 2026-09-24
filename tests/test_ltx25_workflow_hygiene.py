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

    def test_t2v_graph_is_distinct_from_i2v_and_has_no_image_input(self):
        """Regression for issue #1303: the t2v graph must not be a byte-identical
        copy of the i2v graph, and must not carry the i2v image-input/anchoring
        chain (it is a pure text-to-video graph)."""
        i2v = json.loads((WORKFLOW_ROOT / "i2v" / "i2v_draft.json").read_text(encoding="utf-8-sig"))
        for quality in ("draft", "standard", "final"):
            path = WORKFLOW_ROOT / "t2v" / f"t2v_{quality}.json"
            with self.subTest(quality=quality):
                t2v = json.loads(path.read_text(encoding="utf-8-sig"))
                self.assertNotEqual(
                    json.dumps(i2v, sort_keys=True),
                    json.dumps(t2v, sort_keys=True),
                    "t2v graph is a byte-identical copy of the i2v graph",
                )
                classes = {node.get("class_type") for node in t2v.values()}
                image_chain = classes & {
                    "LoadImage", "ResizeImageMaskNode", "GetImageSize",
                    "EmptyImage", "ImageScaleBy", "LTXVPreprocess",
                    "LTXVImgToVideoInplace",
                }
                self.assertFalse(
                    image_chain,
                    f"t2v graph still carries i2v image-input nodes: {sorted(image_chain)}",
                )
