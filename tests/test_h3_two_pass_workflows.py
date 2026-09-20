import json
import unittest
from pathlib import Path


class H3TwoPassWorkflowTests(unittest.TestCase):
    def test_generated_profiles_declare_upscaler_memory_inputs(self):
        # MinimaxH3LatentUpscaler3D requires force_unload and
        # enable_temporal_chunking; ComfyUI rejects prompts missing them.
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "workflows" / "video" / "minimax_h3").glob("*_two_pass.json"))
        for path in paths:
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text(encoding="utf-8"))
                upscaler = next(node for node in workflow.values() if node.get("class_type") == "MinimaxH3LatentUpscaler3D")
                self.assertTrue(upscaler["inputs"]["force_unload"])
                self.assertTrue(upscaler["inputs"]["enable_temporal_chunking"])

    def test_generated_profiles_use_builtin_av_boundary_without_vrgdg_wrappers(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "workflows" / "video" / "minimax_h3").glob("*_two_pass.json"))
        for path in paths:
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text(encoding="utf-8"))
                classes = {node.get("class_type") for node in workflow.values()}
                self.assertIn("LTXVSeparateAVLatent", classes)
                self.assertIn("LTXVConcatAVLatent", classes)
                self.assertIn("MinimaxH3LatentUpscaler3D", classes)
                self.assertNotIn("PrimitiveFloat", classes)
                self.assertFalse(any(str(name).startswith("VRGDG_") for name in classes))
                upscaler = next(node for node in workflow.values() if node.get("class_type") == "MinimaxH3LatentUpscaler3D")
                self.assertEqual("scale by multiplier", upscaler["inputs"]["mode"])
                self.assertEqual(2, upscaler["inputs"]["mode.scale"])
                self.assertTrue(upscaler["inputs"]["keep_proportion"])
                self.assertFalse(upscaler["inputs"]["offload_after_upscale"])
                self.assertNotIn("scale", upscaler["inputs"])
                self.assertNotIn("width", upscaler["inputs"])
                self.assertNotIn("height", upscaler["inputs"])
                self.assertNotIn("megapixels", upscaler["inputs"])

    def test_i2v_profile_declares_frame_capabilities(self):
        # Issue #761: the I2V two-pass profile sidecar must declare the
        # start-only and start-end anchor capabilities consumed by the
        # I2V backend before ComfyUI submission.
        root = Path(__file__).resolve().parents[1]
        profile = json.loads(
            (root / "workflows" / "video" / "minimax_h3" / "i2v_two_pass.profile.json").read_text(encoding="utf-8")
        )
        self.assertEqual("i2v", profile["mode"])
        self.assertEqual(["start_only", "start_end"], profile["frame_capabilities"])
        self.assertEqual("two_pass", profile["pass_strategy"])


if __name__ == "__main__":
    unittest.main()
