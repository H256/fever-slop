import json
import unittest
from pathlib import Path

from feverslop.domain.h3_two_pass import default_h3_two_pass_spec, validate_h3_two_pass_topology


class H3TwoPassWorkflowTests(unittest.TestCase):
    def test_generated_profiles_have_native_two_pass_topology(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "workflows" / "video" / "minimax_h3").glob("*_two_pass.json"))
        self.assertEqual(3, len(paths))
        for path in paths:
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text(encoding="utf-8"))
                validate_h3_two_pass_topology(workflow, default_h3_two_pass_spec("draft"))

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
                self.assertFalse(any(str(name).startswith("VRGDG_") for name in classes))
                upscaler = next(node for node in workflow.values() if node.get("class_type") == "MinimaxH3LatentUpscaler3D")
                self.assertEqual("scale by multiplier", upscaler["inputs"]["mode"])
                self.assertIn("scale", upscaler["inputs"])
                self.assertNotIn("width", upscaler["inputs"])
                self.assertNotIn("height", upscaler["inputs"])
                self.assertNotIn("megapixels", upscaler["inputs"])

    def test_generated_profiles_have_only_the_video_save_output(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "workflows" / "video" / "minimax_h3").glob("*_two_pass.json"))
        for path in paths:
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text(encoding="utf-8"))
                self.assertFalse(
                    any(node.get("class_type") == "VRAMCleanup" for node in workflow.values()),
                    "VRAMCleanup is an additional ComfyUI output node; it must not compete with #SAVE_VIDEO",
                )


if __name__ == "__main__":
    unittest.main()
