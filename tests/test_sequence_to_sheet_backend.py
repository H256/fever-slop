import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.sequence_to_sheet_backend import ComfyUISequenceToSheetBackend


class FakeUploader:
    def resolve_reference_image_name(self, path):
        return f"uploaded/{Path(path).name}"


class SequenceToSheetBackendTests(unittest.TestCase):
    def test_builds_ltx_workflow_from_semantic_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            workflow = Path("workflows/sequence_to_sheet_ltx_v1.json")
            anchor = Path(temp) / "anchor.png"
            anchor.write_bytes(b"anchor")
            backend = ComfyUISequenceToSheetBackend(
                client=object(), workflow_path=workflow, backend="ltx", asset_uploader=FakeUploader()
            )

            patched = backend.build_workflow(
                anchor_images=[anchor], prompt="turnaround", seed=42, width=640, height=384, frames=65
            )

            self.assertEqual("uploaded/anchor.png", patched["14"]["inputs"]["image"])
            self.assertEqual(640, patched["10"]["inputs"]["value"])
            self.assertEqual(65, patched["22"]["inputs"]["value"])
            self.assertEqual("turnaround", patched["27"]["inputs"]["text"])
            self.assertEqual(42, patched["37"]["inputs"]["noise_seed"])

    def test_builds_minimax_workflow_and_clears_unused_reference_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor = Path(temp) / "anchor.png"
            anchor.write_bytes(b"anchor")
            backend = ComfyUISequenceToSheetBackend(
                client=object(),
                workflow_path=Path("workflows/sequence_to_sheet_minimax_h3_v1.json"),
                backend="minimax",
                asset_uploader=FakeUploader(),
            )

            patched = backend.build_workflow(anchor_images=[anchor], prompt="turnaround", seed=7)

            refs = {
                node["_meta"]["title"]: node["inputs"]["image"]
                for node in patched.values()
                if node.get("_meta", {}).get("title", "").startswith("#REF_")
            }
            self.assertEqual("uploaded/anchor.png", refs["#REF_1"])
            self.assertEqual("", refs["#REF_2"])
            self.assertEqual(7, patched["129"]["inputs"]["noise_seed"])


if __name__ == "__main__":
    unittest.main()
