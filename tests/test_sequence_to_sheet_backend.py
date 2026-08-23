import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.sequence_to_sheet_backend import ComfyUISequenceToSheetBackend


class FakeUploader:
    def resolve_reference_image_name(self, path):
        return f"uploaded/{Path(path).name}"


class FakeSchemaClient:
    def get_object_info(self):
        return {
            "ResolutionSelector": {
                "input": {
                    "required": {
                        "aspect_ratio": [[
                            "1:1 (Square)",
                            "9:16 (Vertical)",
                            "16:9 (Wide)",
                        ]],
                    },
                },
            },
        }


class SequenceToSheetBackendTests(unittest.TestCase):
    def test_builds_neutral_character_prompt_with_hard_cut_views(self):
        backend = ComfyUISequenceToSheetBackend(
            client=object(),
            workflow_path=Path("workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"),
            backend="minimax",
        )

        prompt = backend.build_sheet_prompt(
            "a scarred pirate captain in a blue coat", kind="character", frames=124,
        )

        self.assertEqual(6, prompt.shots)
        self.assertEqual(124, prompt.frames)
        self.assertIn("hard cuts", prompt.prompt)
        self.assertIn("tight close-up of the face", prompt.prompt)
        self.assertIn("expression now frightened", prompt.prompt)
        self.assertIn("<Picture 1>", prompt.prompt)
        self.assertIn("rear", prompt.prompt)

    def test_builds_neutral_location_prompt_with_continuous_move(self):
        backend = ComfyUISequenceToSheetBackend(
            client=object(),
            workflow_path=Path("workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"),
            backend="minimax",
        )

        prompt = backend.build_sheet_prompt(
            "a weathered pirate ship at dusk",
            kind="location",
            coverage="continuous move",
            rotation="half",
            frames=124,
        )

        self.assertEqual(0, prompt.shots)
        self.assertEqual(124, prompt.frames)
        self.assertEqual(180, prompt.rotation_degrees)
        self.assertIn("continuous", prompt.prompt)

    def test_builds_minimax_i2va_workflow_with_first_frame_and_turbo_lora(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor = Path(temp) / "anchor.png"
            anchor.write_bytes(b"anchor")
            backend = ComfyUISequenceToSheetBackend(
                client=object(),
                workflow_path=Path("workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"),
                backend="minimax",
                asset_uploader=FakeUploader(),
            )

            patched = backend.build_workflow(anchor_images=[anchor], prompt="turnaround", seed=7)

            self.assertEqual("uploaded/anchor.png", patched["136"]["inputs"]["image"])
            self.assertEqual(
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                patched["139"]["inputs"]["lora_name"],
            )
            self.assertEqual("turnaround", patched["131"]["inputs"]["prompt"])
            self.assertEqual(7, patched["129"]["inputs"]["noise_seed"])

    def test_preserves_two_decimal_megapixel_target(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor = Path(temp) / "anchor.png"
            anchor.write_bytes(b"anchor")
            backend = ComfyUISequenceToSheetBackend(
                client=object(),
                workflow_path=Path("workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"),
                backend="minimax",
                asset_uploader=FakeUploader(),
            )

            patched = backend.build_workflow(
                anchor_images=[anchor], prompt="turnaround", seed=7, width=850, height=1000,
            )

            self.assertEqual(0.85, patched["115"]["inputs"]["megapixels"])

    def test_patches_portrait_aspect_ratio_for_character_sequences(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor = Path(temp) / "anchor.png"
            anchor.write_bytes(b"anchor")
            backend = ComfyUISequenceToSheetBackend(
                client=object(),
                workflow_path=Path("workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"),
                backend="minimax",
                asset_uploader=FakeUploader(),
            )

            patched = backend.build_workflow(
                anchor_images=[anchor],
                prompt="portrait turnaround",
                seed=7,
                aspect_ratio="9:16 (Portrait Widescreen)",
            )

            self.assertEqual("9:16 (Portrait Widescreen)", patched["115"]["inputs"]["aspect_ratio"])

    def test_resolves_semantic_portrait_ratio_from_current_comfyui_enum(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor = Path(temp) / "anchor.png"
            anchor.write_bytes(b"anchor")
            backend = ComfyUISequenceToSheetBackend(
                client=FakeSchemaClient(),
                workflow_path=Path("workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"),
                backend="minimax",
                asset_uploader=FakeUploader(),
            )

            patched = backend.build_workflow(
                anchor_images=[anchor],
                prompt="portrait turnaround",
                seed=7,
                aspect_ratio="portrait",
            )

            self.assertEqual("9:16 (Vertical)", patched["115"]["inputs"]["aspect_ratio"])


if __name__ == "__main__":
    unittest.main()
