import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from feverslop.application.sequence_reference_pipeline import (
    SequenceReferencePipeline,
    SequenceReferenceRequest,
)
from feverslop.tools.reference_bible import SEQUENCE_PHASE_LABELS


class FakeAnchorBackend:
    def __init__(self):
        self.calls = []

    def render_image(self, request):
        self.calls.append(request)
        target = Path(request.output_dir) / "anchor.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 72), "white").save(target)
        return target


class FakeSequenceBackend:
    def __init__(self):
        self.calls = []

    def build_sheet_prompt(self, description, *, kind, shots, frames):
        self.calls.append(("prompt", description, kind, shots, frames))
        return type("Prompt", (), {"prompt": f"{kind}: {description}"})()

    def render(self, *, anchor_images, prompt, output_path, seed, **kwargs):
        self.calls.append(("render", tuple(anchor_images), prompt, seed, kwargs))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"sequence")
        return output


class SequenceReferencePipelineTests(unittest.TestCase):
    def test_character_anchor_uses_identity_plan_instead_of_action_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor_backend = FakeAnchorBackend()
            sequence_backend = FakeSequenceBackend()

            class Planner:
                source = "test"
                fallback_reason = None

                def plan(self, **_kwargs):
                    from feverslop.domain.reference_sheet import ReferenceSheetPlan

                    return ReferenceSheetPlan(
                        kind="character",
                        anchor_description=(
                            "A woman with long silver hair, pale skin, and dark leather armor"
                        ),
                    )

            pipeline = SequenceReferencePipeline(
                anchor_backend=anchor_backend,
                sequence_backend=sequence_backend,
                planner=Planner(),
            )
            request = SequenceReferenceRequest(
                kind="character",
                asset_id="singer",
                name="Singer",
                description="A woman with long silver hair singing intensely.",
                image_prompt=(
                    "A woman with long silver hair singing passionately on a black stone altar."
                ),
                visual_style="dark gothic cinematic fantasy",
                output_dir=Path(temp),
            )

            def fake_extract(_video, output_dir, sample_count):
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(sample_count):
                    path = output_dir / f"frame_{index:04}.png"
                    Image.new("RGB", (64, 36), "white").save(path)
                    paths.append(path)
                return tuple(paths)

            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                fake_extract,
            ), patch(
                "feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames",
                lambda paths, **_: tuple(paths[:6]),
            ):
                result = pipeline.generate(request)

            prompt = anchor_backend.calls[0].prompt
            self.assertIn("A woman with long silver hair, pale skin, and dark leather armor", prompt)
            self.assertIn("dark gothic cinematic fantasy", prompt)
            self.assertNotIn("sing", prompt.lower())
            self.assertNotIn("altar", prompt.lower())
            self.assertEqual(prompt, result.anchor_prompt)
            render_call = next(call for call in sequence_backend.calls if call[0] == "render")
            self.assertNotIn("anchor_prompt", render_call[4])

    def test_reference_phase_log_labels_are_english(self):
        self.assertEqual(
            {
                "anchor_start": "Krea anchor started",
                "anchor_complete": "Krea anchor finished",
                "sequence_start": "MiniMax sequence started",
                "sequence_complete": "MiniMax sequence finished",
            },
            SEQUENCE_PHASE_LABELS,
        )

    def test_generates_anchor_sequence_contact_and_final_sheet(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor_backend = FakeAnchorBackend()
            sequence_backend = FakeSequenceBackend()
            events = []
            pipeline = SequenceReferencePipeline(
                anchor_backend=anchor_backend,
                sequence_backend=sequence_backend,
                on_phase=events.append,
            )

            def fake_extract(_video, output_dir, sample_count):
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(sample_count):
                    path = output_dir / f"frame_{index:04}.png"
                    Image.new("RGB", (64, 36), (index * 10 % 255, 20, 40)).save(path)
                    paths.append(path)
                return tuple(paths)

            request = SequenceReferenceRequest(
                kind="character",
                asset_id="astronaut",
                name="Astronaut",
                description="a solitary astronaut",
                output_dir=Path(temp) / "references",
                seed=7,
                visual_style="comic, bright colors, anime style",
            )
            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                fake_extract,
            ), patch(
                "feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames",
                lambda paths, **_: tuple(paths[:6]),
            ):
                result = pipeline.generate(request)

            self.assertEqual(6, result.selected_frames)
            self.assertTrue(Path(result.anchor_path).is_file())
            self.assertTrue(Path(result.sequence_path).is_file())
            self.assertTrue(Path(result.contact_sheet_path).is_file())
            self.assertTrue(Path(result.sheet_path).is_file())
            self.assertEqual(1, len(anchor_backend.calls))
            self.assertEqual(1, len([call for call in sequence_backend.calls if call[0] == "render"]))
            self.assertIn("comic, bright colors, anime style", anchor_backend.calls[0].prompt)
            self.assertIn("comic, bright colors, anime style", sequence_backend.calls[1][2])
            self.assertEqual(
                ["anchor_start", "anchor_complete", "sequence_start", "sequence_complete"],
                [event["phase"] for event in events],
            )

    def test_location_uses_five_views(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = SequenceReferencePipeline(
                anchor_backend=FakeAnchorBackend(),
                sequence_backend=FakeSequenceBackend(),
            )
            request = SequenceReferenceRequest(
                kind="location",
                asset_id="ship",
                name="Ship",
                description="a weathered ship",
                output_dir=Path(temp),
            )
            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                lambda _video, _output_dir, sample_count: (),
            ):
                with self.assertRaises(ValueError):
                    pipeline.generate(request)


if __name__ == "__main__":
    unittest.main()
