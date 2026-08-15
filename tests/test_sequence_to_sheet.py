import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageFilter, ImageDraw

from feverslop.application.sequence_to_sheet import (
    FrameSelectionConfig,
    compose_contact_sheet,
    generate_sequence_to_sheet,
    select_frames,
    recommended_view_count,
    recommended_sheet_layout,
)
from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset


def make_frame(path: Path, *, marker: int, blurred: bool = False) -> None:
    image = Image.new("RGB", (96, 72), (marker * 20 % 255, 40, 80))
    draw = ImageDraw.Draw(image)
    for x in range(0, 96, 8):
        draw.line((x, 0, x, 72), fill=(255, 255, 255), width=2)
    draw.rectangle((marker * 5 % 70, 20, marker * 5 % 70 + 18, 48), fill=(20, 200, 120))
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(8))
    image.save(path)


class SequenceToSheetTests(unittest.TestCase):
    def test_recommended_view_counts_match_character_and_location_sheets(self):
        self.assertEqual(6, recommended_view_count("character"))
        self.assertEqual(5, recommended_view_count("location"))

    def test_recommended_layout_preserves_video_aspect_ratio(self):
        self.assertEqual((2, (512, 288)), recommended_sheet_layout("character"))
        self.assertEqual((3, (512, 288)), recommended_sheet_layout("location"))

    def test_select_frames_is_deterministic_and_prefers_sharp_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index in range(8):
                path = root / f"frame_{index:04}.png"
                make_frame(path, marker=index, blurred=index in {1, 5})
                paths.append(path)

            config = FrameSelectionConfig(view_count=4)
            first = select_frames(paths, config=config)
            second = select_frames(paths, config=config)

            self.assertEqual(first, second)
            self.assertEqual(4, len(first))
            self.assertNotIn(root / "frame_0001.png", first)
            self.assertNotIn(root / "frame_0005.png", first)

    def test_select_frames_reserves_temporal_segments_for_each_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index in range(16):
                path = root / f"frame_{index:04}.png"
                make_frame(path, marker=index)
                paths.append(path)

            selected = select_frames(paths, config=FrameSelectionConfig(view_count=4))

            positions = [paths.index(path) for path in selected]
            self.assertEqual(4, len(selected))
            self.assertLessEqual(positions[0], 3)
            self.assertGreaterEqual(positions[1], 3)
            self.assertGreaterEqual(positions[2], 7)
            self.assertGreaterEqual(positions[3], 11)

    def test_compose_contact_sheet_creates_requested_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index in range(4):
                path = root / f"frame_{index:04}.png"
                make_frame(path, marker=index)
                paths.append(path)
            output = root / "sheet.png"

            result = compose_contact_sheet(
                paths,
                output,
                columns=2,
                panel_size=(64, 64),
                include_labels=False,
            )

            self.assertEqual(output, result)
            with Image.open(output) as sheet:
                self.assertEqual((128, 128), sheet.size)

    def test_generate_sequence_to_sheet_publishes_selected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sequence = root / "input.mp4"
            sequence.write_bytes(b"video")
            library = GlobalLibraryAdapter(root / "library")
            library.create(GlobalAsset("room", AssetKind.LOCATION, "Room", looks=(AssetLook("default", "Default"),)))

            def fake_extract(_video, output_dir, sample_count):
                _ = sample_count
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index, name in enumerate(("frame_0000.png", "frame_0001.png")):
                    path = output_dir / name
                    make_frame(path, marker=index)
                    paths.append(path)
                return tuple(paths)

            def fake_compose(_frame_paths, output_path, **_kwargs):
                output_path.write_bytes(b"sheet")
                return output_path

            with patch("feverslop.application.sequence_to_sheet.extract_video_frames", fake_extract), patch(
                "feverslop.application.sequence_to_sheet.compose_contact_sheet", fake_compose
            ):
                result = generate_sequence_to_sheet(
                    library,
                    kind=AssetKind.LOCATION,
                    asset_id="room",
                    look_id="default",
                    sequence_video=sequence,
                    view_count=2,
                    backend="minimax",
                )

            self.assertEqual(2, result["revision"])
            stored = library.get(AssetKind.LOCATION, "room").looks[0]
            self.assertEqual("minimax", dict(stored.metadata)["backend"])
            self.assertEqual(
                b"sheet",
                (root / "library" / "location" / "room" / "looks" / "default" / "sheet.png").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
