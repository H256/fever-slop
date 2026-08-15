import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageFilter, ImageDraw

from feverslop.application.sequence_to_sheet import (
    FrameSelectionConfig,
    compose_contact_sheet,
    select_frames,
)


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


if __name__ == "__main__":
    unittest.main()
