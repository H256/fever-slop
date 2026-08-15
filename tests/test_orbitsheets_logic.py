import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from feverslop.application.orbitsheets_logic import (
    build_h3_character_prompt,
    build_h3_location_prompt,
    select_orbitsheet_frames,
)


class OrbitSheetsLogicTests(unittest.TestCase):
    def test_character_prompt_writes_structured_h3_shots(self):
        result = build_h3_character_prompt("a pirate captain", frames=124)
        self.assertEqual(5, result.shots)
        self.assertIn("locked-off shots joined by hard cuts", result.prompt)
        self.assertIn("left profile", result.prompt)
        self.assertIn("rear", result.prompt)

    def test_location_prompt_clamps_rotation_to_take_duration(self):
        result = build_h3_location_prompt("a ship", coverage="continuous move", rotation="full", frames=124)
        self.assertEqual(200, result.rotation_degrees)
        self.assertIn("200-degree turn", result.prompt)

    def test_fallback_selection_spreads_across_content(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for index, color in enumerate(((220, 40, 40), (40, 220, 40), (40, 40, 220), (220, 220, 40))):
                path = Path(temp) / f"frame_{index:04}.png"
                image = Image.new("RGB", (96, 64), color)
                ImageDraw.Draw(image).text((5, 5), str(index), fill="white")
                image.save(path)
                paths.append(path)

            selected = select_orbitsheet_frames(paths, count=4, subject="a ship")

            self.assertEqual(tuple(paths), selected)


if __name__ == "__main__":
    unittest.main()
