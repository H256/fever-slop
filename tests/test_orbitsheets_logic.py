import requests
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from feverslop.application.orbitsheets_logic import select_orbitsheet_frames
from feverslop.domain.orbitsheets_prompts import (
    build_h3_character_prompt,
    build_h3_location_prompt,
)


class _FakeVisionResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class OrbitSheetsLogicTests(unittest.TestCase):
    def test_character_prompt_writes_structured_h3_shots(self):
        result = build_h3_character_prompt("a pirate captain", frames=124)
        self.assertEqual(6, result.shots)
        self.assertIn("Six distinct locked-off views joined by hard cuts", result.prompt)
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

    def test_fallback_selects_one_sharp_frame_per_appearance_cluster(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for index, color in enumerate(((220, 40, 40), (40, 220, 40), (40, 40, 220))):
                for duplicate in range(4):
                    path = Path(temp) / f"frame_{index * 4 + duplicate:04}.png"
                    image = Image.new("RGB", (96, 64), color)
                    ImageDraw.Draw(image).rectangle((10 + duplicate, 10, 70, 50), fill="white")
                    image.save(path)
                    paths.append(path)

            selected = select_orbitsheet_frames(paths, count=3, subject="a ship")

            self.assertEqual(3, len(selected))
            self.assertEqual({paths.index(path) // 4 for path in selected}, {0, 1, 2})


class OrbitSheetsVisionJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.paths = []
        for index, color in enumerate(((220, 40, 40), (40, 220, 40), (40, 40, 220), (220, 220, 40))):
            path = Path(self.temp_dir.name) / f"frame_{index:04}.png"
            image = Image.new("RGB", (96, 64), color)
            ImageDraw.Draw(image).text((5, 5), str(index), fill="white")
            image.save(path)
            self.paths.append(path)
        self.endpoint = "http://vision.local:8000"

    def _fallback(self, count):
        return select_orbitsheet_frames(self.paths, count=count, subject="a ship")

    @patch("requests.post")
    def test_vision_picks_in_range_frames_when_available(self, mocked_post):
        mocked_post.return_value = _FakeVisionResponse('{"picks": [1, 3], "why": "ok"}')

        selected = select_orbitsheet_frames(
            self.paths, count=2, subject="a ship", vision_endpoint=self.endpoint
        )

        self.assertEqual((self.paths[0], self.paths[2]), selected)
        mocked_post.assert_called_once()
        self.assertTrue(mocked_post.call_args.args[0].endswith("/v1/chat/completions"))

    @patch("requests.post")
    def test_vision_malformed_content_falls_back_to_deterministic_selection(self, mocked_post):
        for content in ("no json here", '{"why": "no picks key"}'):
            with self.subTest(content=content):
                mocked_post.return_value = _FakeVisionResponse(content)

                selected = select_orbitsheet_frames(
                    self.paths, count=2, subject="a ship", vision_endpoint=self.endpoint
                )

                self.assertEqual(self._fallback(2), selected)

    @patch("requests.post")
    def test_vision_out_of_range_picks_are_filtered_before_acceptance(self, mocked_post):
        mocked_post.return_value = _FakeVisionResponse('{"picks": [1, 2, 5, 9]}')

        selected = select_orbitsheet_frames(
            self.paths, count=2, subject="a ship", vision_endpoint=self.endpoint
        )

        self.assertEqual((self.paths[0], self.paths[1]), selected)

    @patch("requests.post")
    def test_vision_insufficient_valid_picks_fall_back_to_deterministic_selection(self, mocked_post):
        mocked_post.return_value = _FakeVisionResponse('{"picks": [1, 5, 9]}')

        selected = select_orbitsheet_frames(
            self.paths, count=2, subject="a ship", vision_endpoint=self.endpoint
        )

        self.assertEqual(self._fallback(2), selected)

    @patch("requests.post")
    def test_vision_endpoint_error_falls_back_to_deterministic_selection(self, mocked_post):
        mocked_post.side_effect = requests.exceptions.ConnectionError("probe outage")

        selected = select_orbitsheet_frames(
            self.paths, count=2, subject="a ship", vision_endpoint=self.endpoint
        )

        self.assertEqual(self._fallback(2), selected)


if __name__ == "__main__":
    unittest.main()
