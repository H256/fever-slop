import tempfile
import textwrap
import unittest
from pathlib import Path

from feverslop.pipeline.prompt_relay_builder import (
    lyrics_for_time_range,
    parse_scene_dicts,
    parse_scene_srt,
)

FAKE_SRT = textwrap.dedent("""\
    1
    00:00:00,000 --> 00:00:02,000
    Scene 1 text

    2
    00:00:02,000 --> 00:00:04,000
    Scene 2 text
""")


class TestParseSceneDicts(unittest.TestCase):
    def test_parse_scene_dicts_returns_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            srt_path = Path(temp_dir) / "test.srt"
            srt_path.write_text(FAKE_SRT)
            result = parse_scene_dicts(srt_path)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["scene"], 1)
            self.assertEqual(result[0]["start"], 0.0)
            self.assertEqual(result[0]["end"], 2.0)
            self.assertEqual(result[0]["label"], "Scene 1 text")

    def test_parse_scene_srt_alias_emits_deprecation_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            srt_path = Path(temp_dir) / "test.srt"
            srt_path.write_text(FAKE_SRT)
            with self.assertWarns(FutureWarning) as cm:
                parse_scene_srt(srt_path)
            self.assertIn("parse_scene_dicts", str(cm.warning))


class TestLyricsForTimeRangeFallback(unittest.TestCase):
    def test_falls_back_to_proportional_split_when_no_word_falls_in_window(self):
        # A vocal line spans [0, 4) but its Whisper word midpoints cluster at the
        # start and end, leaving a sub-window with no words. The proportional
        # split must keep the window's share instead of returning "".
        result = lyrics_for_time_range(
            "Ich trug mein Name wie ein Messer",
            0.0,
            4.0,
            1.5,
            2.5,
            (
                {"word": "Ich", "start": 0.0, "end": 0.3},
                {"word": "trug", "start": 0.3, "end": 0.6},
                {"word": "mein", "start": 0.6, "end": 0.9},
                {"word": "Messer", "start": 3.0, "end": 3.9},
            ),
        )

        self.assertEqual("Name", result)

    def test_prefers_timestamped_words_when_any_fall_in_window(self):
        result = lyrics_for_time_range(
            "Ich trug mein Name wie ein Messer",
            0.0,
            4.0,
            1.5,
            2.5,
            (
                {"word": "Ich", "start": 0.0, "end": 0.3},
                {"word": "mein", "start": 1.5, "end": 2.0},
                {"word": "Name", "start": 2.0, "end": 2.4},
                {"word": "Messer", "start": 3.0, "end": 3.9},
            ),
        )

        self.assertEqual("mein Name", result)


if __name__ == "__main__":
    unittest.main()
