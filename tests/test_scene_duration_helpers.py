import os
import tempfile
import unittest

from feverslop.domain.srt import SrtScene
from feverslop.pipeline.scene_duration_enforcer import (
    merge_remaining_short_scenes,
    merge_short_scenes,
    renumber_scenes,
    split_long_scenes,
)


class SceneDurationHelperTests(unittest.TestCase):
    def test_split_long_scenes_returns_legal_chunks(self):
        result = split_long_scenes([SrtScene(scene=7, start=0.0, end=9.0, text="A")], max_duration=4.0)

        self.assertEqual([7, 7, 7], [scene.scene for scene in result])
        self.assertTrue(all(scene.duration <= 4.0 for scene in result))
        self.assertEqual(0.0, result[0].start)
        self.assertEqual(9.0, result[-1].end)

    def test_merge_short_scenes_merges_tail_into_previous_scene(self):
        scenes = [
            SrtScene(scene=1, start=0.0, end=2.0, text="A"),
            SrtScene(scene=2, start=2.0, end=2.5, text="B"),
        ]

        result = merge_short_scenes(scenes, min_duration=1.0, max_duration=4.0)

        self.assertEqual(1, len(result))
        self.assertEqual(2.5, result[0].end)

    def test_renumber_scenes_rewrites_scene_numbers_and_empty_text(self):
        result = renumber_scenes([SrtScene(scene=99, start=0, end=1, text="")])

        self.assertEqual(1, result[0].scene)
        self.assertEqual("Scene 1", result[0].text)


class MergeRemainingShortScenesTests(unittest.TestCase):
    def test_merge_remaining_short_scenes_terminates_when_max_below_twice_min(self):
        """Terminates when min_duration=2, max_duration=3 and scenes are 1.5s each."""
        scenes = [
            SrtScene(scene=1, start=0.0, end=1.5, text="A"),
            SrtScene(scene=2, start=1.5, end=3.0, text="B"),
            SrtScene(scene=3, start=3.0, end=4.5, text="C"),
            SrtScene(scene=4, start=4.5, end=6.0, text="D"),
        ]
        # This should terminate quickly and return a result without hanging
        result = merge_remaining_short_scenes(
            scenes, min_duration=2.0, max_duration=3.0
        )
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)

    def test_merge_remaining_short_scenes_normal_still_converges(self):
        """Normal inputs where merging fixes short scenes still work."""
        scenes = [
            SrtScene(scene=1, start=0.0, end=3.0, text="A"),
            SrtScene(scene=2, start=3.0, end=3.5, text="B"),
            SrtScene(scene=3, start=3.5, end=6.0, text="C"),
        ]
        result = merge_remaining_short_scenes(
            scenes, min_duration=2.0, max_duration=4.0
        )
        self.assertTrue(all(s.duration >= 2.0 for s in result[:-1]))


class SrtDomainTests(unittest.TestCase):
    def test_parse_srt_timestamp_basic(self):
        from feverslop.domain.srt import parse_srt_timestamp
        self.assertAlmostEqual(0.0, parse_srt_timestamp("00:00:00,000"))
        self.assertAlmostEqual(61.5, parse_srt_timestamp("00:01:01,500"))

    def test_format_srt_timestamp_roundtrip(self):
        from feverslop.domain.srt import parse_srt_timestamp, format_srt_timestamp
        original = "01:23:45,678"
        seconds = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(seconds)
        self.assertEqual(original, formatted)

    def test_parse_srt_blocks_empty_file(self):
        from feverslop.domain.srt import parse_srt_blocks
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        result = parse_srt_blocks(path)
        os.unlink(path)
        self.assertEqual([], result)

    def test_parse_srt_blocks_valid_content(self):
        from feverslop.domain.srt import parse_srt_blocks
        content = "1\n00:00:00,000 --> 00:00:02,500\nHello\n\n2\n00:00:02,500 --> 00:00:05,000\nWorld"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        result = parse_srt_blocks(path)
        os.unlink(path)
        self.assertEqual(2, len(result))
        self.assertEqual(1, result[0].index)
        self.assertAlmostEqual(0.0, result[0].start)
        self.assertEqual("Hello", result[0].text)


if __name__ == "__main__":
    unittest.main()
