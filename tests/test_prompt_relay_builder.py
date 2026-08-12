import tempfile
import textwrap
import unittest
from pathlib import Path

from feverslop.pipeline.prompt_relay_builder import parse_scene_dicts, parse_scene_srt


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


if __name__ == "__main__":
    unittest.main()
