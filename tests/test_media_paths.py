import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feverslop.utils.media_paths import safe_file_stem, write_concat_list


class MediaPathTests(unittest.TestCase):
    def test_safe_file_stem_preserves_legacy_convention(self):
        self.assertEqual("La_Entity_01", safe_file_stem(" La Entity 01! ", "fallback"))
        self.assertEqual("fallback", safe_file_stem("!!!", "fallback"))
        self.assertEqual("fallback", safe_file_stem(None, "fallback"))
        self.assertEqual("unsafe_fallback", safe_file_stem(None, " unsafe fallback! "))

    def test_write_concat_list_creates_parent_and_escapes_quotes(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "concat.txt"
            clip = Path("relative") / "artist's clip.mp4"

            result = write_concat_list([clip], output)

            self.assertEqual(output, result)
            escaped = clip.resolve().as_posix().replace("'", r"'\''")
            self.assertEqual(f"file '{escaped}'\n", output.read_text(encoding="utf-8"))
