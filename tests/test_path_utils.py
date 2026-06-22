import tempfile
import unittest
from pathlib import Path

from feverslop.path_utils import coerce_local_path


class PathUtilsTests(unittest.TestCase):
    def test_windows_relative_path_is_coerced_to_current_platform_path(self):
        self.assertEqual(Path("app_config.json"), coerce_local_path(".\\app_config.json"))

    def test_windows_relative_path_is_resolved_against_base_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            path = coerce_local_path("input\\song.mp3", base_dir=base_dir)

        self.assertEqual(base_dir / "input" / "song.mp3", path)

    def test_posix_relative_path_remains_valid(self):
        self.assertEqual(
            Path("workflows") / "video_ltxv_i2v_v1.json",
            coerce_local_path("workflows/video_ltxv_i2v_v1.json"),
        )

    def test_absolute_path_stays_absolute(self):
        absolute = Path.cwd().resolve() / "app_config.json"

        self.assertEqual(absolute, coerce_local_path(absolute))


if __name__ == "__main__":
    unittest.main()
