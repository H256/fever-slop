import json
import tempfile
import unittest
from pathlib import Path

from project_config import ProjectConfig


class ProjectConfigTests(unittest.TestCase):
    def test_loads_zimage_and_ltx_steering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "input_audio": "song.mp3",
                        "steering": {
                            "zimage": "z-image steering",
                            "ltx": "ltx steering",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("z-image steering", config.steering.zimage)
            self.assertEqual("ltx steering", config.steering.ltx)


if __name__ == "__main__":
    unittest.main()
