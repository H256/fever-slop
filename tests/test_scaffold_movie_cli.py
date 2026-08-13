import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestScaffoldMovieCli(unittest.TestCase):
    def test_scaffold_movie_cli_persists_cli_settings_in_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scaffold_movie.py",
                    "--name",
                    "CLI Regression Movie",
                    "--story-text",
                    "A person enters an empty room.",
                    "--desired-length",
                    "2",
                    "--min-duration",
                    "1",
                    "--max-duration",
                    "1",
                    "--width",
                    "640",
                    "--height",
                    "960",
                    "--planner-backend",
                    "deterministic",
                    "--projects-root",
                    temp_dir,
                ],
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(
                (Path(temp_dir) / "cli-regression-movie" / "movie" / "render_plan.json").exists()
            )
            config = json.loads(
                (Path(temp_dir) / "cli-regression-movie" / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual({"width": 640, "height": 960, "fps": 24}, config["video"])
            self.assertEqual(
                {"min_duration": 1.0, "max_duration": 1.0},
                {
                    key: config["scene_generation"][key]
                    for key in ("min_duration", "max_duration")
                },
            )


if __name__ == "__main__":
    unittest.main()
