from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PROJECT = REPOSITORY_ROOT / "example-project"


class ExampleProjectFixtureTests(unittest.TestCase):
    def test_project_uses_the_fixed_song_and_complete_lyrics(self):
        self.assertTrue((EXAMPLE_PROJECT / "config.json").is_file())
        config = json.loads(
            (EXAMPLE_PROJECT / "config.json").read_text(encoding="utf-8"),
        )

        self.assertEqual("feverslop-example-project", config["project_name"])
        self.assertEqual("single", config["subject_mode"])
        self.assertEqual(1, config["max_scene_actors"])
        self.assertEqual("protagonist", config["actors"][0]["id"])
        self.assertIn("adult man", config["actors"][0]["visual_description"])
        self.assertIn("male vocalist", config["actors"][0]["role"])
        self.assertEqual("input/the-parts-they-left.mp3", config["input_audio"])
        self.assertGreaterEqual(len(config["lyrics"]), 200)
        self.assertGreaterEqual(len(config["lyrics"].splitlines()), 10)
        self.assertIn("music_style", config)
        self.assertIn("symphonic power metal", config["music_style"])
        self.assertIn("operatic choir", config["music_style"])
        self.assertIn("dark grandeur", config["music_style"])
        self.assertTrue((EXAMPLE_PROJECT / "input" / "README.md").is_file())
        self.assertTrue((EXAMPLE_PROJECT / config["input_audio"]).is_file())

    def test_starter_project_excludes_benchmark_metadata(self):
        for filename in (
            "asset-provenance.json",
            "benchmark.json",
            "benchmark-plan.json",
        ):
            self.assertFalse((EXAMPLE_PROJECT / filename).exists())

    def test_generated_project_state_is_ignored_but_song_is_trackable(self):
        self.assertTrue((EXAMPLE_PROJECT / ".gitignore").is_file())
        ignore_lines = {
            line.strip()
            for line in (EXAMPLE_PROJECT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("output/", ignore_lines)
        self.assertIn(".studio/", ignore_lines)
        self.assertNotIn("input/", ignore_lines)
        self.assertNotIn("*.mp3", ignore_lines)


if __name__ == "__main__":
    unittest.main()
