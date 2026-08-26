from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PROJECT = REPOSITORY_ROOT / "example-project"


class ExampleProjectFixtureTests(unittest.TestCase):
    def test_skeleton_is_ready_for_the_user_supplied_song_and_lyrics(self):
        self.assertTrue((EXAMPLE_PROJECT / "config.json").is_file())
        config = json.loads(
            (EXAMPLE_PROJECT / "config.json").read_text(encoding="utf-8"),
        )

        self.assertEqual("feverslop-quality-benchmark", config["project_name"])
        self.assertEqual("input/song.mp3", config["input_audio"])
        self.assertEqual("", config["lyrics"])
        self.assertTrue((EXAMPLE_PROJECT / "input" / "README.md").is_file())
        self.assertFalse((EXAMPLE_PROJECT / "input" / "song.mp3").exists())

    def test_pending_asset_provenance_is_explicit(self):
        self.assertTrue((EXAMPLE_PROJECT / "asset-provenance.json").is_file())
        provenance = json.loads(
            (EXAMPLE_PROJECT / "asset-provenance.json").read_text(encoding="utf-8"),
        )

        self.assertEqual("pending", provenance["status"])
        self.assertEqual("input/song.mp3", provenance["asset"])
        self.assertIsNone(provenance["sha256"])

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
