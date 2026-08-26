from __future__ import annotations

import hashlib
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

        self.assertEqual("feverslop-quality-benchmark", config["project_name"])
        self.assertEqual("input/the-parts-they-left.mp3", config["input_audio"])
        self.assertGreaterEqual(len(config["lyrics"]), 200)
        self.assertGreaterEqual(len(config["lyrics"].splitlines()), 10)
        self.assertIn("music_style", config)
        self.assertIn("symphonic power metal", config["music_style"])
        self.assertIn("operatic choir", config["music_style"])
        self.assertIn("dark grandeur", config["music_style"])
        self.assertTrue((EXAMPLE_PROJECT / "input" / "README.md").is_file())
        self.assertTrue((EXAMPLE_PROJECT / config["input_audio"]).is_file())

    def test_pending_asset_provenance_is_explicit(self):
        self.assertTrue((EXAMPLE_PROJECT / "asset-provenance.json").is_file())
        provenance = json.loads(
            (EXAMPLE_PROJECT / "asset-provenance.json").read_text(encoding="utf-8"),
        )

        self.assertEqual("pending_rights", provenance["status"])
        self.assertEqual("input/the-parts-they-left.mp3", provenance["asset"])
        self.assertEqual(
            "e2bbfb8cb25ae859421ddbf115c2aaadeb922e7256d103f59c7e3400a574fbda",
            provenance["sha256"],
        )
        self.assertAlmostEqual(72.400167, provenance["duration_seconds"], places=6)
        self.assertEqual(48_000, provenance["sample_rate_hz"])
        self.assertEqual(2, provenance["channels"])
        audio = EXAMPLE_PROJECT / provenance["asset"]
        self.assertEqual(provenance["sha256"], hashlib.sha256(audio.read_bytes()).hexdigest())

    def test_music_profile_covers_the_benchmark_characteristics(self):
        path = EXAMPLE_PROJECT / "benchmark.json"
        self.assertTrue(path.is_file())
        benchmark = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(1, benchmark["schema_version"])
        self.assertEqual(168, benchmark["music_profile"]["bpm"])
        self.assertEqual("half-time", benchmark["music_profile"]["pulse"])
        self.assertIn("male lead vocals", benchmark["music_profile"]["vocals"])
        self.assertIn("operatic choir", benchmark["music_profile"]["vocals"])
        self.assertIn("electric violin", benchmark["music_profile"]["instrumentation"])
        self.assertEqual({"minimum": 8, "maximum": 12}, benchmark["scene_target_count"])

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
