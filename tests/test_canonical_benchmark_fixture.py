from __future__ import annotations

import json
import unittest
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "canonical_benchmark" / "benchmark_plan.json"


class CanonicalBenchmarkFixtureTests(unittest.TestCase):
    def test_plan_has_ten_ordered_semantic_scenes_and_complete_timing(self):
        plan = json.loads(FIXTURE.read_text(encoding="utf-8"))
        scenes = plan["scenes"]
        self.assertEqual("feverslop.canonical-benchmark-plan/v1", plan["schema"])
        self.assertEqual(10, len(scenes))
        self.assertEqual([f"scene_{index:02d}_" for index in range(1, 11)], [scene["id"][:9] for scene in scenes])
        self.assertEqual(0.0, scenes[0]["start"])
        self.assertEqual(60.0, scenes[-1]["end"])
        for previous, current in zip(scenes, scenes[1:]):
            self.assertEqual(previous["end"], current["start"])
            self.assertTrue(current["expected_reference"])
            self.assertTrue(current["prompt"])

    def test_plan_covers_continuity_performance_and_audio_contract(self):
        plan = json.loads(FIXTURE.read_text(encoding="utf-8"))
        scenes = plan["scenes"]
        self.assertEqual({"tamsin", "soren"}, set(plan["cast"]))
        self.assertEqual({"terminal", "platform"}, set(plan["locations"]))
        self.assertIn("red_thread", plan["props"])
        self.assertTrue(any(scene["performance"] for scene in scenes))
        self.assertTrue(any(not scene["performance"] for scene in scenes))
        self.assertTrue(any(len(scene["actors"]) == 2 for scene in scenes))
        self.assertTrue(any(len(scene["actors"]) == 1 for scene in scenes))
        self.assertEqual("example-project/input/the-parts-they-left.mp3", plan["audio"]["path"])


if __name__ == "__main__":
    unittest.main()
