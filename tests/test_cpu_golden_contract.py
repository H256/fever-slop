import json
import unittest
from pathlib import Path

from feverslop.domain.continuation_dependencies import ContinuationDependencyGraph


class CpuGoldenContractTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    EXAMPLE = ROOT / "example_movie_project"

    def test_example_project_locks_draft_two_pass_profile_and_six_shots(self):
        config = json.loads((self.EXAMPLE / "config.json").read_text(encoding="utf-8"))
        render_plan = json.loads(
            (self.EXAMPLE / "movie" / "render_plan.json").read_text(encoding="utf-8")
        )

        self.assertEqual("minimax-h3-r2v", config["video_pipeline"])
        self.assertEqual(
            {"quality": "draft", "pass_strategy": "two_pass", "postprocess": "none"},
            config["render_profile"],
        )
        self.assertEqual(60.0, render_plan["duration_seconds"])
        self.assertEqual(
            [f"shot_{number:04d}" for number in range(1, 7)],
            [shot["shot_id"] for shot in render_plan["shots"]],
        )

    def test_continuation_graph_is_deterministic_and_suffix_safe(self):
        graph = ContinuationDependencyGraph.from_chains(
            {"main": ["shot_0001", "shot_0002", "shot_0003"], "alt": ["shot_0004"]}
        )

        self.assertEqual(("shot_0001", "shot_0004"), graph.ready())
        graph.mark_complete("shot_0001", anchor_valid=True)
        self.assertEqual(("shot_0002", "shot_0004"), graph.ready())
        self.assertEqual(
            ("shot_0002", "shot_0003"), graph.invalidate_suffix("shot_0002")
        )


if __name__ == "__main__":
    unittest.main()
