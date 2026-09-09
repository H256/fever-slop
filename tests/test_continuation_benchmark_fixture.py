import json
import unittest
from pathlib import Path

from feverslop.composition.continuation_scheduler import ContinuationScheduler
from feverslop.domain.continuation_segments import split_semantic_action
from feverslop.domain.duration_capability import DurationCapability


FIXTURE = Path(__file__).parent / "fixtures" / "continuation" / "continuous_action_24s.json"


class ContinuationBenchmarkFixtureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixture_matches_deterministic_split_and_exact_frame_accounting(self):
        capability = DurationCapability.create(
            fps=self.fixture["fps"],
            **self.fixture["duration_capability"],
        )
        interval = self.fixture["audio_interval"]
        segments = split_semantic_action(
            action_id="lantern-walk",
            start_seconds=interval["start_seconds"],
            duration_seconds=interval["end_seconds"] - interval["start_seconds"],
            max_duration_seconds=99,
            fps=self.fixture["fps"],
            capability=capability,
        )

        expected = self.fixture["segments"]
        self.assertEqual([item["segment_id"] for item in expected], [item.segment_id for item in segments])
        self.assertEqual([item["render_frame_count"] for item in expected], [item.render_frame_count for item in segments])
        self.assertEqual([item["anchor_frames"] for item in expected], [item.anchor_frames for item in segments])
        self.assertEqual(576, sum(round(item.duration_seconds * item.fps) for item in segments))
        self.assertEqual(self.fixture["expected_timeline_frames"], sum(round(item.duration_seconds * item.fps) for item in segments))
        self.assertEqual(20.0, segments[1].start_seconds)
        self.assertEqual(28.0, segments[2].start_seconds)

    def test_interruption_fixture_resumes_from_failed_boundary(self):
        chains = {self.fixture["group_id"]: [item["segment_id"] for item in self.fixture["segments"]]}
        first = []
        scheduler = ContinuationScheduler(chains)
        completed = scheduler.run(lambda segment: first.append(segment) or segment != "lantern-walk-0002")
        self.assertEqual(("lantern-walk-0001",), completed)
        self.assertEqual(["lantern-walk-0001", "lantern-walk-0002"], first)

        resumed = []
        resume_scheduler = ContinuationScheduler({"resume": self.fixture["resume_fixture"]["resume_chain"]})
        self.assertEqual(
            ("lantern-walk-0002", "lantern-walk-0003"),
            resume_scheduler.run(lambda segment: resumed.append(segment) or True),
        )
        self.assertEqual(self.fixture["resume_fixture"]["resume_chain"], resumed)


if __name__ == "__main__":
    unittest.main()
