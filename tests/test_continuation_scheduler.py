import unittest

from feverslop.composition.continuation_scheduler import (
    ContinuationScheduler,
    chains_from_predecessors,
)


class RecordingReporter:
    def __init__(self):
        self.messages = []

    def message(self, text):
        self.messages.append(text)


class ContinuationSchedulerTests(unittest.TestCase):
    def test_builds_serial_chains_and_singletons_from_scene_predecessors(self):
        self.assertEqual(
            {
                "scene-1": ("scene-1", "scene-2", "scene-3"),
                "scene-4": ("scene-4",),
            },
            chains_from_predecessors(
                [1, 2, 3, 4],
                {2: 1, 3: 2},
            ),
        )

    def test_runs_ready_independent_segments_and_unlocks_successors(self):
        reporter = RecordingReporter()
        rendered = []
        scheduler = ContinuationScheduler(
            {"action": ["a-0001", "a-0002"], "other": ["b-0001"]},
            reporter=reporter,
        )

        completed = scheduler.run(lambda segment_id: rendered.append(segment_id) or True)

        self.assertEqual(("a-0001", "a-0002", "b-0001"), completed)
        self.assertEqual(list(completed), rendered)
        self.assertTrue(any("boundary" in message.lower() for message in reporter.messages))

    def test_stops_before_successor_when_boundary_is_invalid(self):
        rendered = []
        scheduler = ContinuationScheduler({"action": ["a-0001", "a-0002"]})

        completed = scheduler.run(lambda segment_id: rendered.append(segment_id) or False)

        self.assertEqual((), completed)
        self.assertEqual(["a-0001"], rendered)

    def test_invalid_chain_does_not_block_independent_chain(self):
        rendered = []
        scheduler = ContinuationScheduler({"action": ["a-0001", "a-0002"], "other": ["b-0001"]})

        def render(segment_id):
            rendered.append(segment_id)
            return segment_id == "b-0001"

        completed = scheduler.run(
            render,
        )

        self.assertEqual(("b-0001",), completed)
        self.assertEqual(["a-0001", "b-0001"], rendered)


if __name__ == "__main__":
    unittest.main()
