import unittest

from feverslop.domain.continuation_segments import split_semantic_action


class ContinuationSegmentTests(unittest.TestCase):
    def test_splits_long_action_with_stable_ids_and_exact_coverage(self):
        segments = split_semantic_action(
            action_id="ritual",
            start_seconds=10.0,
            duration_seconds=25.0,
            max_duration_seconds=12.0,
            fps=24,
        )

        self.assertEqual(["ritual-0001", "ritual-0002", "ritual-0003"], [item.segment_id for item in segments])
        self.assertEqual([False, True, True], [item.starts_with_anchor for item in segments])
        self.assertEqual(10.0, segments[0].start_seconds)
        self.assertEqual(35.0, segments[-1].end_seconds)
        self.assertAlmostEqual(25.0, sum(item.duration_seconds for item in segments), places=5)
        self.assertTrue(all(item.duration_seconds <= 12.0 for item in segments))

    def test_rebalances_short_tail_and_is_deterministic(self):
        kwargs = {
            "action_id": "walk",
            "start_seconds": 0.0,
            "duration_seconds": 13.0,
            "max_duration_seconds": 12.0,
            "fps": 24,
            "min_duration_seconds": 4.0,
        }

        first = split_semantic_action(**kwargs)
        second = split_semantic_action(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual([6.5, 6.5], [item.duration_seconds for item in first])

    def test_rejects_impossible_or_invalid_inputs(self):
        with self.assertRaises(ValueError):
            split_semantic_action(
                action_id="x", start_seconds=0, duration_seconds=1, max_duration_seconds=0, fps=24
            )
        with self.assertRaises(ValueError):
            split_semantic_action(
                action_id="x", start_seconds=0, duration_seconds=3, max_duration_seconds=2, fps=24, min_duration_seconds=2
            )


if __name__ == "__main__":
    unittest.main()
