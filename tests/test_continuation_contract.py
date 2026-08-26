import unittest

from feverslop.domain.continuation_contract import ContinuationGroup
from feverslop.domain.continuation_segments import split_semantic_action


class ContinuationContractTests(unittest.TestCase):
    def test_validates_ordered_segments_and_predecessors(self):
        segments = split_semantic_action(
            action_id="ritual", start_seconds=0, duration_seconds=20,
            max_duration_seconds=12, fps=24,
        )
        group = ContinuationGroup.create(
            group_id="ritual-group", semantic_action="raise the lantern",
            semantic_start_seconds=0, semantic_end_seconds=20, segments=segments,
        )
        self.assertEqual(None, group.predecessor("ritual-0001"))
        self.assertEqual("ritual-0001", group.predecessor("ritual-0002"))

    def test_rejects_gap_and_overlap(self):
        segments = list(split_semantic_action(
            action_id="walk", start_seconds=0, duration_seconds=13,
            max_duration_seconds=12, fps=24,
        ))
        segments[1] = type(segments[1])(
            segment_id=segments[1].segment_id, index=segments[1].index,
            start_seconds=7, end_seconds=13, duration_seconds=6, starts_with_anchor=True,
        )
        with self.assertRaises(ValueError):
            ContinuationGroup.create(
                group_id="walk", semantic_action="walk", semantic_start_seconds=0,
                semantic_end_seconds=13, segments=segments,
            )
