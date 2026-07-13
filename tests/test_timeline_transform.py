from __future__ import annotations

import unittest

from feverslop.domain.timeline import TimelineSegment
from feverslop.domain.timeline_transform import merge_same_kind_segments, normalize_empty_vocals


class NormalizeEmptyVocalsTest(unittest.TestCase):
    def test_converts_short_vocals_to_instrumental(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="ab"),
        ]
        result = normalize_empty_vocals(segments, min_text_chars=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "instrumental")
        self.assertEqual(result[0].text, "")

    def test_keeps_sufficient_vocals(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="hello"),
        ]
        result = normalize_empty_vocals(segments, min_text_chars=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "vocals")
        self.assertEqual(result[0].text, "hello")

    def test_preserves_instrumental(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="instrumental"),
        ]
        result = normalize_empty_vocals(segments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "instrumental")

    def test_empty_input(self):
        result = normalize_empty_vocals([])
        self.assertEqual(result, [])

    def test_mixed_segments(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="a"),
            TimelineSegment(start=2.0, end=4.0, kind="instrumental"),
            TimelineSegment(start=4.0, end=6.0, kind="vocals", text="valid text"),
        ]
        result = normalize_empty_vocals(segments)
        self.assertEqual(result[0].kind, "instrumental")
        self.assertEqual(result[1].kind, "instrumental")
        self.assertEqual(result[2].kind, "vocals")


class MergeSameKindSegmentsTest(unittest.TestCase):
    def test_merges_adjacent_same_kind(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="hello"),
            TimelineSegment(start=2.0, end=4.0, kind="vocals", text="world"),
        ]
        result = merge_same_kind_segments(segments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 4.0)
        self.assertEqual(result[0].text, "hello world")

    def test_does_not_merge_different_kind(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="hello"),
            TimelineSegment(start=2.0, end=4.0, kind="instrumental"),
        ]
        result = merge_same_kind_segments(segments)
        self.assertEqual(len(result), 2)

    def test_merge_gap(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="hello"),
            TimelineSegment(start=2.3, end=4.0, kind="vocals", text="world"),
        ]
        result = merge_same_kind_segments(segments, merge_gap=0.5)
        self.assertEqual(len(result), 1)

    def test_respects_merge_gap(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="hello"),
            TimelineSegment(start=2.6, end=4.0, kind="vocals", text="world"),
        ]
        result = merge_same_kind_segments(segments, merge_gap=0.5)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        result = merge_same_kind_segments([])
        self.assertEqual(result, [])

    def test_single_segment(self):
        segments = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="hello"),
        ]
        result = merge_same_kind_segments(segments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "hello")

    def test_consecutive_instrumental(self):
        segments = [
            TimelineSegment(start=0.0, end=1.0, kind="instrumental"),
            TimelineSegment(start=1.0, end=2.0, kind="instrumental"),
        ]
        result = merge_same_kind_segments(segments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 2.0)


if __name__ == "__main__":
    unittest.main()
