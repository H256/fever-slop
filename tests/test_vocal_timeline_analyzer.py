import unittest
from pathlib import Path

from feverslop.domain.timeline import TimelineSegment
from feverslop.domain.timeline_transform import (
    merge_same_kind_segments,
    normalize_empty_vocals,
)


class TimelineSegmentImmutabilityTests(unittest.TestCase):
    def test_whisper_model_load_is_deferred_until_transcription(self):
        from unittest.mock import patch

        from feverslop.adapters.audio.vocal_timeline_analyzer import (
            VocalTimelineAnalyzer,
        )

        with patch("feverslop.adapters.audio.vocal_timeline_analyzer.whisper.load_model") as load_model:
            analyzer = VocalTimelineAnalyzer()
            load_model.assert_not_called()
            self.assertIsNone(analyzer.model)

    def test_whisper_transcription_requests_word_timestamps(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import (
            VocalTimelineAnalyzer,
        )

        class FakeWhisper:
            def __init__(self):
                self.kwargs = None

            def transcribe(self, _path, **kwargs):
                self.kwargs = kwargs
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        analyzer = VocalTimelineAnalyzer.__new__(VocalTimelineAnalyzer)
        analyzer.model = FakeWhisper()
        analyzer.language = "de"

        analyzer._transcribe(Path("vocals.wav"))

        self.assertTrue(analyzer.model.kwargs["word_timestamps"])

    def test_raw_whisper_segments_are_retained_before_filtering(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import (
            VocalTimelineAnalyzer,
        )

        class FakeWhisper:
            def transcribe(self, _path, **_kwargs):
                return {
                    "segments": [
                        {"start": 1.0, "end": 2.0, "text": "hello", "no_speech_prob": 0.9},
                    ],
                }

        analyzer = VocalTimelineAnalyzer.__new__(VocalTimelineAnalyzer)
        analyzer.model = FakeWhisper()
        analyzer.language = "de"

        self.assertEqual([], analyzer._transcribe(Path("vocals.wav")))
        self.assertEqual(
            [{"start": 1.0, "end": 2.0, "text": "hello", "no_speech_prob": 0.9}],
            analyzer.raw_whisper_segments,
        )

    def test_boundary_words_are_assigned_once_to_best_overlapping_vocal_range(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import (
            VocalTimelineAnalyzer,
        )

        analyzer = VocalTimelineAnalyzer.__new__(VocalTimelineAnalyzer)
        result = analyzer._combine_whisper_and_energy(
            [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "first second",
                    "words": [
                        {"word": "first", "start": 0.5, "end": 1.5},
                        {"word": "second", "start": 1.8, "end": 2.8},
                    ],
                },
            ],
            [(0.0, 2.0), (1.9, 4.0)],
        )

        self.assertEqual(("first",), tuple(item["word"] for item in result[0].word_timestamps))
        self.assertEqual(("second",), tuple(item["word"] for item in result[1].word_timestamps))
        self.assertEqual("first", result[0].text)
        self.assertEqual("second", result[1].text)

    def test_timeline_segment_is_frozen(self):
        seg = TimelineSegment(start=0.0, end=1.0, kind="vocals", text="hello")
        with self.assertRaises(Exception):
            seg.start = 99.0

    def test_normalize_empty_vocals_does_not_mutate_original(self):
        original = [
            TimelineSegment(start=0.0, end=1.0, kind="vocals", text="ab"),
            TimelineSegment(start=1.0, end=2.0, kind="vocals", text="valid text"),
        ]
        first_kind_before = original[0].kind

        result = normalize_empty_vocals(original, min_text_chars=3)

        self.assertEqual(first_kind_before, original[0].kind)
        self.assertEqual("instrumental", result[0].kind)
        self.assertEqual("vocals", result[1].kind)

    def test_normalize_empty_vocals_idempotent(self):
        timeline = [
            TimelineSegment(start=0.0, end=1.0, kind="vocals", text="ab"),
            TimelineSegment(start=1.0, end=2.0, kind="vocals", text="valid text"),
        ]
        result1 = normalize_empty_vocals(timeline, min_text_chars=3)
        result2 = normalize_empty_vocals(timeline, min_text_chars=3)

        self.assertEqual(len(result1), len(result2))
        for a, b in zip(result1, result2):
            self.assertEqual(a, b)

    def test_merge_same_kind_segments_does_not_mutate_original(self):
        original = [
            TimelineSegment(start=0.0, end=1.0, kind="vocals", text="first"),
            TimelineSegment(start=1.1, end=2.0, kind="vocals", text="second"),
        ]
        original_end = original[0].end

        result = merge_same_kind_segments(original, merge_gap=0.5)

        self.assertEqual(original_end, original[0].end)
        self.assertEqual(1, len(result))
        self.assertEqual("first second", result[0].text)
        self.assertAlmostEqual(2.0, result[0].end)

    def test_merge_same_kind_segments_preserves_word_timestamps(self):
        timeline = [
            TimelineSegment(
                start=0.0,
                end=1.0,
                kind="vocals",
                text="first",
                word_timestamps=({"word": "first", "start": 0.0, "end": 1.0},),
            ),
            TimelineSegment(
                start=1.1,
                end=2.0,
                kind="vocals",
                text="second",
                word_timestamps=({"word": "second", "start": 1.1, "end": 2.0},),
            ),
        ]

        result = merge_same_kind_segments(timeline, merge_gap=0.5)

        self.assertEqual(
            ("first", "second"),
            tuple(item["word"] for item in result[0].word_timestamps),
        )

    def test_merge_same_kind_segments_idempotent(self):
        timeline = [
            TimelineSegment(start=0.0, end=1.0, kind="vocals", text="first"),
            TimelineSegment(start=1.1, end=2.0, kind="vocals", text="second"),
            TimelineSegment(start=3.0, end=4.0, kind="instrumental"),
        ]
        result1 = merge_same_kind_segments(timeline, merge_gap=0.5)
        result2 = merge_same_kind_segments(timeline, merge_gap=0.5)

        self.assertEqual(len(result1), len(result2))
        for a, b in zip(result1, result2):
            self.assertEqual(a, b)

    def test_merge_preserves_text_only_when_non_empty(self):
        timeline = [
            TimelineSegment(start=0.0, end=1.0, kind="vocals", text="hello"),
            TimelineSegment(start=1.1, end=2.0, kind="vocals", text=""),
        ]
        result = merge_same_kind_segments(timeline, merge_gap=0.5)

        self.assertEqual(1, len(result))
        self.assertEqual("hello", result[0].text)

    def test_merge_empty_text_first_preserves_second(self):
        timeline = [
            TimelineSegment(start=0.0, end=1.0, kind="vocals", text=""),
            TimelineSegment(start=1.1, end=2.0, kind="vocals", text="world"),
        ]
        result = merge_same_kind_segments(timeline, merge_gap=0.5)

        self.assertEqual(1, len(result))
        self.assertEqual("world", result[0].text)

    def test_normalize_preserves_instrumental_segments(self):
        timeline = [
            TimelineSegment(start=0.0, end=1.0, kind="instrumental"),
            TimelineSegment(start=1.0, end=2.0, kind="vocals", text="valid"),
        ]
        result = normalize_empty_vocals(timeline, min_text_chars=3)

        self.assertEqual(2, len(result))
        self.assertEqual("instrumental", result[0].kind)
        self.assertEqual("vocals", result[1].kind)

    def test_merge_empty_timeline(self):
        self.assertEqual([], merge_same_kind_segments([]))


if __name__ == "__main__":
    unittest.main()
