import unittest

from feverslop.domain.timeline import TimelineSegment
from feverslop.prompting.lyric_alignment import LyricTimelineAligner


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_prompt(self, system_prompt, prompt):
        self.calls.append((system_prompt, prompt))
        return self.response


class LyricTimelineAlignerTests(unittest.TestCase):
    def test_replaces_only_vocal_text_and_preserves_timing(self):
        timeline = [
            TimelineSegment(start=0.0, end=1.5, kind="instrumental", text=""),
            TimelineSegment(start=1.5, end=3.0, kind="vocals", text="helo wrld"),
            TimelineSegment(start=3.0, end=5.0, kind="vocals", text="secnd line"),
        ]
        llm = FakeLLM('{"segment1": "hello world", "segment2": "second line"}')
        aligner = LyricTimelineAligner(llm)

        corrected = aligner.align(timeline, "[Verse]\nhello world\nsecond line")

        self.assertEqual("", corrected[0].text)
        self.assertEqual("hello world", corrected[1].text)
        self.assertEqual("second line", corrected[2].text)
        self.assertEqual((1.5, 3.0, "vocals"), (corrected[1].start, corrected[1].end, corrected[1].kind))
        self.assertIn("REFERENCE_LYRICS", llm.calls[0][1])
        self.assertIn("segment1", llm.calls[0][1])

    def test_raises_when_llm_returns_wrong_segment_count(self):
        timeline = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="one"),
            TimelineSegment(start=2.0, end=4.0, kind="vocals", text="two"),
        ]
        llm = FakeLLM('{"segment1": "one"}')
        aligner = LyricTimelineAligner(llm)

        with self.assertRaisesRegex(ValueError, "Expected 2 corrected lyric segments"):
            aligner.align(timeline, "one\ntwo")

    def test_returns_original_timeline_when_no_vocal_segments_exist(self):
        timeline = [TimelineSegment(start=0.0, end=2.0, kind="instrumental", text="")]
        llm = FakeLLM("{}")
        aligner = LyricTimelineAligner(llm)

        corrected = aligner.align(timeline, "reference")

        self.assertIs(corrected, timeline)
        self.assertEqual([], llm.calls)
