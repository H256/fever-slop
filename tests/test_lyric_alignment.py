import unittest

from feverslop.domain.timeline import TimelineSegment
from feverslop.prompting.lyric_alignment import LyricTimelineAligner
from feverslop.prompting.guide_loader import load_markdown_guide
from tests.prompt_fakes import GeneralModulesFake


class LyricTimelineAlignerTests(unittest.TestCase):
    def test_lyric_alignment_guide_contains_boundary_contract(self):
        guide = load_markdown_guide("lyric-alignment")

        self.assertIn("Do not merge, split, skip, reorder", guide)
        self.assertIn("Return exactly one output value", guide)

    def test_replaces_only_vocal_text_and_preserves_timing(self):
        timeline = [
            TimelineSegment(start=0.0, end=1.5, kind="instrumental", text=""),
            TimelineSegment(start=1.5, end=3.0, kind="vocals", text="helo wrld"),
            TimelineSegment(start=3.0, end=5.0, kind="vocals", text="secnd line"),
        ]
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": "hello world", "segment2": "second line"}})
        aligner = LyricTimelineAligner(object(), modules=modules)

        corrected = aligner.align(timeline, "[Verse]\nhello world\nsecond line")

        self.assertEqual("", corrected[0].text)
        self.assertEqual("hello world", corrected[1].text)
        self.assertEqual("second line", corrected[2].text)
        self.assertEqual((1.5, 3.0, "vocals"), (corrected[1].start, corrected[1].end, corrected[1].kind))
        self.assertIn("hello world", modules.calls[0].payload["REFERENCE_LYRICS"])
        self.assertEqual("segment1", modules.calls[0].payload["WHISPER_SEGMENTS"][0]["key"])

    def test_lyrics_are_reconstructed_from_word_timestamps(self):
        timeline = [
            TimelineSegment(
                start=4.76,
                end=6.94,
                kind="vocals",
                text="wrong corrected phrase",
                word_timestamps=(
                    {"word": "ihr", "start": 3.92, "end": 5.54},
                    {"word": "seid", "start": 5.54, "end": 5.76},
                    {"word": "unglaublich!", "start": 5.76, "end": 6.74},
                ),
            )
        ]
        aligner = LyricTimelineAligner(
            object(),
            modules=GeneralModulesFake(
                lyric_alignment={"segments": {"segment1": "Das ist unser letztes Lied für heute Nacht."}}
            ),
        )

        corrected = aligner.align(timeline, "Das ist unser letztes Lied für heute Nacht.")

        self.assertEqual("ihr seid unglaublich!", corrected[0].text)

    def test_raises_when_llm_returns_wrong_segment_count(self):
        timeline = [
            TimelineSegment(start=0.0, end=2.0, kind="vocals", text="one"),
            TimelineSegment(start=2.0, end=4.0, kind="vocals", text="two"),
        ]
        aligner = LyricTimelineAligner(object(), modules=GeneralModulesFake(lyric_alignment={"segments": {"segment1": "one"}}))

        with self.assertRaisesRegex(ValueError, "Expected 2 corrected lyric segments"):
            aligner.align(timeline, "one\ntwo")

    def test_returns_original_timeline_when_no_vocal_segments_exist(self):
        timeline = [TimelineSegment(start=0.0, end=2.0, kind="instrumental", text="")]
        modules = GeneralModulesFake()
        aligner = LyricTimelineAligner(object(), modules=modules)

        corrected = aligner.align(timeline, "reference")

        self.assertIs(corrected, timeline)
        self.assertEqual([], modules.calls)
