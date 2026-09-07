import unittest

from feverslop.domain.timeline import TimelineSegment
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.lyric_alignment import LyricTimelineAligner
from tests.prompt_fakes import GeneralModulesFake


class LyricTimelineAlignerTests(unittest.TestCase):
    def test_corrected_first_word_keeps_later_whisper_anchors(self):
        words = "Turning the old world into dust".split()
        bounds = [55.60, 56.02, 56.20, 56.65, 57.12, 57.70, 58.32]
        segment = TimelineSegment(55.60, 79.46, "vocals", " ".join(words), tuple(
            {"word": word, "start": bounds[i], "end": bounds[i+1], "source": "whisper",
             "source_index": i, "word_id": f"w{i}"}
            for i, word in enumerate(words)
        ))
        aligned = LyricTimelineAligner._complete_word_timestamps("Churning the old world into dust", segment)
        self.assertEqual(56.02, aligned[1]["start"])
        self.assertEqual(58.32, aligned[-1]["end"])
        self.assertEqual("corrected_from_whisper", aligned[0]["source"])
        self.assertEqual("w0", aligned[0]["word_id"])

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
        self.assertNotIn("[Verse]", modules.calls[0].payload["REFERENCE_LYRICS"])
        self.assertEqual("segment1", modules.calls[0].payload["WHISPER_SEGMENTS"][0]["key"])

    def test_project_lyrics_are_preserved_when_word_timestamps_are_incomplete(self):
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
            ),
        ]
        aligner = LyricTimelineAligner(
            object(),
            modules=GeneralModulesFake(
                lyric_alignment={"segments": {"segment1": "Das ist unser letztes Lied für heute Nacht."}},
            ),
        )

        corrected = aligner.align(timeline, "Das ist unser letztes Lied für heute Nacht.")

        self.assertEqual("Das ist unser letztes Lied für heute Nacht.", corrected[0].text)
        self.assertEqual(
            ["Das", "ist", "unser", "letztes", "Lied", "für", "heute", "Nacht."],
            [item["word"] for item in corrected[0].alignment["targets"]],
        )
        self.assertEqual((), corrected[0].word_timestamps)
        self.assertEqual(8, len(corrected[0].alignment["targets"]))

    def test_prefix_insertions_and_invalid_anchors_remain_unresolved(self):
        segment = TimelineSegment(
            start=41.69,
            end=44.01,
            kind="vocals",
            text="mein Name wie ein Messer",
            word_timestamps=(
                {"word": "mein", "start": 41.64, "end": 41.84},
                {"word": "Name", "start": 41.84, "end": 42.3},
                {"word": "wie", "start": 42.3, "end": 42.7},
                {"word": "ein", "start": 42.7, "end": 43.06},
                {"word": "Messer", "start": 43.06, "end": 43.88},
            ),
        )

        timestamps = LyricTimelineAligner._complete_word_timestamps(
            "Ich trug mein Name wie ein Messer", segment,
        )

        self.assertTrue(all(item["end"] > item["start"] for item in timestamps))
        self.assertEqual(["Name", "wie", "ein", "Messer"], [item["word"] for item in timestamps])
        self.assertEqual(43.88, timestamps[-1]["end"])

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

    def test_removes_section_markers_returned_as_segment_text(self):
        timeline = [TimelineSegment(start=0.0, end=2.0, kind="vocals", text="noise")]
        aligner = LyricTimelineAligner(
            object(),
            modules=GeneralModulesFake(
                lyric_alignment={"segments": {"segment1": "[Verse]"}},
            ),
        )

        corrected = aligner.align(timeline, "[Verse]\nreal words")

        self.assertEqual("", corrected[0].text)
        self.assertEqual((), corrected[0].word_timestamps)


class WordAlignmentContractTests(unittest.TestCase):
    def test_analyzer_merge_then_align_preserves_boundary_diagnostics(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        from feverslop.domain.timeline_transform import merge_same_kind_segments, normalize_empty_vocals
        analyzer = VocalTimelineAnalyzer.__new__(VocalTimelineAnalyzer)
        timeline = analyzer._combine_whisper_and_energy([
            {"start": 0.5, "end": 1.5, "text": "hello", "words": [
                {"word": "hello", "start": 0.5, "end": 1.5}]}], [(0, 1), (1, 2)])
        self.assertEqual(1, timeline[0].word_timestamps[0]["segment_end"])
        merged = merge_same_kind_segments(normalize_empty_vocals(timeline))
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": "hello"}})
        aligned = LyricTimelineAligner(object(), modules=modules).align(merged, "hello")
        self.assertEqual((), aligned[0].word_timestamps)
        self.assertEqual("out_of_segment", aligned[0].alignment["diagnostics"][0]["reason"])

    def test_first_merge_retains_legacy_component_bounds(self):
        from feverslop.domain.timeline_transform import merge_same_kind_segments
        merged = merge_same_kind_segments([
            TimelineSegment(0, 1, "vocals", "hello", ({"word": "hello", "start": 0.5, "end": 1.5},)),
            TimelineSegment(1, 2, "vocals", "world"),
        ])
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": "hello world"}})
        aligned = LyricTimelineAligner(object(), modules=modules).align(merged, "hello world")
        self.assertEqual((), aligned[0].word_timestamps)
        self.assertEqual("out_of_segment", aligned[0].alignment["diagnostics"][0]["reason"])

    def test_empty_correction_survives_pipeline_normalization_on_resume(self):
        from feverslop.domain.timeline_transform import normalize_empty_vocals
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": ""}})
        aligner = LyricTimelineAligner(object(), modules=modules)
        timeline = aligner.align([TimelineSegment(0, 2, "vocals", "noise")], "reference")
        resumed = aligner.align(normalize_empty_vocals(timeline), "reference")
        self.assertEqual("vocals", resumed[0].kind)
        self.assertEqual(1, len(modules.calls))

    def test_merged_realign_does_not_promote_out_of_component_timing(self):
        from feverslop.domain.timeline_transform import merge_same_kind_segments
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": "hello", "segment2": "world"}})
        aligned = LyricTimelineAligner(object(), modules=modules).align([
            TimelineSegment(0, 1, "vocals", "hello", ({"word": "hello", "start": 0.5, "end": 1.5},)),
            TimelineSegment(1, 2, "vocals", "world"),
        ], "hello world")
        merged = merge_same_kind_segments(aligned)
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": "hello world!"}})
        changed = LyricTimelineAligner(object(), modules=modules).align(merged, "hello world!")
        self.assertEqual((), changed[0].word_timestamps)
        self.assertEqual("out_of_segment", changed[0].alignment["diagnostics"][0]["reason"])

    def test_reports_progress_without_lyric_payloads(self):
        from unittest.mock import Mock
        reporter = Mock()
        aligner = LyricTimelineAligner(object(), modules=GeneralModulesFake(
            lyric_alignment={"segments": {"segment1": "secret lyric"}}))
        aligner.set_reporter(reporter)
        aligner.align([TimelineSegment(0, 2, "vocals", "private raw")], "secret lyric")
        messages = " ".join(call.args[0] for call in reporter.message.call_args_list)
        self.assertIn("0/1", messages)
        self.assertIn("1/1", messages)
        self.assertIn("unresolved=2", messages)
        self.assertNotIn("secret lyric", messages)
        self.assertNotIn("private raw", messages)

    def test_merge_preserves_raw_alignment_and_unresolved_positions(self):
        from feverslop.domain.timeline_transform import merge_same_kind_segments
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": "oh hello", "segment2": "world"}})
        aligned = LyricTimelineAligner(object(), modules=modules).align([
            TimelineSegment(0, 1, "vocals", "hello"),
            TimelineSegment(1, 2, "vocals", "world"),
        ], "oh hello world")
        merged = merge_same_kind_segments(aligned)[0]
        self.assertEqual("hello world", merged.alignment["raw_text"])
        self.assertEqual([0, 1, 2], [row["target_index"] for row in merged.alignment["targets"]])
        self.assertEqual(2, len(merged.alignment["components"]))

    def align(self, source, target, rows=None):
        from feverslop.domain.word_alignment import align_words
        rows = rows if rows is not None else tuple(
            {"word": word, "start": i, "end": i + 0.5, "source": "whisper", "word_id": f"w{i}"}
            for i, word in enumerate(source.split())
        )
        return align_words(source, rows, target, 0, 100)

    def test_insert_delete_and_punctuation_preserve_unique_anchors(self):
        result = self.align("hello old world", "Oh HELLO, world!")
        self.assertEqual(["HELLO,", "world!"], [w["word"] for w in result["timed_words"]])
        self.assertEqual([0, 2], [w["start"] for w in result["timed_words"]])
        first = result["targets"][0]
        self.assertEqual("unresolved", first["source"])
        self.assertNotIn("start", first)
        self.assertEqual("w0", first["next_word_id"])
        self.assertIn("deletion", [operation["operation"] for operation in result["operations"]])

    def test_equal_optimal_repeat_assignments_are_ambiguous(self):
        result = self.align("go go home", "go home")
        self.assertEqual(["home"], [w["word"] for w in result["timed_words"]])
        self.assertEqual("ambiguous", result["targets"][0]["reason"])
        self.assertEqual([0, 1], result["targets"][0]["candidate_source_indexes"])

    def test_unrelated_phrase_is_never_timestamped(self):
        result = self.align("sky moon rain", "books dance velvet")
        self.assertEqual([], result["timed_words"])

    def test_invalid_words_are_diagnosed_without_clamping(self):
        rows = [
            {"word": "missing"},
            {"word": "nan", "start": float("nan"), "end": 2},
            {"word": "zero", "start": 3, "end": 3},
            {"word": "outside", "start": -1, "end": 1},
            {"word": "valid", "start": 5, "end": 6},
            {"word": "backward", "start": 4, "end": 4.5},
        ]
        result = self.align("missing nan zero outside valid backward", "missing nan zero outside valid backward", rows)
        self.assertEqual(["valid"], [w["word"] for w in result["timed_words"]])
        self.assertEqual({"missing_timing", "nonfinite_timing", "nonpositive_duration", "out_of_segment", "nonmonotone_timing"},
                         {d["reason"] for d in result["diagnostics"]})
        self.assertEqual(-1, result["raw_words"][3]["start"])

    def test_legacy_timestamps_are_not_promoted(self):
        result = self.align("Turning world", "Churning world", [
            {"word": "Turning", "start": 0, "end": 1},
            {"word": "world", "start": 1, "end": 2},
        ])
        self.assertEqual(["legacy_unverified", "legacy_unverified"],
                         [w["source"] for w in result["timed_words"]])

    def test_json_resume_is_idempotent_and_changed_reference_uses_raw(self):
        import tempfile
        from pathlib import Path
        from feverslop.adapters.local_artifacts import JsonArtifactStore
        from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
        from feverslop.pipeline.utils import save_timeline_json
        segment = TimelineSegment(0, 10, "vocals", "Turning world", (
            {"word": "Turning", "start": 0, "end": 1, "source": "whisper"},
            {"word": "world", "start": 1, "end": 2, "source": "whisper"},
        ))
        modules = GeneralModulesFake(lyric_alignment={"segments": {"segment1": "Churning world"}})
        aligner = LyricTimelineAligner(object(), modules=modules)
        timeline = aligner.align([segment], "Churning world")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.json"
            for _ in range(3):
                save_timeline_json(timeline, path)
                loaded = AudioTimelinePipeline._load_existing_timeline(path, JsonArtifactStore())
                self.assertEqual(timeline, loaded)
                timeline = aligner.align(loaded, "Churning world")
            self.assertEqual(1, len(modules.calls))
            aligner.align(timeline, "Burning world")
            self.assertEqual("Turning world", modules.calls[-1].payload["WHISPER_SEGMENTS"][0]["text"])
            self.assertEqual(2, len(modules.calls))
