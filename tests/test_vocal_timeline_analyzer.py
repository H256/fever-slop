import os
from pathlib import Path
import subprocess
import sys
import unittest

from feverslop.domain.timeline import TimelineSegment
from feverslop.domain.timeline_transform import (
    merge_same_kind_segments,
    normalize_empty_vocals,
)


class TimelineSegmentImmutabilityTests(unittest.TestCase):
    def test_cli_import_does_not_require_whisper_runtime(self):
        environment = os.environ | {"PYTHONPATH": str(Path.cwd() / "src")}
        result = subprocess.run(
            [sys.executable, "-c", "import feverslop.cli.run_cli"],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )

        self.assertEqual(0, result.returncode, result.stderr)

    def test_whisper_model_load_is_deferred_until_transcription(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import (
            VocalTimelineAnalyzer,
        )

        analyzer = VocalTimelineAnalyzer()

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

        self.assertEqual(1, len(analyzer._transcribe(Path("vocals.wav"))))
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
        self.assertEqual("whisper", result[0].word_timestamps[0]["source"])
        self.assertEqual("whisper:0:0", result[0].word_timestamps[0]["word_id"])
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


class VocalEvidenceTests(unittest.TestCase):
    def test_decisions(self):
        from feverslop.domain.vocal_evidence import decide_transcript
        base = dict(start=58.32, end=64.14, text="synthetic", no_speech_prob=.926, avg_logprob=-.206)
        self.assertEqual("accepted", decide_transcript(base).status)
        for update, status in [
            ({"avg_logprob": None}, "uncertain"),
            ({"avg_logprob": -.8}, "uncertain"),
            ({"start": float("nan")}, "rejected"),
            ({"end": 58.32}, "rejected"),
            ({"start": -1}, "rejected"),
            ({"text": "Untertitelung des ZDF"}, "rejected"),
            ({"text": ""}, "rejected"),
            ({"no_speech_prob": None, "avg_logprob": None}, "uncertain"),
        ]:
            with self.subTest(update=update):
                self.assertEqual(status, decide_transcript(base | update).status)

    def test_high_no_speech_good_quality_retained_without_raw_mutation_or_text_logs(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        raw = dict(start=58.32, end=64.14, text="synthetic private transcript", no_speech_prob=.926, avg_logprob=-.206)
        class FakeWhisper:
            def transcribe(self, *_args, **_kwargs):
                return {"segments": [raw]}
        class Reporter:
            def __init__(self):
                self.messages = []
            def message(self, text):
                self.messages.append(text)
        reporter = Reporter()
        analyzer = VocalTimelineAnalyzer(reporter=reporter)
        analyzer.model = FakeWhisper()
        self.assertEqual([raw], analyzer._transcribe(Path("vocals.wav")))
        self.assertEqual([raw], analyzer.raw_whisper_segments)
        self.assertNotIn("evidence", raw)
        self.assertTrue(any("accepted=1" in m for m in reporter.messages))
        self.assertFalse(any(raw["text"] in m for m in reporter.messages))

    def test_rms_only_survives_normalization_and_merge(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        result = VocalTimelineAnalyzer()._combine_whisper_and_energy([], [(214.62, 238.18)])
        normalized = normalize_empty_vocals(result)
        self.assertEqual("vocals", normalized[0].kind)
        self.assertEqual("uncertain", normalized[0].evidence.activity_status)
        self.assertEqual("missing", normalized[0].evidence.transcript_status)
        merged = merge_same_kind_segments(normalized + [TimelineSegment(238.18, 240, "vocals", "hello")])
        self.assertEqual("uncertain", merged[0].evidence.activity_status)
        self.assertIn("rms_without_transcript", merged[0].evidence.reason_codes)

    def test_outside_rms_words_survive_once_as_conflict(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        raw = dict(start=1, end=4, text="inside outside", no_speech_prob=.1, avg_logprob=-.2,
                   words=[dict(word="inside", start=1, end=2), dict(word="outside", start=3, end=4)])
        result = VocalTimelineAnalyzer()._combine_whisper_and_energy([raw], [(1, 2)])
        self.assertEqual(["inside", "outside"], [w["word"] for s in result for w in s.word_timestamps])
        self.assertEqual("conflict", result[-1].evidence.activity_status)

    def test_fallback_text_assigned_once(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        result = VocalTimelineAnalyzer()._combine_whisper_and_energy(
            [dict(start=0, end=4, text="one anchor")], [(0, 1), (2, 4)])
        self.assertEqual(1, sum("one anchor" in s.text for s in result))

    def test_evidence_roundtrip(self):
        import json
        import tempfile
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        from feverslop.adapters.project_timeline_documents import ProjectTimelineDocuments
        from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
        from feverslop.pipeline.utils import save_timeline_json
        timeline = VocalTimelineAnalyzer()._combine_whisper_and_energy([], [(214.62, 238.18)])
        class Store:
            def read_json(self, path):
                return json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "render/timing/timeline.json"
            save_timeline_json(timeline, path)
            documents = ProjectTimelineDocuments(Path(directory))
            self.assertIn("evidence", documents.read_timeline()[0])
            documents.write_timeline(documents.read_timeline())
            self.assertEqual(timeline, AudioTimelinePipeline._load_existing_timeline(path, Store()))

    def test_merging_legacy_does_not_confirm_unknown_region(self):
        from feverslop.domain.vocal_evidence import VocalEvidence
        confirmed = TimelineSegment(0, 1, "vocals", "hello", evidence=VocalEvidence("confirmed", "accepted"))
        legacy = TimelineSegment(1, 2, "vocals", "")
        result = merge_same_kind_segments([confirmed, legacy])
        self.assertEqual("uncertain", result[0].evidence.activity_status)
        self.assertIn("legacy_evidence_unknown", result[0].evidence.reason_codes)

    def test_evidence_rejects_invalid_statuses(self):
        from feverslop.domain.vocal_evidence import VocalEvidence
        for value in [dict(activity_status="bogus", transcript_status="accepted"),
                      dict(activity_status="confirmed", transcript_status="bogus"),
                      dict(activity_status="confirmed", transcript_status="accepted", reason_codes="reason")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                VocalEvidence.from_dict(value)

    def test_conflict_merge_preserves_all_reasons(self):
        from feverslop.domain.vocal_evidence import VocalEvidence
        one = TimelineSegment(0, 1, "vocals", "hello", evidence=VocalEvidence("conflict", "accepted", ("outside",)))
        two = TimelineSegment(1, 2, "vocals", "", evidence=VocalEvidence("uncertain", "missing", ("rms_only",)))
        merged = merge_same_kind_segments([one, two])[0]
        self.assertEqual("conflict", merged.evidence.activity_status)
        self.assertEqual("uncertain", merged.evidence.transcript_status)
        self.assertEqual(("outside", "rms_only"), merged.evidence.reason_codes)

    def test_partially_outside_rms_word_is_conflict(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        raw = dict(start=0, end=2, text="boundary", avg_logprob=-.2,
                   words=[dict(word="boundary", start=0, end=2)])
        segment = VocalTimelineAnalyzer()._combine_whisper_and_energy([raw], [(1, 2)])[0]
        self.assertEqual("conflict", segment.evidence.activity_status)
        self.assertEqual(0, segment.word_timestamps[0]["start"])

    def test_legacy_json_omits_evidence_and_raw_diagnostics_are_unchanged(self):
        import json
        import tempfile
        from feverslop.pipeline.utils import save_timeline_json
        raw = [dict(start=58.32, end=64.14, text="synthetic", no_speech_prob=.926, avg_logprob=-.206)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.json"
            save_timeline_json([TimelineSegment(0, 1, "vocals", "hello")], path, whisper_raw=raw)
            self.assertNotIn("evidence", json.loads(path.read_text())[0])
            self.assertEqual(raw, json.loads(path.with_name("timeline_whisper_raw.json").read_text()))
            diagnostics = json.loads(path.with_name("timeline_whisper_evidence.json").read_text())
            self.assertEqual([{"index": 0, "status": "accepted", "reason_codes": ["quality_overrides_no_speech"]}], diagnostics)

    def test_lyric_alignment_clone_keeps_evidence(self):
        from feverslop.domain.vocal_evidence import VocalEvidence
        from feverslop.prompting.lyric_alignment import LyricTimelineAligner
        from tests.prompt_fakes import GeneralModulesFake
        evidence = VocalEvidence("conflict", "uncertain", ("transcript_outside_rms",))
        original = TimelineSegment(0, 2, "vocals", "old", evidence=evidence)
        aligner = LyricTimelineAligner(object(), modules=GeneralModulesFake(
            lyric_alignment={"segments": {"segment1": "new"}}))
        self.assertEqual(evidence, aligner.align([original], "new")[0].evidence)

    def test_rejected_raw_segment_has_persisted_reason_without_transcript(self):
        import json
        import tempfile
        from feverslop.pipeline.utils import save_timeline_json
        raw = [dict(start=0, end=1, text="Untertitelung des ZDF")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.json"
            save_timeline_json([], path, whisper_raw=raw)
            diagnostics = json.loads(path.with_name("timeline_whisper_evidence.json").read_text())
            self.assertEqual([{"index": 0, "status": "rejected", "reason_codes": ["subtitle_hallucination"]}], diagnostics)
            self.assertEqual(raw, json.loads(path.with_name("timeline_whisper_raw.json").read_text()))

    def test_rms_candidates_survive_but_real_gap_remains_instrumental(self):
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer

        analyzer = VocalTimelineAnalyzer()
        candidates = analyzer._combine_whisper_and_energy(
            [
                dict(start=214.62, end=220, text=""),
                dict(start=230, end=238.18, text="Untertitelung des ZDF"),
            ],
            [(214.62, 220), (230, 238.18)],
        )
        timeline = merge_same_kind_segments(normalize_empty_vocals(
            analyzer._insert_instrumental_segments(candidates, 238.18),
        ))

        self.assertEqual(
            [(0, 214.62, "instrumental"), (214.62, 220, "vocals"),
             (220, 230, "instrumental"), (230, 238.18, "vocals")],
            [(segment.start, segment.end, segment.kind) for segment in timeline],
        )
        for segment in (timeline[1], timeline[3]):
            self.assertEqual("", segment.text)
            self.assertEqual("uncertain", segment.evidence.activity_status)
            self.assertEqual("missing", segment.evidence.transcript_status)
            self.assertIn("rms_without_transcript", segment.evidence.reason_codes)
        self.assertIsNone(timeline[2].evidence)

    def test_resume_preserves_merge_decisions_at_precise_gap_boundary(self):
        import json
        import tempfile
        from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
        from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
        from feverslop.pipeline.utils import save_timeline_json

        analyzer = VocalTimelineAnalyzer()
        candidates = analyzer._combine_whisper_and_energy([], [(0, 1.002), (1.504, 3)])
        original = merge_same_kind_segments(normalize_empty_vocals(
            analyzer._insert_instrumental_segments(candidates, 3),
        ), merge_gap=.5)
        self.assertEqual(2, len(original))

        class Store:
            def read_json(self, path):
                return json.loads(path.read_text())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.json"
            save_timeline_json(original, path)
            resumed = AudioTimelinePipeline._load_existing_timeline(path, Store())
            self.assertEqual(original, merge_same_kind_segments(
                normalize_empty_vocals(resumed), merge_gap=.5,
            ))


if __name__ == "__main__":
    unittest.main()
