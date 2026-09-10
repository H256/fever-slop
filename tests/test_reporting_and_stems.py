import io
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
from feverslop.composition.stage_runners import _discover_stem_files
from feverslop.ports.reporting import ConsoleReporter, ReporterLoggingHandler


class ReportingAndStemDiscoveryTests(unittest.TestCase):
    def test_console_reporter_prefixes_messages_with_local_timestamp(self):
        output = io.StringIO()
        reporter = ConsoleReporter(Console(file=output, force_terminal=False))

        reporter.message("pipeline started")

        self.assertRegex(output.getvalue(), r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] pipeline started\n$")

    def test_console_reporter_timestamps_each_multiline_message_line(self):
        output = io.StringIO()
        reporter = ConsoleReporter(Console(file=output, force_terminal=False))

        reporter.message("first\nsecond")

        self.assertRegex(
            output.getvalue(),
            r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] first\n"
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] second\n$",
        )

    def test_logging_handler_forwards_records_to_reporter(self):
        reporter = SimpleNamespace(messages=[], warnings=[])
        reporter.message = lambda text: reporter.messages.append(text)
        reporter.warning = lambda text, title=None: reporter.warnings.append((title, text))
        handler = ReporterLoggingHandler(reporter)

        handler.emit(logging.LogRecord("test", logging.WARNING, "", 0, "warning", (), None))

        self.assertEqual([("test", "warning")], reporter.warnings)

    def test_stem_discovery_returns_all_four_or_six_generated_stems(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stems_dir = Path(temp_dir) / "stems"
            stems_dir.mkdir()
            input_audio = Path(temp_dir) / "song.mp3"
            for name in ("vocals", "drums", "bass", "other", "guitar", "piano"):
                (stems_dir / f"{name}_song.wav").write_bytes(b"audio")

            result = _discover_stem_files(stems_dir, input_audio)

        self.assertEqual(
            {"vocals", "drums", "bass", "other", "guitar", "piano"},
            set(result or {}),
        )

    def test_audio_pipeline_reports_all_generated_stems_but_keeps_canonical_analysis_inputs(self):
        reporter = SimpleNamespace()
        reporter.messages = []
        reporter.message = lambda text: reporter.messages.append(text)
        reporter.table = lambda *args: reporter.messages.append(args)
        separator = SimpleNamespace(
            separate=lambda _input, output: {
                name: output / f"{name}.wav"
                for name in ("vocals", "drums", "bass", "other", "guitar", "piano")
            },
            close=lambda: None,
        )
        analyzer = SimpleNamespace(analyze=lambda _vocals: [], close=lambda: None)
        artifact_store = SimpleNamespace(read_json=lambda _path: {"bpm": 120, "beats": [], "source_used_for_beats": "fake"})
        beat_analyzer = SimpleNamespace(analyze_to_json_file=lambda **kwargs: None)

        pipeline = AudioTimelinePipeline(
            separator_factory=lambda _config: separator,
            vocal_analyzer_factory=lambda _config: analyzer,
            beat_analyzer_factory=lambda: beat_analyzer,
            normalize_empty_vocals=lambda timeline: timeline,
            merge_same_kind_segments=lambda timeline, merge_gap: timeline,
            save_timeline_json=lambda timeline, path: None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = {
                "config": SimpleNamespace(
                    input_audio=root / "song.mp3",
                    lyrics="",
                    audio=SimpleNamespace(),
                    vocal_detection=SimpleNamespace(merge_gap=0.25),
                ),
                "paths": SimpleNamespace(stems_dir=root / "stems"),
                "timeline_json": root / "timeline.json",
                "beat_json": root / "beat.json",
                "log_step": lambda _title: None,
                "log_file": lambda _label, _path: None,
                "run_spinner": lambda _description, func: func(),
                "reporter": reporter,
                "request": SimpleNamespace(skip_stem_separation=False, skip_whisper=False, skip_beat_analysis=False),
                "artifact_store": artifact_store,
            }

            result = pipeline.run(context)

        self.assertEqual({"vocals", "drums", "bass", "other", "guitar", "piano"}, set(result["stem_files"]))
        table = next(message for message in reporter.messages if isinstance(message, tuple) and message[0] == "Generated Stems")
        self.assertEqual({"vocals", "drums", "bass", "other", "guitar", "piano"}, {row[0] for row in table[2]})


if __name__ == "__main__":
    unittest.main()
