import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoprompter.application.audio_timeline_pipeline import AudioTimelinePipeline
from autoprompter.application.pipeline_context import GenerateRenderPlanContext


class FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, message):
        self.messages.append(message)


class FakeArtifactStore:
    def __init__(self):
        self.json = {}

    def read_json(self, path):
        return self.json[str(path)]


class FakeTimelineSegment:
    def __init__(self, kind, start, end, text=""):
        self.kind = kind
        self.start = start
        self.end = end
        self.text = text


class FakeSeparator:
    def __init__(self):
        self.calls = []

    def separate(self, input_audio, output_dir):
        self.calls.append((input_audio, output_dir))
        return {
            "vocals": output_dir / "vocals.wav",
            "drums": output_dir / "drums.wav",
            "bass": output_dir / "bass.wav",
            "other": output_dir / "other.wav",
        }


class FakeVocalAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(self, vocals_path):
        self.calls.append(vocals_path)
        return [
            FakeTimelineSegment("vocals", 0.0, 1.0, "line"),
            FakeTimelineSegment("instrumental", 1.0, 2.0),
        ]


class FakeBeatAnalyzer:
    def __init__(self, artifact_store):
        self.artifact_store = artifact_store
        self.calls = []

    def analyze_to_json_file(self, **kwargs):
        self.calls.append(kwargs)
        self.artifact_store.json[str(kwargs["output_json_path"])] = {
            "bpm": 120,
            "beats": [0.0, 0.5],
            "source_used_for_beats": "fake",
        }


def _audio_context(temp: Path, artifact_store: FakeArtifactStore) -> GenerateRenderPlanContext:
    config = SimpleNamespace(
        input_audio=temp / "song.wav",
        audio=SimpleNamespace(demucs_model="fake-demucs", whisper_model="fake-whisper", language="en"),
        vocal_detection=SimpleNamespace(
            merge_gap=0.25,
            min_vocal_duration=0.1,
            min_silence_duration=0.1,
            rms_low_percentile=10,
            rms_high_percentile=90,
            rms_ratio=1.5,
            smooth_frames=3,
        ),
    )
    paths = SimpleNamespace(stems_dir=temp / "stems")
    return GenerateRenderPlanContext(
        config=config,
        paths=paths,
        song_id="demo",
        video_settings=SimpleNamespace(fps=24),
        artifact_store=artifact_store,
        console=FakeConsole(),
        log_step=lambda title: None,
        log_file=lambda label, path: None,
        run_spinner=lambda description, func: func(),
        timeline_json=temp / "timeline.json",
        beat_json=temp / "beat.json",
    )


class GeneratePipelineDependencyTests(unittest.TestCase):
    def test_audio_pipeline_uses_injected_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            artifact_store = FakeArtifactStore()
            separator = FakeSeparator()
            vocal_analyzer = FakeVocalAnalyzer()
            beat_analyzer = FakeBeatAnalyzer(artifact_store)
            pipeline = AudioTimelinePipeline(
                separator_factory=lambda config: separator,
                vocal_analyzer_factory=lambda config: vocal_analyzer,
                beat_analyzer_factory=lambda: beat_analyzer,
            )

            context = pipeline.execute(_audio_context(temp, artifact_store))

            self.assertEqual([(temp / "song.wav", temp / "stems")], separator.calls)
            self.assertEqual([temp / "stems" / "vocals.wav"], vocal_analyzer.calls)
            self.assertEqual(temp / "beat.json", beat_analyzer.calls[0]["output_json_path"])
            self.assertEqual(120, context.beat_data["bpm"])
            self.assertIn("stem_files", context.keys())


if __name__ == "__main__":
    unittest.main()
