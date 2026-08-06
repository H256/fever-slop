import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline


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

    def write_json(self, path, data):
        self.json[str(path)] = data
        return Path(path)


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


class FakeLyricAligner:
    def __init__(self):
        self.calls = []

    def align(self, timeline, reference_lyrics):
        self.calls.append((timeline, reference_lyrics))
        for segment in timeline:
            if segment.kind == "vocals":
                segment.text = "corrected line"
        return timeline


class FakePromptPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.used_for_concepts = False
        self.saved = {}

    def create_story_idea(self, lyrics, notes=""):
        return "story"

    def create_style_block(self, lyrics, notes=""):
        return "style"

    def create_subject_and_locations(self, story_idea, notes=""):
        return {"subject": "subject", "locations": ["location"]}

    def create_concept_prompts(self, stage1_segments, story_idea, global_context=None, notes=""):
        self.used_for_concepts = True
        return {segment["segment_id"]: "concept" for segment in stage1_segments}

    def create_scene_details(self, concept_prompts, stage1_segments=None, global_context=None):
        return {
            segment["segment_id"]: {"camera_motion": "camera", "character_motion": "motion"}
            for segment in (stage1_segments or [])
        }

    def save_json(self, path, data, *, artifact_store):
        self.saved[str(path)] = data
        artifact_store.write_json(path, data)
        return Path(path)


class FakeConceptBatcher:
    def __init__(self, llm, batch_size, request_timeout_seconds=None):
        self.llm = llm
        self.batch_size = batch_size
        self.request_timeout_seconds = request_timeout_seconds
        self.used = False

    def create_concept_prompts_batched(self, stage1_segments, story_idea, global_context, notes=""):
        self.used = True
        return {segment["segment_id"]: "batched concept" for segment in stage1_segments}


class FakeScenePromptBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.calls = []

    def build_scene_prompts(self, **kwargs):
        self.calls.append(kwargs)
        output_json_path = kwargs["output_json_path"]
        Path(output_json_path).write_text("[]", encoding="utf-8")
        return Path(output_json_path)


def _audio_context(temp: Path, artifact_store: FakeArtifactStore) -> GenerateRenderPlanContext:
    config = SimpleNamespace(
        input_audio=temp / "song.wav",
        lyrics="",
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
    def _audio_pipeline(self, separator, vocal_analyzer, beat_analyzer, lyric_aligner_factory=None):
        return AudioTimelinePipeline(
            separator_factory=lambda config: separator,
            vocal_analyzer_factory=lambda config: vocal_analyzer,
            beat_analyzer_factory=lambda: beat_analyzer,
            lyric_aligner_factory=lyric_aligner_factory,
            normalize_empty_vocals=lambda timeline: timeline,
            merge_same_kind_segments=lambda timeline, merge_gap: timeline,
            save_timeline_json=lambda timeline, path: None,
        )

    def test_audio_pipeline_uses_injected_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            artifact_store = FakeArtifactStore()
            separator = FakeSeparator()
            vocal_analyzer = FakeVocalAnalyzer()
            beat_analyzer = FakeBeatAnalyzer(artifact_store)
            pipeline = self._audio_pipeline(separator, vocal_analyzer, beat_analyzer)

            context = pipeline.execute(_audio_context(temp, artifact_store))

            self.assertEqual([(temp / "song.wav", temp / "stems")], separator.calls)
            self.assertEqual([temp / "stems" / "vocals.wav"], vocal_analyzer.calls)
            self.assertEqual(temp / "beat.json", beat_analyzer.calls[0]["output_json_path"])
            self.assertEqual(120, context.beat_data["bpm"])
            self.assertIn("stem_files", context.keys())

    def test_audio_pipeline_aligns_lyrics_when_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            artifact_store = FakeArtifactStore()
            separator = FakeSeparator()
            vocal_analyzer = FakeVocalAnalyzer()
            beat_analyzer = FakeBeatAnalyzer(artifact_store)
            lyric_aligner = FakeLyricAligner()
            context = _audio_context(temp, artifact_store)
            context.config.lyrics = "full reference lyrics"
            pipeline = self._audio_pipeline(
                separator,
                vocal_analyzer,
                beat_analyzer,
                lyric_aligner_factory=lambda context: lyric_aligner,
            )

            result = pipeline.execute(context)

            self.assertEqual(1, len(lyric_aligner.calls))
            self.assertEqual("full reference lyrics", lyric_aligner.calls[0][1])
            self.assertEqual("corrected line", result.timeline[0].text)

    def test_audio_pipeline_skips_alignment_without_configured_lyrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            artifact_store = FakeArtifactStore()
            separator = FakeSeparator()
            vocal_analyzer = FakeVocalAnalyzer()
            beat_analyzer = FakeBeatAnalyzer(artifact_store)
            lyric_aligner = FakeLyricAligner()
            context = _audio_context(temp, artifact_store)
            context.config.lyrics = ""
            pipeline = self._audio_pipeline(
                separator,
                vocal_analyzer,
                beat_analyzer,
                lyric_aligner_factory=lambda context: lyric_aligner,
            )

            result = pipeline.execute(context)

            self.assertEqual([], lyric_aligner.calls)
            self.assertEqual("line", result.timeline[0].text)

    def test_prompt_pipeline_uses_injected_dependencies_without_batching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            llm = object()
            prompt_pipeline = FakePromptPipeline(llm)
            scene_prompt_builder = FakeScenePromptBuilder(llm)
            pipeline = PromptGenerationPipeline(
                llm_factory=lambda app_config: llm,
                prompt_pipeline_factory=lambda llm_arg: prompt_pipeline,
                concept_batcher_factory=lambda llm_arg, batch_size: self.fail("batcher should not be used"),
                scene_prompt_builder_factory=lambda llm_arg: scene_prompt_builder,
            )
            context = _prompt_context(temp, concept_batch_size=0)

            result = pipeline.execute(context)

            self.assertTrue(prompt_pipeline.used_for_concepts)
            self.assertEqual(1, len(scene_prompt_builder.calls))
            self.assertEqual({"segment_001": "concept"}, result.concept_prompts)

    def test_prompt_pipeline_uses_injected_concept_batcher_when_batching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            llm = object()
            prompt_pipeline = FakePromptPipeline(llm)
            concept_batcher = FakeConceptBatcher(llm, 2)
            scene_prompt_builder = FakeScenePromptBuilder(llm)
            def concept_batcher_factory(llm_arg, batch_size, request_timeout_seconds=None):
                concept_batcher.batch_size = batch_size
                concept_batcher.request_timeout_seconds = request_timeout_seconds
                return concept_batcher

            pipeline = PromptGenerationPipeline(
                llm_factory=lambda app_config: llm,
                prompt_pipeline_factory=lambda llm_arg: prompt_pipeline,
                concept_batcher_factory=concept_batcher_factory,
                scene_prompt_builder_factory=lambda llm_arg: scene_prompt_builder,
            )
            context = _prompt_context(temp, concept_batch_size=2)

            result = pipeline.execute(context)

            self.assertFalse(prompt_pipeline.used_for_concepts)
            self.assertTrue(concept_batcher.used)
            self.assertEqual(2, concept_batcher.batch_size)
            self.assertEqual(180.0, concept_batcher.request_timeout_seconds)
            self.assertEqual({"segment_001": "batched concept"}, result.concept_prompts)


def _prompt_context(temp: Path, concept_batch_size: int) -> GenerateRenderPlanContext:
    return GenerateRenderPlanContext(
        request=SimpleNamespace(concept_batch_size=concept_batch_size),
        config=SimpleNamespace(
            story_idea="",
            style="",
            subject="",
            locations=[],
            steering=SimpleNamespace(
                global_="",
                story_idea="",
                style="",
                subject="",
                locations="",
                concepts="",
                zimage="",
                ltx="",
                final_prompts="",
            ),
            prompt_guidance=SimpleNamespace(as_prompt_context=lambda: {}),
        ),
        app_config=SimpleNamespace(llm=SimpleNamespace(
            base_url="http://fake",
            model="fake",
            temperature=0,
            max_tokens=100,
            request_timeout_seconds=180.0
        )),
        stage1_segments=[
            {
                "segment_id": "segment_001",
                "scene": 1,
                "type": "vocals",
                "lyrics": "line",
            }
        ],
        artifact_store=FakeArtifactStore(),
        console=FakeConsole(),
        log_step=lambda title: None,
        log_file=lambda label, path: None,
        run_spinner=lambda description, func: func(),
        resolved_context_json=temp / "resolved_context.json",
        concept_prompts_json=temp / "concept_prompts.json",
        scene_details_json=temp / "scene_details.json",
        scene_prompts_json=temp / "scene_prompts.json",
    )


if __name__ == "__main__":
    unittest.main()
