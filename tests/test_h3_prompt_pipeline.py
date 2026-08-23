import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.application.h3_prompt_pipeline import (
    _attach_beat_events,
    _attach_relay_segments,
    _configured_audio_paths,
)
from feverslop.domain.performance_sync import select_performance_stems


class ConfiguredAudioPathTests(unittest.TestCase):
    def test_attaches_scene_local_beat_and_downbeat_events(self):
        result = _attach_beat_events(
            [{"segment_id": "s2", "start": 8.0, "end": 10.0}],
            {
                "bpm": 120,
                "beats": [
                    {"time": 7.5, "downbeat": False, "impact": 0.1},
                    {"time": 8.5, "downbeat": True, "impact": 0.8},
                    {"time": 9.0, "downbeat": False, "impact": 0.4},
                    {"time": 10.5, "downbeat": False, "impact": 0.2},
                ],
            },
        )

        timing = result[0]["performance_timing"]
        self.assertEqual(120.0, timing["bpm"])
        self.assertEqual([0.5, 1.0], [beat["time_seconds"] for beat in timing["beats"]])
        self.assertTrue(timing["beats"][0]["downbeat"])

    def test_selects_drum_stem_for_instrumental_drummer_scene(self):
        segment = {
            "type": "instrumental",
            "references": {"actor_reference_descriptions": [
                {"name": "Drummer", "role": "Percussionist"},
                {"name": "Bassist", "role": "Bassist"},
            ]},
            "ltx": {"prompt_relay": [{"state": "instrumental"}]},
        }

        selected = select_performance_stems(
            segment,
            available_stems={"vocals", "drums", "bass", "other", "full_mix"},
        )

        self.assertEqual(["drums", "full_mix"], selected)

    def test_selects_vocal_stem_for_visible_singer_during_vocal_scene(self):
        segment = {
            "type": "vocals",
            "references": {"actor_reference_descriptions": [
                {"name": "Lead Vocalist", "role": "Lead Singer"},
            ]},
            "ltx": {"prompt_relay": [{"state": "singing"}]},
        }

        selected = select_performance_stems(
            segment,
            available_stems={"vocals", "drums", "full_mix"},
        )

        self.assertEqual(["vocals", "full_mix"], selected)
    def test_attaches_relay_scene_to_matching_stage1_segment(self):
        result = _attach_relay_segments(
            [{"segment_id": "segment_002", "scene": 2, "type": "instrumental"}],
            [{
                "scene": 2,
                "fps": 24,
                "duration_seconds": 3.16,
                "prompt_relay": [{"frame_start": 0, "frame_end": 76, "state": "instrumental"}],
            }],
        )

        self.assertEqual(24, result[0]["fps"])
        self.assertEqual(76, result[0]["ltx"]["prompt_relay"][0]["frame_end"])
    def test_selects_configured_stems_in_configured_order(self):
        config = SimpleNamespace(
            minimax_h3_audio_refs=SimpleNamespace(stems=["vocals", "full_mix"]),
        )
        stem_files = {
            "drums": Path("drums.wav"),
            "full_mix": Path("full_mix.wav"),
            "vocals": Path("vocals.wav"),
            "bass": Path("bass.wav"),
        }

        selected = _configured_audio_paths(config, stem_files)

        self.assertEqual(
            {"vocals": Path("vocals.wav"), "full_mix": Path("full_mix.wav")},
            selected,
        )
        self.assertEqual(["vocals", "full_mix"], list(selected))

    def test_returns_none_when_no_configured_stem_exists(self):
        config = SimpleNamespace(
            minimax_h3_audio_refs=SimpleNamespace(stems=["vocals", "full_mix"]),
        )

        self.assertIsNone(_configured_audio_paths(config, {"drums": Path("drums.wav")}))

    def test_adds_input_audio_as_configured_full_mix(self):
        config = SimpleNamespace(
            minimax_h3_audio_refs=SimpleNamespace(stems=["vocals", "full_mix"]),
        )

        selected = _configured_audio_paths(
            config,
            {"vocals": Path("vocals.wav")},
            Path("song.wav"),
        )

        self.assertEqual(
            {"vocals": Path("vocals.wav"), "full_mix": Path("song.wav")},
            selected,
        )


class DspyPromptPipelineSelectionTests(unittest.TestCase):
    def test_run_accepts_typed_context_when_relay_path_is_optional(self):
        from feverslop.application.h3_prompt_pipeline import H3PromptPipeline

        captured = []

        class FakeBuilder:
            def build_all_h3_prompts(self, **kwargs):
                captured.append(kwargs["stage1_segments"])

        class FakeArtifactStore:
            def read_json(self, path):
                if path == Path("relay.json"):
                    return [{
                        "metadata": {"segment_id": "s1"},
                        "fps": 24,
                        "ltx": {"prompt_relay": [{"frame_start": 0}]},
                    }]
                return []

        config = SimpleNamespace(
            video_pipeline="minimax-h3-t2v",
            minimax_h3_audio_refs=SimpleNamespace(stems=[]),
            project_dir=None,
        )
        context = GenerateRenderPlanContext(
            app_config={},
            config=config,
            stage1_segments=[{"segment_id": "s1"}],
            concept_prompts={},
            scene_details={},
            global_context={},
            h3_prompts_json=Path("h3.json"),
            artifact_store=FakeArtifactStore(),
            log_step=lambda message: None,
            log_file=lambda label, path: None,
            ltx_prompt_relay_json=Path("relay.json"),
        )
        pipeline = H3PromptPipeline(
            llm_factory=lambda current_config: None,
            h3_prompt_builder_factory=lambda llm: FakeBuilder(),
            dspy_prompt_builder_factory=None,
        )

        pipeline.run(context)

        self.assertEqual(24, captured[0][0]["fps"])

    def _run_pipeline(self, video_pipeline, reporter=None):
        from feverslop.application.h3_prompt_pipeline import H3PromptPipeline

        calls = []

        class FakeBuilder:
            def __init__(self, name):
                self.name = name

            def build_all_h3_prompts(self, **kwargs):
                calls.append((self.name, kwargs))

        class FakeArtifactStore:
            def read_json(self, path):
                return []

        config = SimpleNamespace(
            video_pipeline=video_pipeline,
            minimax_h3_audio_refs=SimpleNamespace(stems=[]),
        )
        pipeline = H3PromptPipeline(
            llm_factory=lambda config: None,
            h3_prompt_builder_factory=lambda llm: "legacy",
            dspy_prompt_builder_factory=lambda llm: "dspy",
        )
        builders = {"legacy": FakeBuilder("legacy"), "dspy": FakeBuilder("dspy")}
        pipeline.h3_prompt_builder_factory = lambda llm: builders["legacy"]
        pipeline.dspy_prompt_builder_factory = lambda llm: builders["dspy"]
        context = {
            "app_config": {},
            "config": config,
            "stage1_segments": [{"segment_id": "s1"}],
            "concept_prompts": {},
            "scene_details": {},
            "global_context": {},
            "h3_prompts_json": "test.json",
            "artifact_store": FakeArtifactStore(),
            "log_step": lambda message: None,
            "log_file": lambda label, path: None,
        }
        if reporter is not None:
            context["reporter"] = reporter
        pipeline.run(context)
        return calls

    def test_minimax_uses_dspy_for_all_h3_modes(self):
        for video_pipeline in (
            "minimax-h3-t2v",
            "minimax-h3-i2v",
            "minimax-h3-fl2v",
            "minimax-h3-l2v",
            "minimax-h3-r2v",
        ):
            with self.subTest(video_pipeline=video_pipeline):
                calls = self._run_pipeline(video_pipeline)
                self.assertEqual(1, len(calls))
                expected_mode = video_pipeline.removeprefix("minimax-h3-")
                self.assertEqual("dspy", calls[0][0])
                self.assertEqual(expected_mode, calls[0][1]["mode"])

    def test_minimax_always_uses_dspy_and_non_minimax_keeps_legacy_builder(self):
        for video_pipeline in ("minimax-h3-r2v", "minimax-h3-t2v", "ltx_i2v"):
            with self.subTest(video_pipeline=video_pipeline):
                calls = self._run_pipeline(video_pipeline)
                self.assertEqual(1, len(calls))
                expected_builder = "legacy" if video_pipeline == "ltx_i2v" else "dspy"
                self.assertEqual(expected_builder, calls[0][0])

    def test_unknown_video_pipeline_fallback_is_reported(self):
        messages = []

        class FakeReporter:
            def message(self, text):
                messages.append(text)

        calls = self._run_pipeline("ltx_i2v", reporter=FakeReporter())

        self.assertEqual(1, len(calls))
        self.assertEqual("legacy", calls[0][0])
        self.assertEqual(1, len(messages))
        self.assertIn("ltx_i2v", messages[0])

    def test_minimax_pipeline_emits_no_fallback_warning(self):
        messages = []

        class FakeReporter:
            def message(self, text):
                messages.append(text)

        calls = self._run_pipeline("minimax-h3-i2v", reporter=FakeReporter())

        self.assertEqual(1, len(calls))
        self.assertEqual("dspy", calls[0][0])
        self.assertEqual(0, len(messages))
if __name__ == "__main__":
    unittest.main()
