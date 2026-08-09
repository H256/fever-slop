import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.application.h3_prompt_pipeline import _configured_audio_paths
from feverslop.composition.arg_parser import build_arg_parser
from feverslop.adapters.pipeline_runner_options import runner_options_from_args


class ConfiguredAudioPathTests(unittest.TestCase):
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
    def _run_pipeline(self, video_pipeline, *, use_dspy_prompts=False):
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
            "request": SimpleNamespace(use_dspy_prompts=use_dspy_prompts),
            "stage1_segments": [{"segment_id": "s1"}],
            "concept_prompts": {},
            "scene_details": {},
            "global_context": {},
            "h3_prompts_json": "test.json",
            "artifact_store": FakeArtifactStore(),
            "log_step": lambda message: None,
            "log_file": lambda label, path: None,
        }
        pipeline.run(context)
        return calls

    def test_explicit_option_selects_dspy_for_all_minimax_h3_modes(self):
        for video_pipeline in (
            "minimax-h3-t2v",
            "minimax-h3-i2v",
            "minimax-h3-fl2v",
            "minimax-h3-l2v",
            "minimax-h3-r2v",
        ):
            with self.subTest(video_pipeline=video_pipeline):
                calls = self._run_pipeline(video_pipeline, use_dspy_prompts=True)
                self.assertEqual(1, len(calls))
                expected_mode = "ref" if video_pipeline.endswith("-r2v") else "base"
                self.assertEqual("dspy", calls[0][0])
                self.assertEqual(expected_mode, calls[0][1]["mode"])

    def test_minimax_always_uses_dspy_and_non_minimax_keeps_legacy_builder(self):
        for video_pipeline in ("minimax-h3-r2v", "minimax-h3-t2v", "ltx_i2v"):
            with self.subTest(video_pipeline=video_pipeline):
                calls = self._run_pipeline(video_pipeline)
                self.assertEqual(1, len(calls))
                expected_builder = "legacy" if video_pipeline == "ltx_i2v" else "dspy"
                self.assertEqual(expected_builder, calls[0][0])


class DspyPromptArgumentTests(unittest.TestCase):
    def test_use_dspy_prompts_flag_maps_to_runner_options(self):
        args = build_arg_parser().parse_args(["--use-dspy-prompts"])
        self.assertTrue(args.use_dspy_prompts)
        self.assertTrue(runner_options_from_args(args)["use_dspy_prompts"])

    def test_use_dspy_prompts_defaults_to_false(self):
        args = build_arg_parser().parse_args([])
        self.assertFalse(args.use_dspy_prompts)


if __name__ == "__main__":
    unittest.main()