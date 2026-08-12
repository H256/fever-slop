import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.dspy_h3_prompt_builder import (
    DspyH3PromptBuilder,
    _format_relay_shots,
    _normalize_relay_segments,
    _scene_references,
)
from feverslop.adapters.movie_minimax_visual import _h3_movie_prompt
from feverslop.prompting.dspy_h3_analyzer import LocalImageAnalyzer
from feverslop.prompting.dspy_h3_generator import VideoPromptGenerator
from feverslop.prompting.dspy_h3_models import (
    PromptMode,
    ReferenceAsset,
    ReferenceKind,
    ReferenceLimits,
    ReferenceVideoPrompt,
)
from feverslop.prompting.dspy_h3_generator_core import VideoPromptGenerator as CoreVideoPromptGenerator
from feverslop.prompting.dspy_h3_signatures import build_dspy_signatures


class FakeGeneratedPrompt:
    rendered_prompt = "subject_definitions: <Subject 1>\ndetailed_description: test"


class FakeGenerator:
    def __init__(self, result=None):
        self.requests = []
        self.result = result or FakeGeneratedPrompt()

    def __call__(self, request):
        self.requests.append(request)
        return self.result


class IncompleteAudioPrompt:
    rendered_prompt = """subject_definitions:
<Subject 1> is a singer.

summary: [reference generation] A singer performs.

retention_analysis:
<Subject 1>: fully_preserved - The singer remains recognizable.

detailed_description: <Subject 1> sings in a close-up.

overall_soundscape: A quiet room tone.

non_diegetic_music: N/A"""


class DspyH3PromptBuilderTests(unittest.TestCase):
    def test_passes_general_steering_and_prompt_guidance_to_generator(self):
        generator = FakeGenerator()
        DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={"segment_id": "seg-1", "duration_seconds": 2, "fps": 24},
            concept="A singer performs.",
            scene_details={},
            global_context={
                "story_idea": "A singer crosses a mountain.",
                "style": "Cinematic dark fantasy.",
                "subject": "The same singer throughout.",
                "steering": {"global": "Use only the configured locations."},
                "prompt_guidance": {"camera_motion": "Use deliberate tracking shots."},
            },
            mode="base",
        )

        notes = generator.requests[0]["notes"]
        self.assertIn("Use only the configured locations.", notes)
        self.assertIn("Use deliberate tracking shots.", notes)
        self.assertIn("A singer crosses a mountain.", notes)
    def test_minimax_movie_prompt_preserves_r2v_prompt_and_adds_relay_shots(self):
        prompt = _h3_movie_prompt({
            "h3": {"prompt": "Use <Picture 1> for the actor."},
            "duration_seconds": 6.4,
            "fps": 24,
            "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 154, "prompt": "The actor looks left."}]},
            "references": {"actor_msr_paths": ["actors/bard_woman/views/msr_sheet.png"]},
        })

        self.assertIn("<Picture 1>", prompt)
        self.assertIn("actors/bard_woman/views/msr_sheet.png", prompt)
        self.assertIn("[Shot 1, 0.00-6.40sec]", prompt)

    def test_normalizes_relay_frames_to_timed_shots(self):
        segment = {
            "duration_seconds": 6.4,
            "fps": 24,
            "ltx": {
                "prompt_relay": [
                    {"frame_start": 36, "frame_end": 153, "state": "vocals", "prompt": "The singer turns."},
                    {"frame_start": 153, "frame_end": 240, "state": "instrumental", "prompt": "The camera pulls back."},
                ]
            },
        }

        self.assertEqual(
            [
                {"shot": 1, "start_seconds": 1.5, "end_seconds": 6.375, "state": "vocals", "prompt": "The singer turns."},
                {"shot": 2, "start_seconds": 6.375, "end_seconds": 6.4, "state": "instrumental", "prompt": "The camera pulls back."},
            ],
            _normalize_relay_segments(segment),
        )

    def test_formats_relay_shots_with_minimax_syntax(self):
        shots = [
            {
                "shot": 1,
                "start_seconds": 1.5,
                "end_seconds": 6.4,
                "state": "vocals",
                "prompt": "The singer turns.",
                "source_prompt": "The singer cooks a rabbit over the fire.",
            },
        ]

        formatted = _format_relay_shots(shots)
        self.assertIn("[Shot 1, 1.50-6.40sec] (vocals) The singer turns.", formatted)
        self.assertIn("Required action and props to preserve: The singer cooks a rabbit over the fire.", formatted)

    def test_normalizes_image_like_relay_without_losing_source_action(self):
        shots = _normalize_relay_segments({
            "duration_seconds": 4,
            "fps": 24,
            "ltx": {"prompt_relay": [{
                "frame_start": 0,
                "frame_end": 96,
                "prompt": "Cinematic close-up, warm firelight, the singer cooks a rabbit over the fire.",
                "source_prompt": "Cinematic close-up, warm firelight, the singer cooks a rabbit over the fire.",
            }]},
        })

        self.assertIn("cooks a rabbit", _format_relay_shots(shots))

    def test_passes_relay_segments_to_generator_and_appends_timed_shots(self):
        generator = FakeGenerator()
        builder = DspyH3PromptBuilder(generator)

        result = builder.build_h3_prompt(
            segment={
                "segment_id": "seg-1",
                "type": "vocals",
                "duration_seconds": 6.4,
                "fps": 24,
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 154, "prompt": "The singer turns."}]},
                "references": {"actor_sheet_paths": ["actor.png"]},
            },
            concept="A singer performs.",
            scene_details={},
            global_context={},
            mode="ref",
        )

        self.assertIn("relay_segments_json", generator.requests[0])
        self.assertIn('"start_seconds": 0.0', generator.requests[0]["relay_segments_json"])
        self.assertIn("[Shot 1, 0.00-6.40sec]", result["prompt"])
        self.assertIn("actor.png", " ".join(reference["source"] for reference in result["references"]))

    def test_repairs_missing_audio_reuse_references_in_all_required_sections(self):
        builder = DspyH3PromptBuilder(FakeGenerator(IncompleteAudioPrompt()))

        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "vocals"},
            concept="A singer performs.",
            scene_details={},
            global_context={},
            mode="ref",
            audio_paths={
                "vocals": Path("output/stems/vocals.wav"),
                "full_mix": Path("input/song.mp3"),
            },
        )

        prompt = result["prompt"]
        self.assertIn("<Audio 1> is the synchronized vocals audio reference and is reused for the scene.", prompt)
        self.assertIn("<Audio 2> is the synchronized full_mix audio reference and is reused for the scene.", prompt)
        self.assertIn("[reference generation + audio reuse]", prompt)
        self.assertIn("<Audio 1>: partially_copy", prompt)
        self.assertIn("<Audio 2>: partially_copy", prompt)
        self.assertIn("<Audio 1> and <Audio 2>", prompt)
        self.assertIn("overall_soundscape: A quiet room tone. The synchronized audio behavior follows <Audio 1> and <Audio 2>.", prompt)
        self.assertIn("non_diegetic_music: The synchronized audio references are scene inputs, not non-diegetic music: <Audio 1> and <Audio 2>.", prompt)

    def test_repairs_existing_audio_labels_with_wrong_role(self):
        weak = IncompleteAudioPrompt.rendered_prompt.replace(
            "<Subject 1> is a singer.",
            "<Subject 1> is a singer.\n<Audio 1> is a reference track.\n<Audio 2> is a reference track.",
        ).replace(
            "[reference generation]",
            "[reference generation + audio reuse]",
        ).replace(
            "<Subject 1>: fully_preserved - The singer remains recognizable.",
            "<Subject 1>: fully_preserved - The singer remains recognizable.\n<Audio 1>: reference - do not copy.\n<Audio 2>: reference - do not copy.",
        )
        generated = type("Generated", (), {"rendered_prompt": weak})()
        builder = DspyH3PromptBuilder(FakeGenerator(generated))

        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "instrumental"},
            concept="A singer performs.",
            scene_details={},
            global_context={},
            audio_paths={"vocals": Path("output/stems/vocals.wav"), "full_mix": Path("input/song.mp3")},
        )

        prompt = result["prompt"]
        self.assertIn("<Audio 1> is the synchronized vocals audio reference and is reused for the scene.", prompt)
        self.assertIn("<Audio 2>: partially_copy - the synchronized full_mix audio is reused for this scene.", prompt)
        self.assertNotIn("is a reference track.", prompt)
        self.assertNotIn("do not copy.", prompt)

    def test_keeps_complete_audio_references_unchanged(self):
        complete = IncompleteAudioPrompt.rendered_prompt.replace(
            "<Subject 1> is a singer.",
            "<Subject 1> is a singer.\n<Audio 1> is the synchronized vocals audio reference and is reused for the scene.\n<Audio 2> is the synchronized full_mix audio reference and is reused for the scene.",
        ).replace(
            "[reference generation]",
            "[reference generation + audio reuse] A singer performs using <Audio 1> and <Audio 2>",
        ).replace(
            "<Subject 1>: fully_preserved - The singer remains recognizable.",
            "<Subject 1>: fully_preserved - The singer remains recognizable.\n<Audio 1>: partially_copy - retained.\n<Audio 2>: partially_copy - retained.",
        ).replace(
            "<Subject 1> sings in a close-up.",
            "<Subject 1> sings in a close-up with <Audio 1> and <Audio 2>.",
        ).replace(
            "A quiet room tone.",
            "A quiet room tone with <Audio 1> and <Audio 2>. The synchronized audio behavior follows <Audio 1> and <Audio 2>.",
        ).replace(
            "non_diegetic_music: N/A",
            "non_diegetic_music: The synchronized audio references are scene inputs, not non-diegetic music: <Audio 1> and <Audio 2>.",
        )
        generated = type("Generated", (), {"rendered_prompt": complete})()
        builder = DspyH3PromptBuilder(FakeGenerator(generated))

        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "vocals"},
            concept="A singer performs.",
            scene_details={},
            global_context={},
            mode="ref",
            audio_paths={"vocals": Path("output/stems/vocals.wav"), "full_mix": Path("input/song.mp3")},
        )

        self.assertEqual(complete, result["prompt"])

    def test_reference_prompt_accepts_audio_tags_in_non_diegetic_music(self):
        prompt = ReferenceVideoPrompt(
            subject_definitions=[],
            summary="A scene.",
            retention_analysis=[],
            detailed_description="A detailed scene.",
            overall_soundscape="The song is audible.",
            non_diegetic_music=(
                "N/A\n"
                "<Audio 1> (audio_transfer - vocal singing lip-synced to the audio signal)\n"
                "<Audio 2> (full_mix - original song for beat and rhythm continuity)"
            ),
        )

        self.assertIn("<Audio 1>", prompt.non_diegetic_music)

    def test_image_analysis_is_cached_for_repeated_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "reference.png"
            image_path.write_bytes(b"image")
            calls = []

            class Analysis:
                objective_description = "A reference image"
                visible_subjects = []
                environment = ""
                visual_style = ""
                composition = ""
                lighting = ""
                visible_text = []

            def predictor(**kwargs):
                calls.append(kwargs)
                return type("Prediction", (), {"analysis": Analysis()})()

            reference = ReferenceAsset(
                kind=ReferenceKind.PICTURE,
                source=str(image_path),
                role="subject",
            )
            analyzer = LocalImageAnalyzer(predictor)

            first = analyzer.analyze(reference)
            second = analyzer.analyze(reference)

        self.assertEqual("A reference image", first)
        self.assertEqual(first, second)
        self.assertEqual(1, len(calls))

    def test_generator_converts_openai_url_object_for_dspy(self):
        class UrlObject:
            def __str__(self):
                return "http://your-llm-server.local/v1"

        class Client:
            base_url = UrlObject()
            api_key = "none-needed"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            temperature = 0.75
            max_tokens = 16384

        guides = Path(__file__).parents[1] / "src" / "feverslop" / "prompting" / "guides"
        with patch("dspy.LM") as lm_factory:
            VideoPromptGenerator(
                base_guide_path=guides / "minimax-h3-base.md",
                reference_guide_path=guides / "minimax-h3-references.md",
                llm=LLM(),
            )

        self.assertEqual("http://your-llm-server.local/v1", lm_factory.call_args.kwargs["api_base"])
        self.assertFalse(lm_factory.call_args.kwargs["cache"])

    def test_generator_passes_dspy_cache_setting_to_lm(self):
        class Client:
            base_url = "http://your-llm-server.local/v1"
            api_key = "none-needed"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            temperature = 0.75
            max_tokens = 16384
            dspy_cache = True

        guides = Path(__file__).parents[1] / "src" / "feverslop" / "prompting" / "guides"
        with patch("dspy.LM") as lm_factory:
            VideoPromptGenerator(
                base_guide_path=guides / "minimax-h3-base.md",
                reference_guide_path=guides / "minimax-h3-references.md",
                llm=LLM(),
            )

        self.assertTrue(lm_factory.call_args.kwargs["cache"])

    def test_generator_passes_dspy_temperature_to_lm(self):
        class Client:
            base_url = "http://your-llm-server.local/v1"
            api_key = "none-needed"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            temperature = 0.75
            dspy_temperature = 0.25
            max_tokens = 16384
            dspy_cache = False

        guides = Path(__file__).parents[1] / "src" / "feverslop" / "prompting" / "guides"
        with patch("dspy.LM") as lm_factory:
            VideoPromptGenerator(
                base_guide_path=guides / "minimax-h3-base.md",
                reference_guide_path=guides / "minimax-h3-references.md",
                llm=LLM(),
            )

        self.assertEqual(0.25, lm_factory.call_args.kwargs["temperature"])

    def test_reference_limits_use_plural_picture_field(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        generator.limits = ReferenceLimits()
        generator.image_analyzer = type("Analyzer", (), {"should_analyze": lambda *_: False})()

        resolved = generator._resolve_references([
            ReferenceAsset(kind=ReferenceKind.PICTURE, source="actor.png", description="actor"),
            ReferenceAsset(kind=ReferenceKind.PICTURE, source="location.png", description="location"),
        ])

        self.assertEqual([item.label for item in resolved], ["<Picture 1>", "<Picture 2>"])

    def test_generator_components_have_dedicated_modules(self):
        self.assertEqual(VideoPromptGenerator.__module__, "feverslop.prompting.dspy_h3_generator")
        self.assertEqual(LocalImageAnalyzer.__module__, "feverslop.prompting.dspy_h3_analyzer")
        self.assertEqual(PromptMode.__module__, "feverslop.prompting.dspy_h3_models")
        self.assertTrue(callable(build_dspy_signatures))
        self.assertEqual(ReferenceAsset.__module__, "feverslop.prompting.dspy_h3_models")

    def test_dspy_signatures_resolve_nested_pydantic_output_types(self):
        analyze_image, *_ = build_dspy_signatures()

        self.assertIs(
            analyze_image.output_fields["analysis"].annotation,
            __import__("feverslop.prompting.dspy_h3_models", fromlist=["ImageAnalysis"]).ImageAnalysis,
        )

    def test_relay_signature_inputs_are_optional_with_empty_array_default(self):
        _, build_plan, render_base, render_reference = build_dspy_signatures()

        for signature in (build_plan, render_base, render_reference):
            field = signature.input_fields["relay_segments_json"]
            self.assertFalse(field.is_required())
            self.assertEqual(field.default, "[]")

    def test_integrated_guides_are_bundled_with_prompting_package(self):
        guides = Path(__file__).parents[1] / "src" / "feverslop" / "prompting" / "guides"

        base = (guides / "minimax-h3-base.md").read_text(encoding="utf-8")
        reference = (guides / "minimax-h3-references.md").read_text(encoding="utf-8")

        self.assertIn("integrated_multimodal_description", base)
        self.assertIn("subject_definitions", reference)
        self.assertIn("retention_analysis", reference)

        self.assertTrue((guides / "krea-actor.md").is_file())
        self.assertTrue((guides / "krea-location.md").is_file())

    def test_scene_reference_roles_are_preserved_for_full_generator(self):
        references, _ = _scene_references(
            {
                "references": {
                    "actor_ids": ["actor"],
                    "actor_msr_paths": ["actor.png"],
                    "location_msr_path": "location.png",
                }
            },
            {"vocals": Path("vocals.wav")},
            None,
        )

        self.assertEqual(
            [reference["role"] for reference in references],
            ["subject", "environment", "audio_reuse"],
        )

    def test_reports_progress_after_each_scene(self):
        progress = []
        statuses = []
        builder = DspyH3PromptBuilder(FakeGenerator())

        with tempfile.TemporaryDirectory() as temp_dir:
            builder.build_all_h3_prompts(
                stage1_segments=[
                    {"segment_id": "seg-1", "type": "vocals"},
                    {"segment_id": "seg-2", "type": "instrumental"},
                ],
                concept_prompts={"seg-1": "one", "seg-2": "two"},
                scene_details={},
                global_context={},
                output_json_path=Path(temp_dir) / "h3.json",
                artifact_store=JsonArtifactStore(),
                progress_callback=lambda current, total: progress.append((current, total)),
                status_callback=lambda current, total, status: statuses.append((current, total, status)),
            )

        self.assertEqual([(1, 2), (2, 2)], progress)
        self.assertEqual(
            [(1, 2, "started"), (1, 2, "completed"), (2, 2, "started"), (2, 2, "completed")],
            statuses,
        )

    def test_maps_scene_references_and_audio_stems_to_generator_request(self):
        generator = FakeGenerator()
        builder = DspyH3PromptBuilder(generator)

        result = builder.build_h3_prompt(
            segment={
                "segment_id": "seg-1",
                "type": "vocals",
                "lyrics": "Ein Lied",
                "references": {
                    "actor_ids": ["elara"],
                    "location_id": "tavern",
                    "actor_msr_paths": ["movie/references/elara.png"],
                    "location_msr_path": "movie/references/tavern.png",
                },
            },
            concept="A singer in a tavern",
            scene_details={"camera_motion": "slow push in"},
            global_context={"style": "cinematic", "story_idea": "loss"},
            mode="ref",
            audio_paths={"vocals": Path("output/stems/vocals.wav")},
        )

        request = generator.requests[0]
        references = request["references"]
        self.assertEqual([ref["source"] for ref in references[:2]], [
            "movie/references/elara.png",
            "movie/references/tavern.png",
        ])
        self.assertEqual([ref["label"] for ref in references[:2]], [
            "<Picture 1>",
            "<Picture 2>",
        ])
        self.assertEqual(references[2]["label"], "<Audio 1>")
        self.assertEqual(request["music_intent"], "none")
        self.assertEqual(result["prompt"], FakeGeneratedPrompt.rendered_prompt)

    def test_does_not_force_music_mode_without_scene_audio(self):
        generator = FakeGenerator()

        DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={"segment_id": "seg-1"},
            concept="A silent scene",
            scene_details={},
            global_context={},
        )

        self.assertNotIn("music_intent", generator.requests[0])

    def test_resolves_existing_picture_paths_only_for_generator(self):
        generator = FakeGenerator()
        builder = DspyH3PromptBuilder(generator, reference_root=Path.cwd())

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            picture = root / "output" / "actor.png"
            picture.parent.mkdir()
            picture.write_bytes(b"not-an-image")
            builder = DspyH3PromptBuilder(generator, reference_root=root)
            result = builder.build_h3_prompt(
                segment={
                    "segment_id": "seg-1",
                    "references": {"actor_msr_paths": ["output/actor.png"]},
                },
                concept="A singer",
                scene_details={},
                global_context={},
            )

        self.assertEqual(generator.requests[-1]["references"][0]["source"], str(picture))
        self.assertEqual(result["references"][0]["source"], "output/actor.png")

    def test_falls_back_to_existing_prompt_when_generator_fails(self):
        class BrokenGenerator:
            def __call__(self, request):
                raise RuntimeError("DSPy unavailable")

        builder = DspyH3PromptBuilder(BrokenGenerator())
        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "instrumental"},
            concept="fallback scene",
            scene_details={},
            global_context={},
            mode="ref",
        )

        self.assertEqual(result["prompt"], "fallback scene")
        self.assertEqual(result["dspy_error"], "DSPy unavailable")

    def test_sanitizes_embedded_image_data_in_fallback_error(self):
        payload = "data:image/png;base64," + ("A" * 400)

        class BrokenGenerator:
            def __call__(self, request):
                raise RuntimeError(payload)

        result = DspyH3PromptBuilder(BrokenGenerator()).build_h3_prompt(
            segment={"segment_id": "seg-1"},
            concept="fallback scene",
            scene_details={},
            global_context={},
        )

        self.assertNotIn("data:image", result["dspy_error"])
        self.assertNotIn("A" * 100, result["dspy_error"])
        self.assertIn("embedded image omitted", result["dspy_error"])

    def test_production_mode_does_not_hide_dspy_failure(self):
        class BrokenGenerator:
            def __call__(self, request):
                raise RuntimeError("connection refused")

        with self.assertRaisesRegex(RuntimeError, "DSPy H3 generation failed: connection refused"):
            DspyH3PromptBuilder(BrokenGenerator(), allow_fallback=False).build_h3_prompt(
                segment={"segment_id": "seg-1"},
                concept="fallback scene",
                scene_details={},
                global_context={},
            )


if __name__ == "__main__":
    unittest.main()
