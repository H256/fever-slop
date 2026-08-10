import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder, _scene_references
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


class DspyH3PromptBuilderTests(unittest.TestCase):
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
                return "http://llm.elysium.lan/v1"

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

        self.assertEqual("http://llm.elysium.lan/v1", lm_factory.call_args.kwargs["api_base"])
        self.assertFalse(lm_factory.call_args.kwargs["cache"])

    def test_generator_passes_dspy_cache_setting_to_lm(self):
        class Client:
            base_url = "http://llm.elysium.lan/v1"
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