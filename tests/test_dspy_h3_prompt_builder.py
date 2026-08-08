import unittest
import tempfile
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder


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
    def test_reports_progress_after_each_scene(self):
        progress = []
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
            )

        self.assertEqual([(1, 2), (2, 2)], progress)

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
        self.assertEqual(result["prompt"], FakeGeneratedPrompt.rendered_prompt)

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


if __name__ == "__main__":
    unittest.main()