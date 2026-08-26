import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder


class StructuredH3BuilderTests(unittest.TestCase):
    def test_compiles_sections_without_calling_llm(self):
        class FailingGenerator:
            def __call__(self, _request):
                raise AssertionError("structured path must not call the legacy generator")

        builder = DspyH3PromptBuilder(FailingGenerator())
        result = builder.build_h3_prompt(
            segment={"segment_id": "scene-01", "duration_seconds": 5},
            concept="ignored legacy prose",
            scene_details={},
            global_context={},
            mode="ref",
            structured_sections={
                "facts": LockedSceneFacts.create(
                    scene_id="scene-01",
                    facts=[{"category": "wardrobe", "key": "hero", "value": "silver cloak", "source_id": "cast:hero"}],
                ),
                "shots": [CreativeShotPayload(
                    shot_id="shot-01",
                    visible_action="The singer raises the lantern.",
                    performance="restrained grief",
                    transition_intent="continue from the boundary frame",
                )],
                "shot_windows": {"shot-01": (0, 5)},
                "references": {"shot-01": ["<Picture 1>"]},
            },
        )

        self.assertIn("FULL REFERENCE PROMPT", result["prompt"])
        self.assertIn("The singer raises the lantern.", result["prompt"])
        self.assertEqual("deterministic_h3_compiler", result["prompt_provenance"]["compiler"])


if __name__ == "__main__":
    unittest.main()
