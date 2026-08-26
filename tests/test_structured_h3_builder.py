import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.dspy_h3_models import H3PromptSections, MusicIntent, PlannedShot, ResolvedPromptPlan
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder


class StructuredH3BuilderTests(unittest.TestCase):
    def test_sections_round_trip_preserves_typed_plan(self):
        plan = ResolvedPromptPlan(
            creative_intent="restrained performance",
            shots=[],
            overall_soundscape="wind",
            music_intent=MusicIntent.NONE,
        )
        sections = H3PromptSections.from_plan(plan)
        self.assertEqual(plan, sections.to_plan())

    def test_section_plan_is_compiled_without_model_prose(self):
        class SectionGenerator:
            def __call__(self, _request):
                class Result:
                    plan = ResolvedPromptPlan(
                        creative_intent="a restrained performance",
                        shots=[PlannedShot(
                            shot_number=1,
                            start_seconds=0,
                            end_seconds=4,
                            description="The singer raises the lantern.",
                        )],
                        overall_soundscape="wind",
                        music_intent=MusicIntent.NONE,
                    )
                return Result()

        result = DspyH3PromptBuilder(SectionGenerator()).build_h3_prompt(
            segment={"segment_id": "scene-01", "duration_seconds": 4},
            concept="ignored prose",
            scene_details={},
            global_context={},
            mode="ref",
        )
        self.assertIn("The singer raises the lantern.", result["prompt"])
        self.assertEqual("dspy_section_plan", result["prompt_provenance"]["source"])
        self.assertIn("sections", result)

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
