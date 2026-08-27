import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.dspy_h3_models import (
    H3PromptSections,
    MusicIntent,
    PlannedShot,
    PromptJudgeResult,
    ResolvedPromptPlan,
    SubjectDefinition,
)
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
        self.assertEqual(7, result["prompt_provenance"]["compiler_version"])

    def test_checkpoint_revision_tracks_the_deterministic_compiler(self):
        builder = DspyH3PromptBuilder(lambda _request: None)

        revision = builder.checkpoint_revision()

        self.assertEqual("deterministic_h3_compiler", revision["compiler"])
        self.assertEqual(7, revision["compiler_version"])

    def test_resume_recompiles_saved_plan_with_guide_compiler_and_rejudges(self):
        class JudgeOnlyGenerator:
            def __init__(self):
                self.judged_prompts = []

            def __call__(self, _request):
                raise AssertionError("resume recompile must not regenerate creative fields")

            def judge_compiled_prompt(self, **kwargs):
                self.judged_prompts.append(kwargs["final_prompt"])
                return PromptJudgeResult(verdict="good", issues=[])

        generator = JudgeOnlyGenerator()
        sections = H3PromptSections(
            creative_intent="A restrained performance.",
            subjects=[SubjectDefinition(
                label="<Subject 1>",
                name="the singer",
                description="a woman in a silver cloak",
                source_references=["<Picture 1>"],
            )],
            shots=[PlannedShot(
                shot_number=1,
                start_seconds=0,
                end_seconds=5,
                description="The singer raises the lantern.",
                reference_labels=["<Subject 1>", "<Picture 1>"],
            )],
            overall_soundscape="Wind moves through the room.",
            music_intent=MusicIntent.NONE,
        )

        result = DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={"segment_id": "scene-01", "duration_seconds": 5},
            concept="ignored legacy prose",
            scene_details={},
            global_context={"language": "English"},
            mode="r2v",
            structured_sections={
                "h3_sections": sections.model_dump(),
                "facts": LockedSceneFacts.create(scene_id="scene-01", facts=[]).to_dict(),
                "resolved_references": [{
                    "label": "<Picture 1>",
                    "source": "singer.png",
                    "kind": "picture",
                    "name": "the singer",
                    "description": "a woman in a silver cloak",
                    "role": "subject",
                }],
            },
        )

        self.assertTrue(result["prompt"].startswith("subject_definitions:"))
        self.assertEqual("good", result["prompt_judge"]["verdict"])
        self.assertEqual([result["prompt"]], generator.judged_prompts)


if __name__ == "__main__":
    unittest.main()
