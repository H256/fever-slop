import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.dspy_h3_models import (
    H3PromptSections,
    MusicIntent,
    PlannedShot,
    PromptJudgeResult,
    ReferenceUsage,
    ResolvedPromptPlan,
    SubjectDefinition,
)
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder


class StructuredH3BuilderTests(unittest.TestCase):
    @staticmethod
    def rich_description() -> str:
        sentence = (
            "The composition, subject position, appearance, environment, lighting, visible "
            "action, physical state change, camera movement, and current sound are explicit."
        )
        return " ".join([sentence] * 32)

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
            judge_attempts = 1

            def __call__(self, _request):
                class Result:
                    plan = ResolvedPromptPlan(
                        creative_intent="a restrained performance",
                        style_opening="Live-action cinematic imagery uses cool practical lighting.",
                        shots=[PlannedShot(
                            shot_number=1,
                            start_seconds=0,
                            end_seconds=4,
                            description=StructuredH3BuilderTests.rich_description()
                            + " The singer raises the lantern.",
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
        self.assertEqual(10, result["prompt_provenance"]["compiler_version"])

    def test_checkpoint_revision_tracks_the_deterministic_compiler(self):
        builder = DspyH3PromptBuilder(lambda _request: None)

        revision = builder.checkpoint_revision()

        self.assertEqual("deterministic_h3_compiler", revision["compiler"])
        self.assertEqual(10, revision["compiler_version"])

    def test_resume_recompiles_saved_plan_with_guide_compiler_and_rejudges(self):
        class JudgeOnlyGenerator:
            def __init__(self):
                self.judged_prompts = []
                self.judged_references = []
                self.judged_plans = []

            def __call__(self, _request):
                raise AssertionError("resume recompile must not regenerate creative fields")

            def judge_compiled_prompt(self, **kwargs):
                self.judged_prompts.append(kwargs["final_prompt"])
                self.judged_references.append(kwargs["references"])
                self.judged_plans.append(kwargs["plan"])
                return PromptJudgeResult(verdict="good", issues=[])

        generator = JudgeOnlyGenerator()
        sections = H3PromptSections(
            creative_intent="A restrained performance.",
            style_opening="Live-action cinematic imagery uses cool practical lighting.",
            subjects=[SubjectDefinition(
                label="<Subject 1>",
                name="the singer",
                description="a woman in a silver cloak",
                source_references=["<Picture 1>"],
            )],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 1>",
                purpose="audio reference",
                details="Use the original song for beat and rhythm continuity.",
            )],
            shots=[PlannedShot(
                shot_number=1,
                start_seconds=0,
                end_seconds=5,
                description=self.rich_description() + " The singer raises the lantern.",
                reference_labels=["<Subject 1>", "<Picture 1>", "<Audio 1>"],
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
                }, {
                    "label": "<Audio 1>",
                    "source": "song.wav",
                    "kind": "audio",
                    "name": "full_mix",
                    "description": "full_mix - original song for beat and rhythm continuity",
                    "role": "audio_reuse",
                    "copy_mode": "fully_copy",
                }],
            },
        )

        self.assertTrue(result["prompt"].startswith("subject_definitions:"))
        self.assertEqual("good", result["prompt_judge"]["verdict"])
        self.assertEqual([result["prompt"]], generator.judged_prompts)
        self.assertEqual("reference", result["references"][1]["copy_mode"])
        self.assertEqual("reference", generator.judged_references[0][1]["copy_mode"])
        [audio_usage] = generator.judged_plans[0].reference_usage
        self.assertEqual("audio reference", audio_usage.purpose)
        self.assertIn("without copying", audio_usage.details)

    def test_resume_regenerates_when_saved_plan_fails_current_guide_contract(self):
        from pathlib import Path
        from types import SimpleNamespace
        from tempfile import TemporaryDirectory

        from feverslop.adapters.local_artifacts import JsonArtifactStore

        short_sections = H3PromptSections(
            creative_intent="Too short.",
            style_opening="Live-action cinematic imagery uses cool practical lighting.",
            shots=[PlannedShot(
                shot_number=1,
                start_seconds=0,
                end_seconds=5,
                description="A singer turns.",
            )],
            overall_soundscape="Room tone.",
            music_intent=MusicIntent.NONE,
        )
        rich_plan = short_sections.to_plan().model_copy(update={
            "creative_intent": "A complete performance.",
            "shots": [short_sections.shots[0].model_copy(update={
                "description": self.rich_description(),
            })],
        })

        class Generator:
            judge_attempts = 1

            def __init__(self):
                self.calls = 0

            def __call__(self, _request):
                self.calls += 1
                return SimpleNamespace(plan=rich_plan)

            def judge_compiled_prompt(self, **_kwargs):
                return PromptJudgeResult(verdict="good", issues=[])

        class StaleStore:
            def __init__(self):
                self.saved = []

            def load(self, _request):
                return None

            def load_for_resume(self, _request):
                return SimpleNamespace(generated={
                    "sections": {
                        "h3_sections": short_sections.model_dump(),
                        "facts": LockedSceneFacts.create(scene_id="scene-01", facts=[]).to_dict(),
                    },
                    "references": [],
                })

            def invalidated_stages(self, _request, _checkpoint):
                return frozenset({"compiler"})

            def save(self, _request, generated):
                self.saved.append(generated)

        generator = Generator()
        store = StaleStore()
        statuses = []
        with TemporaryDirectory() as temp_dir:
            DspyH3PromptBuilder(generator, allow_fallback=False).build_all_h3_prompts(
                stage1_segments=[{"scene": 1, "segment_id": "scene-01", "duration": 5}],
                concept_prompts={"scene-01": "A complete performance."},
                scene_details={},
                global_context={"language": "English"},
                mode="r2v",
                output_json_path=Path(temp_dir) / "h3.json",
                artifact_store=JsonArtifactStore(),
                checkpoint_store=store,
                generator_revision={"compiler_version": 8},
                status_callback=lambda current, total, status: statuses.append(status),
            )

        self.assertEqual(1, generator.calls)
        self.assertIn("regenerating", statuses)
        self.assertEqual("dspy_section_plan", store.saved[-1]["prompt_provenance"]["source"])

    def test_resume_regenerates_when_recompiled_plan_is_rejected_by_judge(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from feverslop.adapters.local_artifacts import JsonArtifactStore

        sections = H3PromptSections(
            creative_intent="A complete performance.",
            style_opening="Live-action cinematic imagery uses cool practical lighting.",
            shots=[PlannedShot(
                shot_number=1,
                start_seconds=0,
                end_seconds=5,
                description=self.rich_description(),
            )],
            overall_soundscape="Room tone.",
            music_intent=MusicIntent.NONE,
        )

        class Generator:
            judge_attempts = 1

            def __init__(self):
                self.calls = 0
                self.judges = 0

            def __call__(self, _request):
                self.calls += 1
                return SimpleNamespace(plan=sections.to_plan())

            def judge_compiled_prompt(self, **_kwargs):
                self.judges += 1
                if self.judges == 1:
                    return PromptJudgeResult(verdict="bad", issues=["creative detail mismatch"])
                return PromptJudgeResult(verdict="good", issues=[])

        class StaleStore:
            def __init__(self):
                self.saved = []

            def load(self, _request):
                return None

            def load_for_resume(self, _request):
                return SimpleNamespace(generated={
                    "sections": {
                        "h3_sections": sections.model_dump(),
                        "facts": LockedSceneFacts.create(scene_id="scene-01", facts=[]).to_dict(),
                    },
                    "references": [],
                })

            def invalidated_stages(self, _request, _checkpoint):
                return frozenset({"compiler"})

            def save(self, _request, generated):
                self.saved.append(generated)

        generator = Generator()
        store = StaleStore()
        statuses = []
        with TemporaryDirectory() as temp_dir:
            DspyH3PromptBuilder(generator, allow_fallback=False).build_all_h3_prompts(
                stage1_segments=[{"scene": 1, "segment_id": "scene-01", "duration": 5}],
                concept_prompts={"scene-01": "A complete performance."},
                scene_details={},
                global_context={"language": "English"},
                mode="r2v",
                output_json_path=Path(temp_dir) / "h3.json",
                artifact_store=JsonArtifactStore(),
                checkpoint_store=store,
                generator_revision={"compiler_version": 8},
                status_callback=lambda current, total, status: statuses.append(status),
            )

        self.assertEqual(1, generator.calls)
        self.assertEqual(2, generator.judges)
        self.assertIn("regenerating", statuses)
        self.assertEqual("good", store.saved[-1]["prompt_judge"]["verdict"])


if __name__ == "__main__":
    unittest.main()
