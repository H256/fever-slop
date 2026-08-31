import unittest
from pathlib import Path

from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.config.project_config import (
    ActorConfig,
    ProjectConfig,
    StructuredLocationConfig,
)


class FakePromptPipeline:
    def __init__(self):
        self.subject_locations_calls = 0

    def create_story_idea(self, lyrics, notes):
        return "story"

    def create_style_block(self, lyrics, notes):
        return "style"

    def create_subject_and_locations(self, story_idea, notes):
        self.subject_locations_calls += 1
        return {
            "subject": "fallback subject",
            "actors": [{"id": "fallback_actor", "name": "Fallback"}],
            "locations": [{"id": "fallback_location", "name": "Fallback Location"}],
        }


class EnrichingPromptPipeline(FakePromptPipeline):
    def create_subject_and_locations(self, story_idea, notes):
        self.subject_locations_calls += 1
        self.last_notes = notes
        return {
            "subject": "fallback subject",
            "actors": [{
                "id": "singer",
                "name": "Lead Singer",
                "role": "generated role",
                "gender": "female",
                "visual_description": "short dark hair and a silver stage coat",
                "image_prompt": "cinematic portrait of a singer in a silver stage coat",
            }],
            "locations": [{"id": "fallback_location", "name": "Fallback Location"}],
        }


class PromptGenerationReferencesTests(unittest.TestCase):
    def test_resolved_global_context_includes_actors_and_structured_locations(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
            subject="legacy subject",
            locations=["Mirror Stage"],
            actors=(ActorConfig(
                id="singer", name="Mara", role="lead singer", gender="female",
                visual_description="short dark hair", image_prompt="portrait",
            ),),
            structured_locations=(StructuredLocationConfig(
                id="stage", name="Mirror Stage", visual_description="a mirror stage", image_prompt="stage",
            ),),
        )
        pipeline = PromptGenerationPipeline(
            llm_factory=lambda app_config: None,
            prompt_pipeline_factory=lambda llm: None,
            concept_batcher_factory=lambda llm, size: None,
            scene_prompt_builder_factory=lambda llm: None,
        )

        prompt_pipeline = FakePromptPipeline()
        context = pipeline.build_resolved_global_context(
            config=config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        self.assertEqual("legacy subject", context["subject"])
        self.assertEqual("singer", context["actors"][0]["id"])
        self.assertEqual("stage", context["structured_locations"][0]["id"])
        self.assertEqual("multi", context["subject_mode"])
        self.assertEqual(4, context["max_scene_actors"])
        self.assertNotIn("reference_profile", context)
        self.assertEqual("en", context["language"])
        self.assertEqual(0, prompt_pipeline.subject_locations_calls)

    def test_resolved_global_context_uses_llm_generated_actors_when_config_omits_them(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
        )
        pipeline = PromptGenerationPipeline(
            llm_factory=lambda app_config: None,
            prompt_pipeline_factory=lambda llm: None,
            concept_batcher_factory=lambda llm, size: None,
            scene_prompt_builder_factory=lambda llm: None,
        )

        context = pipeline.build_resolved_global_context(
            config=config,
            prompt_pipeline=FakePromptPipeline(),
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        self.assertEqual("fallback subject", context["subject"])
        self.assertEqual(["Fallback Location"], context["locations"])
        self.assertEqual("fallback_actor", context["actors"][0]["id"])
        self.assertEqual("fallback_location", context["structured_locations"][0]["id"])

    def test_single_subject_mode_context_limits_scene_actor_count(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
            subject_mode="single",
            max_scene_actors=1,
        )
        pipeline = PromptGenerationPipeline(
            llm_factory=lambda app_config: None,
            prompt_pipeline_factory=lambda llm: None,
            concept_batcher_factory=lambda llm, size: None,
            scene_prompt_builder_factory=lambda llm: None,
        )

        context = pipeline.build_resolved_global_context(
            config=config,
            prompt_pipeline=FakePromptPipeline(),
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        self.assertEqual("single", context["subject_mode"])
        self.assertEqual(1, context["max_scene_actors"])

    def test_partial_configured_actor_keeps_gender_and_role_but_gets_llm_creative_fields(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
            subject="male lead singer in a broken digital reality",
            locations=["Mirror Stage"],
            actors=(ActorConfig(id="singer", name="Lead Singer", role="lead vocalist", gender="male"),),
            structured_locations=(StructuredLocationConfig(id="stage", name="Mirror Stage"),),
        )
        pipeline = PromptGenerationPipeline(
            llm_factory=lambda app_config: None,
            prompt_pipeline_factory=lambda llm: None,
            concept_batcher_factory=lambda llm, size: None,
            scene_prompt_builder_factory=lambda llm: None,
        )
        prompt_pipeline = EnrichingPromptPipeline()

        context = pipeline.build_resolved_global_context(
            config=config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        actor = context["actors"][0]
        self.assertEqual(1, prompt_pipeline.subject_locations_calls)
        self.assertEqual("male", actor["gender"])
        self.assertEqual("lead vocalist", actor["role"])
        self.assertEqual("male short dark hair and a silver stage coat", actor["visual_description"])
        self.assertEqual("male cinematic portrait of a singer in a silver stage coat", actor["image_prompt"])
        self.assertIn("role=lead vocalist; gender=male", prompt_pipeline.last_notes)
