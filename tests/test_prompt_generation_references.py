import unittest
from pathlib import Path

from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.config.project_config import ActorConfig, ProjectConfig, StructuredLocationConfig


class FakePromptPipeline:
    def create_story_idea(self, lyrics, notes):
        return "story"

    def create_style_block(self, lyrics, notes):
        return "style"

    def create_subject_and_locations(self, story_idea, notes):
        return {
            "subject": "fallback subject",
            "actors": [{"id": "fallback_actor", "name": "Fallback"}],
            "locations": [{"id": "fallback_location", "name": "Fallback Location"}],
        }


class PromptGenerationReferencesTests(unittest.TestCase):
    def test_resolved_global_context_includes_actors_and_structured_locations(self):
        config = ProjectConfig(
            project_dir=Path("."),
            project_name="test",
            input_audio=Path("song.mp3"),
            subject="legacy subject",
            locations=["Mirror Stage"],
            actors=(ActorConfig(id="singer", name="Mara", image_prompt="portrait"),),
            structured_locations=(StructuredLocationConfig(id="stage", name="Mirror Stage", image_prompt="stage"),),
            reference_profile="live_concert",
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

        self.assertEqual("legacy subject", context["subject"])
        self.assertEqual("singer", context["actors"][0]["id"])
        self.assertEqual("stage", context["structured_locations"][0]["id"])
        self.assertEqual("multi", context["subject_mode"])
        self.assertEqual(4, context["max_scene_actors"])
        self.assertEqual("live_concert", context["reference_profile"])
        self.assertEqual("en", context["language"])

    def test_resolved_global_context_uses_llm_generated_actors_when_config_omits_them(self):
        config = ProjectConfig(
            project_dir=Path("."),
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
            project_dir=Path("."),
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
