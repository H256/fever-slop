import unittest
from pathlib import Path

from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.config.project_config import ActorConfig, ProjectConfig
from feverslop.errors import FeverSlopValidationError


class GenericBandPromptPipeline:
    """Fake prompt pipeline that returns generic 'A person' text per actor."""

    def __init__(self, actors):
        self.actors = actors
        self.subject_locations_calls = 0
        self.last_notes = ""

    def create_story_idea(self, lyrics, notes):
        return "a female lead singer performs with a band on a mirror stage"

    def create_style_block(self, lyrics, notes):
        return "cinematic concert film"

    def create_subject_and_locations(self, story_idea, notes):
        self.subject_locations_calls += 1
        self.last_notes = notes
        return {
            "subject": "a band plays a mirror stage",
            "actors": [dict(actor) for actor in self.actors],
            "locations": [{"id": "stage", "name": "Mirror Stage"}],
        }


def build_pipeline():
    return PromptGenerationPipeline(
        llm_factory=lambda app_config: None,
        prompt_pipeline_factory=lambda llm: None,
        concept_batcher_factory=lambda llm, size: None,
        scene_prompt_builder_factory=lambda llm: None,
    )


class PromptGenerationCastPreservationTests(unittest.TestCase):
    def test_full_auto_band_keeps_explicit_gender_bound_to_each_role(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
        )
        prompt_pipeline = GenericBandPromptPipeline([
            {
                "id": "singer", "name": "Lead Singer", "role": "lead singer", "gender": "female",
                "visual_description": "A person on a mirror stage",
                "image_prompt": "A person singing into a microphone",
            },
            {
                "id": "guitarist", "name": "Guitarist", "role": "guitarist", "gender": "male",
                "visual_description": "A person with a guitar",
                "image_prompt": "A person playing guitar",
            },
            {
                "id": "drummer", "name": "Drummer", "role": "drummer", "gender": "female",
                "visual_description": "A person behind the drums",
                "image_prompt": "A person playing drums",
            },
            {
                "id": "bassist", "name": "Bassist", "role": "bassist", "gender": "female",
                "visual_description": "A person with a bass",
                "image_prompt": "A person playing bass",
            },
        ])

        context = build_pipeline().build_resolved_global_context(
            config=config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        by_id = {actor["id"]: actor for actor in context["actors"]}
        self.assertEqual(
            "A female person on a mirror stage",
            by_id["singer"]["visual_description"],
        )
        self.assertEqual(
            "A female person singing into a microphone",
            by_id["singer"]["image_prompt"],
        )
        self.assertEqual(
            "A male person with a guitar",
            by_id["guitarist"]["visual_description"],
        )
        self.assertEqual(
            "A male person playing guitar",
            by_id["guitarist"]["image_prompt"],
        )
        self.assertEqual(
            "A female person behind the drums",
            by_id["drummer"]["visual_description"],
        )
        self.assertEqual(
            "A female person playing drums",
            by_id["drummer"]["image_prompt"],
        )
        self.assertEqual(
            "A female person with a bass",
            by_id["bassist"]["visual_description"],
        )
        self.assertEqual(
            "A female person playing bass",
            by_id["bassist"]["image_prompt"],
        )

    def test_unspecified_gender_keeps_generic_text_byte_identical(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
        )
        prompt_pipeline = GenericBandPromptPipeline([
            {
                "id": "mimic", "name": "Mimic", "role": "dancer", "gender": "",
                "visual_description": "A person in a grey suit",
                "image_prompt": "A person dancing on a mirror stage",
            },
            {
                "id": "statue", "name": "Statue", "role": "statue", "gender": "none",
                "visual_description": "A stone figure shaped like a person",
                "image_prompt": "A person-shaped monument",
            },
        ])
        original = {actor["id"]: dict(actor) for actor in prompt_pipeline.actors}

        context = build_pipeline().build_resolved_global_context(
            config=config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        by_id = {actor["id"]: actor for actor in context["actors"]}
        for actor_id in ("mimic", "statue"):
            self.assertEqual(
                original[actor_id]["visual_description"],
                by_id[actor_id]["visual_description"],
            )
            self.assertEqual(
                original[actor_id]["image_prompt"],
                by_id[actor_id]["image_prompt"],
            )
            for field in ("visual_description", "image_prompt"):
                text = by_id[actor_id][field].lower()
                self.assertNotIn("female", text)
                self.assertNotIn("male", text)

    def test_contradicted_cast_constraint_stops_with_validation_error(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
            actors=(ActorConfig(id="singer", name="Lead Singer", role="lead singer", gender="female"),),
        )
        prompt_pipeline = GenericBandPromptPipeline([
            {
                "id": "singer", "name": "Lead Singer", "role": "lead singer", "gender": "female",
                "visual_description": "A male singer with dark hair",
                "image_prompt": "A person singing",
            },
        ])

        with self.assertRaises(FeverSlopValidationError) as caught:
            build_pipeline().build_resolved_global_context(
                config=config,
                prompt_pipeline=prompt_pipeline,
                all_lyrics="",
                run_spinner=lambda _description, func: func(),
                console=None,
            )

        message = str(caught.exception)
        self.assertIn("Lead Singer", message)
        self.assertIn("'female' contradicts 'male'", message)
        self.assertIn("visual_description", message)

    def test_partial_configured_actor_repaired_text_includes_gender(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
            subject="female lead singer in a broken digital reality",
            locations=["Mirror Stage"],
            actors=(ActorConfig(id="singer", name="Lead Singer", role="lead vocalist", gender="female"),),
        )
        prompt_pipeline = GenericBandPromptPipeline([
            {
                "id": "singer", "name": "Lead Singer", "role": "generated role", "gender": "female",
                "visual_description": "A person in a silver stage coat",
                "image_prompt": "A person on a mirror stage",
            },
        ])

        context = build_pipeline().build_resolved_global_context(
            config=config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        actor = context["actors"][0]
        self.assertEqual(1, prompt_pipeline.subject_locations_calls)
        self.assertEqual("female", actor["gender"])
        self.assertEqual("lead vocalist", actor["role"])
        self.assertEqual("A female person in a silver stage coat", actor["visual_description"])
        self.assertEqual("A female person on a mirror stage", actor["image_prompt"])

    def test_person_word_capitalization_is_preserved_when_repairing(self):
        config = ProjectConfig(
            project_dir=Path(),
            project_name="test",
            input_audio=Path("song.mp3"),
        )
        prompt_pipeline = GenericBandPromptPipeline([
            {
                "id": "singer", "name": "Lead Singer", "role": "lead singer", "gender": "female",
                "visual_description": "Person in a red coat on stage",
                "image_prompt": "individual with a microphone",
            },
        ])

        context = build_pipeline().build_resolved_global_context(
            config=config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            console=None,
        )

        actor = context["actors"][0]
        self.assertEqual("Female person in a red coat on stage", actor["visual_description"])
        self.assertEqual("female individual with a microphone", actor["image_prompt"])
