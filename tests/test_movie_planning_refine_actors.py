import json
import sys
import unittest
from pathlib import Path

from feverslop.adapters.movie_artifact_writer import LocalMovieArtifactWriter
from feverslop.domain.movie import MovieActor


class TestRefineActorPromptsPrompt(unittest.TestCase):
    def test_actor_refinement_guide_is_a_dspy_resource(self):
        from feverslop.prompting.guide_loader import load_markdown_guide

        guide = load_markdown_guide("movie-refine-actors")
        self.assertTrue(guide.strip())
        self.assertIn("stable physical appearance", guide)

    def test_krea_guide_is_selected_only_for_krea_workflow(self):
        from feverslop.adapters.movie_planning_prompts import _krea_reference_guides

        location_guide, actor_guide = _krea_reference_guides("workflows/image_t2i_startframe_krea_v1.json")
        self.assertIn("Krea Location", location_guide)
        self.assertIn("Krea Actor", actor_guide)
        self.assertEqual(("", ""), _krea_reference_guides("workflows/image_t2i_startframe_v1.json"))


class TestLLMMoviePlannerRefineActors(unittest.TestCase):
    class FakeModules:
        def __init__(self, *, actor_result=None, bible_result=None, error=None):
            self.actor_result = actor_result or {"actors": []}
            self.bible_result = bible_result or {}
            self.error = error
            self.calls = []

        def refine_actors(self, payload):
            self.calls.append(("refine_actors", payload))
            if self.error:
                raise self.error
            return self.actor_result

        def movie_bible(self, payload):
            self.calls.append(("movie_bible", payload))
            return self.bible_result

    def test_refine_actors_calls_llm_and_returns_enriched_actors(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner

        actors = (
            MovieActor(id="hans", name="Hans", role="soldier", visual_description="mud covered soldier"),
            MovieActor(id="karl", name="Karl", role="recruit", visual_description="trembling young recruit"),
        )
        modules = self.FakeModules(actor_result={"actors": [
            {"id": "hans", "visual_description": "Central European, lean face, short brown hair, wearing field-grey M1916 tunic"},
            {"id": "karl", "visual_description": "Young Central European, round face, blond hair, field-gray uniform"},
        ]})
        planner = LLMMoviePlanner(object(), modules=modules)
        result = planner.refine_actors(
            actors,
            source_text="EXT. SOMME - DAY\nGerman soldiers.",
            premise="WWI Somme 1916",
        )

        self.assertEqual(len(modules.calls), 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "hans")
        self.assertIn("Central European", result[0].visual_description)
        self.assertIn("M1916", result[0].visual_description)
        self.assertEqual(result[1].id, "karl")
        self.assertIn("blond", result[1].visual_description)

    def test_refine_actors_includes_krea_guide_for_krea_workflow(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner

        modules = self.FakeModules()
        planner = LLMMoviePlanner(object(), reference_hero_workflow="image_t2i_startframe_krea_v1.json", modules=modules)
        planner.refine_actors(
            (MovieActor(id="hans", name="Hans", role="soldier", visual_description="soldier"),),
            source_text="soldier",
            premise="war",
        )

        self.assertIn("Krea Actor Reference Prompt Guide", modules.calls[0][1]["guide"])

    def test_refine_actors_falls_back_on_parse_error(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner

        actors = (
            MovieActor(id="hans", name="Hans", role="soldier", visual_description="mud covered soldier"),
        )
        planner = LLMMoviePlanner(object(), modules=self.FakeModules(error=RuntimeError("LLM error")))
        result = planner.refine_actors(
            actors,
            source_text="some text",
            premise="some premise",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "hans")
        self.assertEqual(result[0].visual_description, "mud covered soldier")


class TestGenerateMovieBibleRefineActors(unittest.TestCase):
    def test_generate_movie_bible_refines_actors_when_configured(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import StoryArch

        modules = TestLLMMoviePlannerRefineActors.FakeModules(
            bible_result={"title": "Test", "premise": "Test premise", "actors": [
                {"id": "hans", "name": "Hans", "role": "soldier", "visual_description": "mud covered soldier"},
            ], "locations": [{"id": "trench", "name": "Trench", "visual_description": "muddy trench"}]},
            actor_result={"actors": [{"id": "hans", "visual_description": "Central European, lean, brown hair, field-grey M1916 tunic"}]},
        )
        planner = LLMMoviePlanner(object(), modules=modules)
        arch = StoryArch(title="Test", premise="Test premise", beats=("beat1",))
        bible = planner.generate_movie_bible(
            title="Test",
            source_type="screenplay",
            story_text="EXT. TRENCH - DAY\nGerman soldier.",
            desired_length=30,
            story_arch=arch,
            config={"refine_actor_prompts": True},
        )

        self.assertEqual(bible.actors[0].id, "hans")
        self.assertIn("Central European", bible.actors[0].visual_description)
        self.assertNotIn("mud covered soldier", bible.actors[0].visual_description)

    def test_generate_movie_bible_skips_refine_when_not_configured(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import StoryArch

        modules = TestLLMMoviePlannerRefineActors.FakeModules(bible_result={"title": "Test", "premise": "Test premise", "actors": [
            {"id": "hans", "name": "Hans", "role": "soldier", "visual_description": "mud covered soldier"},
        ], "locations": [{"id": "trench", "name": "Trench", "visual_description": "muddy trench"}]})
        planner = LLMMoviePlanner(object(), modules=modules)
        arch = StoryArch(title="Test", premise="Test premise", beats=("beat1",))
        bible = planner.generate_movie_bible(
            title="Test",
            source_type="screenplay",
            story_text="EXT. TRENCH - DAY\nGerman soldier.",
            desired_length=30,
            story_arch=arch,
            config={"refine_actor_prompts": False},
        )

        self.assertEqual(bible.actors[0].visual_description, "mud covered soldier")
        self.assertEqual(["movie_bible"], [call[0] for call in modules.calls])


class TestConfigFlags(unittest.TestCase):
    def test_project_create_request_has_refine_actor_prompts(self):
        from feverslop.studio.projects import ProjectCreateRequest

        req = ProjectCreateRequest(
            project_type="movie",
            name="Test",
            movie_refine_actor_prompts=True,
        )
        self.assertTrue(req.movie_refine_actor_prompts)

    def test_movie_project_config_includes_refine_actor_prompts(self):
        from feverslop.studio.projects import ProjectCreateRequest
        from feverslop.studio.project_repository import movie_project_config

        req = ProjectCreateRequest(
            project_type="movie",
            name="Test",
            movie_refine_actor_prompts=True,
        )
        config = movie_project_config(req)
        self.assertTrue(config["refine_actor_prompts"])

    def test_movie_project_config_defaults_refine_actor_prompts_to_false(self):
        from feverslop.studio.projects import ProjectCreateRequest
        from feverslop.studio.project_repository import movie_project_config

        req = ProjectCreateRequest(
            project_type="movie",
            name="Test",
        )
        config = movie_project_config(req)
        self.assertFalse(config["refine_actor_prompts"])


class TestScaffoldCLIFlag(unittest.TestCase):
    def test_scaffold_movie_accepts_refine_actors_flag(self):
        import subprocess
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scaffold_movie.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        self.assertIn("--refine-actors", result.stdout)


class TestScaffoldWithRefineActors(unittest.TestCase):
    def test_scaffold_with_refine_actors_produces_detailed_descriptions(self):
        import tempfile
        from pathlib import Path
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.domain.movie import StoryArch, CinematicShot, MovieBible, MovieLocation, MovieContinuityRule

        class RefineActorPlanner:
            def __init__(self):
                self.refine_called = False

            def generate_story_arch(self, **kwargs):
                return StoryArch(title=kwargs["title"], premise="Test premise", beats=("beat1",))

            def generate_movie_bible(self, **kwargs):
                story_arch = kwargs["story_arch"]
                config = kwargs["config"]
                actors = [
                    MovieActor(id="hans", name="Hans", role="soldier", visual_description="mud covered soldier"),
                ]
                if config.get("refine_actor_prompts"):
                    self.refine_called = True
                    actors = [
                        MovieActor(id="hans", name="Hans", role="soldier", visual_description="Central European, lean face, brown hair, M1916 tunic"),
                    ]
                return MovieBible(
                    title=story_arch.title,
                    premise=story_arch.premise,
                    story_arch=story_arch,
                    actors=tuple(actors),
                    locations=(MovieLocation(id="trench", name="Trench", visual_description="muddy trench"),),
                    continuity=(MovieContinuityRule(id="v1", description="wardrobe continuity"),),
                    style_constraints=(),
                    runtime_constraints={"max_scene_actors": 4},
                )

            def plan_shots_from_bible(self, **kwargs):
                bible = kwargs["bible"]
                return (
                    CinematicShot(
                        shot_id="shot_0001",
                        description="Hans in trench",
                        duration_seconds=kwargs["desired_length"],
                        camera="static",
                        action="Hans looks",
                        expression="focused",
                        location=bible.locations[0].name,
                        dialogue="",
                        actor_ids=("hans",),
                        location_id="trench",
                    ),
                )

            def generate_movie_continuity_plan(self, **_kwargs):
                return {}

            def generate_movie_story_design(self, **_kwargs):
                return {}

            def generate_movie_screenplay(self, **_kwargs):
                return {}

            def generate_movie_narrative_plan(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as temp_dir:
            planner = RefineActorPlanner()
            ScaffoldMovieUseCase(planner=planner, projects_root=Path(temp_dir), artifact_writer=LocalMovieArtifactWriter()).execute(
                MovieInput(
                    name="Refine Test",
                    source_type="short_story",
                    story_text="A soldier in a trench during WWI.",
                    desired_length=12,
                    config={"refine_actor_prompts": True},
                )
            )

            self.assertTrue(planner.refine_called)
            root = Path(temp_dir) / "refine-test"
            bible = json.loads((root / "movie" / "bible.json").read_text(encoding="utf-8"))
            self.assertIn("Central European", bible["actors"][0]["visual_description"])


if __name__ == "__main__":
    unittest.main()
