import json
import sys
import unittest
from pathlib import Path

from feverslop.domain.movie import MovieActor


class TestRefineActorPromptsPrompt(unittest.TestCase):
    def test_prompt_includes_actor_ids_and_source_text(self):
        from feverslop.adapters.movie_planning import _refine_actor_prompts_prompt

        actors = (
            MovieActor(id="hans", name="Hans", role="soldier", visual_description="mud covered soldier"),
            MovieActor(id="karl", name="Karl", role="soldier", visual_description="trembling young recruit"),
        )
        source_text = "EXT. SOMME - DAY\nGerman soldiers in WWI trenches."
        prompt = _refine_actor_prompts_prompt(actors, source_text, "WWI Somme 1916")

        self.assertIn("hans", prompt)
        self.assertIn("karl", prompt)
        self.assertIn("Hans", prompt)
        self.assertIn("Karl", prompt)
        self.assertIn("mud covered soldier", prompt)
        self.assertIn("SOMME", prompt)
        self.assertIn("ethnicity", prompt.lower())
        self.assertIn("face", prompt.lower())
        self.assertIn("hair", prompt.lower())
        self.assertIn("stature", prompt.lower())
        self.assertIn("image_prompt", prompt)


class TestLLMMoviePlannerRefineActors(unittest.TestCase):
    def test_refine_actors_calls_llm_and_returns_enriched_actors(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, *, system_prompt):
                self.calls.append((prompt, system_prompt))
                return json.dumps({
                    "actors": [
                        {"id": "hans", "visual_description": "Central European, lean face, short brown hair, wearing field-grey M1916 tunic", "image_prompt": "refined prompt"},
                        {"id": "karl", "visual_description": "Young Central European, round face, blond hair, field-gray uniform", "image_prompt": "refined prompt 2"},
                    ]
                })

        actors = (
            MovieActor(id="hans", name="Hans", role="soldier", visual_description="mud covered soldier"),
            MovieActor(id="karl", name="Karl", role="recruit", visual_description="trembling young recruit"),
        )
        llm = FakeLLM()
        planner = LLMMoviePlanner(llm)
        result = planner.refine_actors(
            actors,
            source_text="EXT. SOMME - DAY\nGerman soldiers.",
            premise="WWI Somme 1916",
        )

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "hans")
        self.assertIn("Central European", result[0].visual_description)
        self.assertIn("M1916", result[0].visual_description)
        self.assertEqual(result[1].id, "karl")
        self.assertIn("blond", result[1].visual_description)

    def test_refine_actors_falls_back_on_parse_error(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner

        class BrokenLLM:
            def complete_prompt(self, prompt, *, system_prompt):
                raise RuntimeError("LLM error")

        actors = (
            MovieActor(id="hans", name="Hans", role="soldier", visual_description="mud covered soldier"),
        )
        planner = LLMMoviePlanner(BrokenLLM())
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

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, *, system_prompt):
                self.calls.append((prompt, system_prompt))
                if "film development producer" in system_prompt:
                    return json.dumps({
                        "title": "Test",
                        "premise": "Test premise",
                        "actors": [
                            {"id": "hans", "name": "Hans", "role": "soldier", "visual_description": "mud covered soldier"},
                        ],
                        "locations": [
                            {"id": "trench", "name": "Trench", "visual_description": "muddy trench"},
                        ],
                    })
                elif "character designer" in system_prompt:
                    return json.dumps({
                        "actors": [
                            {"id": "hans", "visual_description": "Central European, lean, brown hair, field-grey M1916 tunic", "image_prompt": "refined"},
                        ]
                    })
                return "{}"

        planner = LLMMoviePlanner(FakeLLM())
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

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, *, system_prompt):
                self.calls.append((prompt, system_prompt))
                return json.dumps({
                    "title": "Test",
                    "premise": "Test premise",
                    "actors": [
                        {"id": "hans", "name": "Hans", "role": "soldier", "visual_description": "mud covered soldier"},
                    ],
                    "locations": [
                        {"id": "trench", "name": "Trench", "visual_description": "muddy trench"},
                    ],
                })

        llm = FakeLLM()
        planner = LLMMoviePlanner(llm)
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
        system_prompts = [call[1] for call in llm.calls]
        self.assertNotIn("character designer", " ".join(system_prompts).lower())


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

        with tempfile.TemporaryDirectory() as temp_dir:
            planner = RefineActorPlanner()
            ScaffoldMovieUseCase(planner=planner, projects_root=Path(temp_dir)).execute(
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
