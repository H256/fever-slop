import unittest
from contextlib import nullcontext

from feverslop.prompting.movie_planning_modules import MoviePlanningModules
from feverslop.prompting.movie_planning_signatures import build_movie_planning_signature_bundle
from feverslop.prompting.guide_loader import load_markdown_guide


class MoviePlanningDspyContractTests(unittest.TestCase):
    def test_signature_bundle_covers_all_movie_planning_contracts(self):
        bundle = build_movie_planning_signature_bundle()

        self.assertEqual(
            {
                "story_arch", "movie_bible", "refine_locations", "refine_actors",
                "continuity_plan", "story_design", "screenplay", "narrative_plan",
                "shot_plan_from_bible", "shot_plan",
            },
            set(bundle),
        )
        self.assertIn("guide", bundle["story_arch"].input_fields)
        self.assertIn("result", bundle["movie_bible"].output_fields)
        self.assertIn("result", bundle["shot_plan"].output_fields)

    def test_legacy_transport_receives_markdown_guide_and_structured_payload(self):
        class LLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, **kwargs):
                self.calls.append(kwargs)
                return '{"title":"T","premise":"P","beats":["B"]}'

        llm = LLM()
        modules = MoviePlanningModules(llm)
        result = modules.story_arch({"title": "T", "source_type": "idea", "story_text": "P"})

        self.assertEqual("{\"title\":\"T\",\"premise\":\"P\",\"beats\":[\"B\"]}", result)
        self.assertIn("story arch", llm.calls[0]["prompt"].lower())
        self.assertIn('"story_text": "P"', llm.calls[0]["prompt"])
        self.assertNotIn("{{", llm.calls[0]["prompt"])

    def test_dspy_transport_passes_typed_payload_and_timeout_to_each_contract(self):
        calls = []

        class Predictor:
            def __init__(self, name):
                self.name = name

            def __call__(self, **kwargs):
                calls.append((self.name, kwargs))
                return {"result": {"title": "T"}, "shots": []}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor(signature.__name__))

        modules = MoviePlanningModules(LLM(), dspy_runtime=Runtime())
        modules.story_arch({"title": "T"}, timeout=17.0)

        self.assertEqual("StoryArch", calls[0][0])
        self.assertEqual({"title": "T"}, calls[0][1]["payload"])
        self.assertEqual(17.0, calls[0][1]["config"]["timeout"])

    def test_dspy_module_exposes_all_ten_contract_calls(self):
        calls = []

        class Predictor:
            def __init__(self, name):
                self.name = name

            def __call__(self, **kwargs):
                calls.append(self.name)
                return {"result": {}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor(signature.__name__))

        modules = MoviePlanningModules(LLM(), dspy_runtime=Runtime())
        for name in (
            "story_arch", "movie_bible", "refine_locations", "refine_actors", "continuity_plan",
            "story_design", "screenplay", "narrative_plan", "shot_plan_from_bible", "shot_plan",
        ):
            getattr(modules, name)({"contract": name})

        self.assertEqual(
            {"StoryArch", "MovieBible", "RefineLocations", "RefineActors", "ContinuityPlan", "StoryDesign", "Screenplay", "NarrativePlan", "ShotPlanFromBible", "ShotPlan"},
            set(calls),
        )

    def test_all_movie_guides_are_bundled_markdown_resources(self):
        for name in (
            "movie-story-arch", "movie-bible", "movie-refine-locations", "movie-refine-actors",
            "movie-continuity-plan", "movie-story-design", "movie-screenplay", "movie-narrative-plan",
            "movie-shot-plan-bible", "movie-shot-plan",
        ):
            self.assertTrue(load_markdown_guide(name).strip())


if __name__ == "__main__":
    unittest.main()
