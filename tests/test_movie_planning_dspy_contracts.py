import unittest
from contextlib import nullcontext
from unittest.mock import patch

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.movie_planning_modules import MoviePlanningModules
from feverslop.prompting.movie_planning_signatures import (
    MovieBiblePayload,
    MovieBibleResult,
    StoryArchPayload,
    build_movie_planning_signature_bundle,
)


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

    def test_movie_planning_fails_clearly_without_dspy_predictors(self):
        class LLM:
            model = "fake-model"
            client = object()

        with patch.dict("sys.modules", {"dspy": None}):
            with self.assertRaisesRegex(RuntimeError, "DSPy.*movie planning"):
                MoviePlanningModules(LLM())

    def test_movie_planning_rejects_unstructured_predictor_output(self):
        with self.assertRaises(TypeError):
            from feverslop.adapters.movie_planning_llm import _module_data

            _module_data('{"title":"T"}')

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
        self.assertIsInstance(calls[0][1]["payload"], StoryArchPayload)
        self.assertEqual("T", calls[0][1]["payload"].title)
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

    def test_movie_signatures_use_typed_payload_and_output_models(self):
        bundle = build_movie_planning_signature_bundle()

        self.assertIs(bundle["story_arch"].input_fields["payload"].annotation, StoryArchPayload)
        self.assertIs(bundle["movie_bible"].input_fields["payload"].annotation, MovieBiblePayload)
        self.assertIs(bundle["movie_bible"].output_fields["result"].annotation, MovieBibleResult)


if __name__ == "__main__":
    unittest.main()
