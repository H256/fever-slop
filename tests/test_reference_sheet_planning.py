import unittest

from feverslop.adapters.sequence_to_sheet_backend import ComfyUISequenceToSheetBackend
from feverslop.application.reference_sheet_planning import (
    DeterministicReferenceSheetPlanner,
    ReferenceSheetPlanner,
    compile_reference_sheet_plan,
)
from feverslop.domain.reference_sheet import CompiledReferenceSheetPlan
from feverslop.prompting.reference_sheet_modules import ReferenceSheetPlanningModules


class ReferenceSheetPlanningTests(unittest.TestCase):
    def test_compiler_enforces_character_views_and_constraints(self):
        plan = {
            "view_labels": ["invented"],
            "view_count": 1,
            "framing": "tight crop",
            "coverage": "freeform",
            "rotation": "none",
            "backdrop": "busy street",
            "identity_constraints": [],
            "negative_constraints": [],
        }

        compiled = compile_reference_sheet_plan(
            plan,
            kind="character",
            description="A silver astronaut with a red visor",
        )

        self.assertEqual(
            ("full_body", "front", "left_profile", "right_profile", "back", "closeup"),
            compiled.view_labels,
        )
        self.assertEqual(6, compiled.view_count)
        self.assertEqual("full body, generous margin", compiled.framing)
        self.assertIn("clothing, hair, colors, proportions", compiled.identity_constraints)
        self.assertIn("text, watermark, split screen", compiled.negative_constraints)

    def test_location_compiler_selects_landscape_cut_views(self):
        compiled = compile_reference_sheet_plan(
            {"coverage": "continuous move", "rotation": "full"},
            kind="location",
            description="An abandoned observatory",
        )

        self.assertEqual(5, compiled.view_count)
        self.assertEqual("cut views", compiled.coverage)
        self.assertEqual("landscape", compiled.framing)
        self.assertEqual(
            ("front", "right_side", "rear", "left_side", "wide_establishing"),
            compiled.view_labels,
        )

    def test_compiler_uses_requested_frame_count_for_duration(self):
        compiled = compile_reference_sheet_plan(
            {}, kind="character", description="A pilot", frames=124,
        )

        self.assertAlmostEqual(124 / 24, compiled.duration_seconds)

    def test_character_anchor_sanitizes_action_and_location_without_losing_later_identity(self):
        compiled = compile_reference_sheet_plan(
            {
                "anchor_description": (
                    "A woman singing passionately, with long silver hair and dark leather armor "
                    "on a black stone altar"
                ),
            },
            kind="character",
            description="A singer",
        )

        self.assertIn("long silver hair", compiled.anchor_description)
        self.assertIn("dark leather armor", compiled.anchor_description)
        self.assertNotIn("sing", compiled.anchor_description.lower())
        self.assertNotIn("altar", compiled.anchor_description.lower())

    def test_character_anchor_preserves_unpunctuated_identity_after_action(self):
        compiled = compile_reference_sheet_plan(
            {
                "anchor_description": (
                    "A woman singing with long silver hair and dark leather armor "
                    "in a smoky nightclub"
                ),
            },
            kind="character",
            description="A singer",
        )

        self.assertIn("long silver hair", compiled.anchor_description)
        self.assertIn("dark leather armor", compiled.anchor_description)
        self.assertNotIn("sing", compiled.anchor_description.lower())
        self.assertNotIn("nightclub", compiled.anchor_description.lower())

    def test_h3_location_serializer_emits_anchor_rule(self):
        backend = ComfyUISequenceToSheetBackend(
            client=object(),
            workflow_path="workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json",
            backend="minimax",
        )
        prompt = backend.build_sheet_prompt_from_plan(
            CompiledReferenceSheetPlan(
                kind="location",
                view_count=5,
                view_labels=("front", "right_side", "rear", "left_side", "wide_establishing"),
                framing="landscape",
                coverage="cut views",
                rotation="none",
                backdrop="empty station",
                duration_seconds=124 / 24,
                anchor_rule="the anchor image is fully referenced as the first frame",
                identity_constraints="empty station",
                negative_constraints="no people",
            ),
        )

        self.assertIn("anchor image is fully referenced as the first frame", prompt.prompt)

    def test_deterministic_planner_is_available_without_dspy(self):
        plan = DeterministicReferenceSheetPlanner().plan(
            kind="character",
            description="A silver astronaut",
            asset_context={"wardrobe": "white suit"},
        )

        self.assertEqual("character", plan.kind)
        self.assertEqual(6, plan.view_count)
        self.assertIn("white suit", plan.identity_constraints)
        self.assertIn("silver astronaut", plan.anchor_description.lower())

    def test_dspy_module_returns_structured_plan(self):
        class Result:
            plan = {
                "kind": "character",
                "view_count": 6,
                "view_labels": ["front", "left_profile", "right_profile", "back", "closeup", "full_body"],
                "framing": "full body, generous margin",
                "coverage": "cut views",
                "rotation": "none",
                "backdrop": "neutral studio",
                "duration_seconds": 5.0,
                "anchor_rule": "anchor is the first frame",
                "identity_constraints": ["preserve the suit"],
                "negative_constraints": ["no text"],
            }

        class Runtime:
            def predict(self, signature):
                return lambda **kwargs: Result()

            def context(self, *, lm):
                from contextlib import nullcontext

                return nullcontext()

            def make_lm(self, llm):
                return object()

        class LLM:
            model = "planner"
            client = object()

        modules = ReferenceSheetPlanningModules(LLM(), dspy_runtime=Runtime())
        result = modules.plan(
            kind="character",
            description="A silver astronaut",
            asset_context={"wardrobe": "white suit"},
        )

        self.assertEqual("character", result.kind)
        self.assertEqual(6, result.view_count)

    def test_planner_falls_back_when_dspy_output_is_incomplete(self):
        class Runtime:
            def predict(self, signature):
                return lambda **kwargs: {"plan": {"kind": "character"}}

            def context(self, *, lm):
                from contextlib import nullcontext

                return nullcontext()

            def make_lm(self, llm):
                return object()

        class LLM:
            model = "planner"
            client = object()

        planner = ReferenceSheetPlanner(llm=LLM(), dspy_runtime=Runtime())
        result = planner.plan(kind="location", description="A ruined station", asset_context={})

        self.assertEqual("location", result.kind)
        self.assertEqual(5, result.view_count)

    def test_planner_accepts_semantic_view_suggestion_for_compiler_normalization(self):
        class Runtime:
            def predict(self, signature):
                return lambda **kwargs: {
                    "plan": {
                        "kind": "character",
                        "view_count": 5,
                        "view_labels": ["front", "profile_left", "profile_right", "back", "three_quarter"],
                        "framing": "full body",
                        "coverage": "wardrobe views",
                        "rotation": "incremental",
                        "backdrop": "neutral studio",
                        "identity_constraints": ["preserve the suit"],
                        "negative_constraints": ["no logos"],
                    },
                }

            def context(self, *, lm):
                from contextlib import nullcontext

                return nullcontext()

            def make_lm(self, llm):
                return object()

        class LLM:
            model = "planner"
            client = object()

        planner = ReferenceSheetPlanner(llm=LLM(), dspy_runtime=Runtime())
        result = planner.plan(kind="character", description="A pilot", asset_context={})

        self.assertEqual("dspy", planner.source)
        self.assertEqual(5, result.view_count)


if __name__ == "__main__":
    unittest.main()
