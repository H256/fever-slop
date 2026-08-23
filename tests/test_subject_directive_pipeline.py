import unittest
import json
from unittest.mock import patch
from pathlib import Path

from feverslop.domain.subject_directives import (
    PropBinding,
    SpatialRelation,
    SubjectDirective,
    SubjectDirectivePlan,
    TemporalScope,
)
from feverslop.prompting.subject_directive_planning import (
    DspySubjectDirectivePlanner,
    build_shared_staging_plan,
    SubjectDirectivePlanner,
    project_directives_to_prompt,
    validate_projected_prompt,
)
from feverslop.prompting.subject_directive_projections import project_subject_directives
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder
from feverslop.application.ingredients_render_plan import project_ingredients_runtime_scene
from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline


def _plan():
    scope = TemporalScope(0, 4)
    return SubjectDirectivePlan(
        shot_id="scene-47-shot-1",
        temporal_scope=scope,
        subjects=(
            SubjectDirective(
                "drummer", "drummer", "rear left behind the drum kit", "plays the drums to the beat",
                prop_bindings=(PropBinding("drum-kit", "played"),), temporal_scope=scope,
            ),
            SubjectDirective(
                "keyboarder", "keyboard player", "rear right at the keyboard", "plays sustained chords",
                prop_bindings=(PropBinding("keyboard", "played"),), temporal_scope=scope,
            ),
        ),
        spatial_relations=(SpatialRelation("drummer", "left_of", "keyboarder"),),
    )


class SubjectDirectivePipelineTests(unittest.TestCase):
    def test_shared_staging_creates_directives_from_one_scene_input(self):
        plan = build_shared_staging_plan({
            "shot_id": "shot-1",
            "duration_seconds": 2,
            "subjects": [
                {"subject_id": "singer", "role": "singer", "position": "front center", "action": "sings"}
            ],
        })
        self.assertEqual(("singer",), tuple(item.subject_id for item in plan.subjects))
        self.assertEqual("front center", plan.subjects[0].position)

    def test_planner_calls_one_predictor_with_the_complete_scene(self):
        calls = []

        def predictor(payload):
            calls.append(payload)
            return {"subject_directives": {
                "schema_version": "subject-directives/v1",
                "shot_id": "shot-1",
                "temporal_scope": {"start_seconds": 0, "end_seconds": 2},
                "subjects": [{
                    "subject_id": "singer", "role": "singer", "position": "front",
                    "action": "sings", "temporal_scope": {"start_seconds": 0, "end_seconds": 2},
                }],
            }}

        plan = SubjectDirectivePlanner(predictor=predictor).plan({
            "shot_id": "shot-1", "concept": "singer performs", "subjects": [{"subject_id": "singer"}],
        })
        self.assertEqual(1, len(calls))
        self.assertIn("concept", calls[0]["scene"])
        self.assertEqual("singer", plan.subjects[0].subject_id)

    def test_planner_rejects_zero_length_model_scopes(self):
        with self.assertRaisesRegex(ValueError, "temporal scope requires"):
            SubjectDirectivePlanner(predictor=lambda _payload: {
                "shot_id": "shot-1",
                "temporal_scope": {"start_seconds": 0, "end_seconds": 0},
                "subjects": [{
                    "subject_id": "singer", "role": "singer", "position": "front",
                    "action": "sings", "temporal_scope": {"start_seconds": 0, "end_seconds": 0},
                }],
            }).plan({"shot_id": "shot-1", "duration_seconds": 4})

    def test_dspy_planner_decodes_nested_json_fields(self):
        class Runtime:
            def predict(self, _signature):
                return lambda **_kwargs: {
                    "staging_plan": {
                        "schema_version": "subject-directives/v1",
                        "shot_id": "shot-1",
                        "temporal_scope": '{"start_seconds": 0, "end_seconds": 0}',
                        "subjects": '[{"subject_id":"singer","role":"singer","position":"front","action":"sings","temporal_scope":{"start_seconds":0,"end_seconds":0}}]',
                        "spatial_relations": "[]",
                    }
                }

            def make_lm(self, _llm, **_kwargs):
                return object()

            def context(self, **_kwargs):
                from contextlib import nullcontext
                return nullcontext()

        with patch("dspy.Module", object):
            with patch("feverslop.prompting.dspy_subject_directive_signatures.build_subject_directive_signature", return_value=object()):
                planner = DspySubjectDirectivePlanner(object(), dspy_runtime=Runtime())
        plan = planner.plan({"shot_id": "shot-1", "duration_seconds": 2})
        self.assertEqual("singer", plan.subjects[0].subject_id)
        self.assertEqual(2, plan.temporal_scope.end_seconds)
        self.assertEqual(2, plan.subjects[0].temporal_scope.end_seconds)
        self.assertEqual(2, len(planner.last_repairs))

    def test_projection_is_explicit_and_backend_neutral(self):
        text = project_directives_to_prompt(_plan())
        self.assertIn("drummer", text)
        self.assertIn("drum-kit (played)", text)
        self.assertIn("drummer left_of keyboarder", text)
        self.assertNotIn("their instruments", text)

    def test_missing_subject_or_prop_coverage_fails_before_rendering(self):
        with self.assertRaisesRegex(ValueError, "keyboarder"):
            validate_projected_prompt(_plan(), "drummer plays the drum-kit (played).")

    def test_scene_52_explicit_absent_prop_is_preserved(self):
        scope = TemporalScope(0, 2)
        plan = SubjectDirectivePlan(
            shot_id="scene-52-shot-1", temporal_scope=scope,
            subjects=(SubjectDirective(
                "crowd", "crowd", "background", "moves with the beat",
                prop_bindings=(PropBinding("instruments", "absent"),), temporal_scope=scope,
            ),),
        )
        text = project_directives_to_prompt(plan)
        self.assertIn("instruments (absent)", text)
        validate_projected_prompt(plan, text)

    def test_h3_projection_carries_the_shared_directives(self):
        plan = _plan()

        class Generator:
            def __call__(self, request):
                return {"rendered_prompt": project_directives_to_prompt(plan)}

        result = DspyH3PromptBuilder(Generator()).build_h3_prompt(
            segment={"segment_id": "scene-47", "subject_directives": plan.to_dict()},
            concept="live performance",
            scene_details={}, global_context={}, mode="ref",
        )
        self.assertIn("keyboarder", result["prompt"])
        self.assertEqual("subject-directives/v1", result["subject_directives"]["schema_version"])

    def test_ingredients_projection_keeps_directives_in_static_prompt(self):
        plan = _plan()
        scene = {
            "scene": 47,
            "duration_seconds": 4,
            "subject_directives": plan.to_dict(),
            "ltx": {"prompt_relay": [{"state": "instrumental", "prompt": "the band performs"}]},
            "ingredients_global_prompt": "A concert stage",
        }
        projected = project_ingredients_runtime_scene(scene)
        self.assertIn("drum-kit (played)", projected["ltx"]["static_prompt"])

    def test_all_backend_projections_share_the_same_fact_coverage(self):
        prompts = [
            project_subject_directives(_plan(), backend=backend).prompt
            for backend in ("minimax-h3-r2v", "ltx-t2v", "ltx-msr", "ltx-ingredients")
        ]
        self.assertEqual(1, len({prompt for prompt in prompts}))

    def test_regression_fixtures_build_without_model_calls(self):
        fixture = Path(__file__).parent / "fixtures" / "subject_directives" / "regression_scenes.json"
        cases = json.loads(fixture.read_text(encoding="utf-8"))
        plans = [build_shared_staging_plan(case) for case in cases]
        self.assertEqual(3, len(plans))
        self.assertEqual(5, plans[2].subjects[0].cardinality)
        self.assertEqual("absent", plans[2].subjects[0].prop_bindings[0].state)

    def test_prompt_pipeline_auto_generates_and_persists_directives_for_real_llm(self):
        generated = _plan()

        class Store:
            def __init__(self):
                self.payload = [{"segment_id": "seg-1"}]

            def read_json(self, _path):
                return self.payload

            def write_json(self, _path, payload):
                self.payload = payload

        class LLM:
            model = "test-model"
            client = object()

        store = Store()
        with patch(
            "feverslop.application.prompt_generation_pipeline.DspySubjectDirectivePlanner"
        ) as planner_type:
            planner_type.return_value.plan.return_value = generated
            PromptGenerationPipeline._generate_subject_directives(
                llm=LLM(), stage1_segments=[{"segment_id": "seg-1"}],
                concept_prompts={}, scene_details={}, global_context={},
                scene_prompts_json="scene-prompts.json", artifact_store=store,
                reporter=type("Reporter", (), {"message": lambda *_args: None})(),
            )
        self.assertEqual("scene-47-shot-1", store.payload[0]["subject_directives"]["shot_id"])
        planner_type.assert_called_once()

    def test_subject_staging_retry_uses_rich_panel_output(self):
        generated = _plan()

        class Store:
            def read_json(self, _path):
                return [{"segment_id": "seg-1"}]

            def write_json(self, _path, _payload):
                pass

        class LLM:
            model = "test-model"
            client = object()

        class Reporter:
            def __init__(self):
                self.panels = []
                self.messages = []

            def panel(self, text, *, title=None):
                self.panels.append((title, text))

            def message(self, text):
                self.messages.append(text)

        reporter = Reporter()
        with patch(
            "feverslop.application.prompt_generation_pipeline.DspySubjectDirectivePlanner"
        ) as planner_type:
            planner_type.return_value.plan.side_effect = [ValueError("malformed scope"), generated]
            PromptGenerationPipeline._generate_subject_directives(
                llm=LLM(), stage1_segments=[{"segment_id": "seg-1"}],
                concept_prompts={}, scene_details={}, global_context={},
                scene_prompts_json="scene-prompts.json", artifact_store=Store(),
                reporter=reporter,
            )

        self.assertEqual(1, len(reporter.panels))
        title, text = reporter.panels[0]
        self.assertIn("Subject staging", title)
        self.assertIn("Retry 2/3", title)
        self.assertIn("malformed scope", text)


if __name__ == "__main__":
    unittest.main()
