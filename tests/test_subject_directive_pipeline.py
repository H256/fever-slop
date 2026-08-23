import unittest
import json
from pathlib import Path

from feverslop.domain.subject_directives import (
    PropBinding,
    SpatialRelation,
    SubjectDirective,
    SubjectDirectivePlan,
    TemporalScope,
)
from feverslop.prompting.subject_directive_planning import (
    build_shared_staging_plan,
    project_directives_to_prompt,
    validate_projected_prompt,
)
from feverslop.prompting.subject_directive_projections import project_subject_directives
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder
from feverslop.application.ingredients_render_plan import project_ingredients_runtime_scene


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


if __name__ == "__main__":
    unittest.main()
