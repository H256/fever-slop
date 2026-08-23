import unittest

from feverslop.domain.render_plan import RenderPlan
from feverslop.application.render_plan_validation import validate_render_plan_subject_directives
from feverslop.domain.subject_directives import (
    PROP_STATES,
    SpatialRelation,
    SubjectDirective,
    SubjectDirectivePlan,
    TemporalScope,
    validate_subject_directive_plan,
)


class SubjectDirectiveContractTests(unittest.TestCase):
    def test_round_trips_versioned_backend_neutral_plan(self):
        payload = {
            "schema_version": "subject-directives/v1",
            "shot_id": "scene-47-shot-1",
            "temporal_scope": {"start_seconds": 0.0, "end_seconds": 4.0},
            "subjects": [
                {
                    "subject_id": "drummer",
                    "role": "drummer",
                    "position": "rear left behind the drum kit",
                    "action": "plays the drums to the beat",
                    "prop_bindings": [{"prop_id": "drum-kit", "state": "played"}],
                    "visibility": "visible",
                    "cardinality": 1,
                    "temporal_scope": {"start_seconds": 0.0, "end_seconds": 4.0},
                }
            ],
        }

        plan = SubjectDirectivePlan.from_dict(payload)
        self.assertEqual(payload, plan.to_dict())
        self.assertEqual(PROP_STATES, {"held", "played", "attached", "placed", "absent"})

    def test_render_scene_preserves_legacy_scene_without_directives(self):
        scene = {"scene": 1, "h3": {"prompt": "legacy prompt"}}
        plan = RenderPlan.from_dicts([scene])
        self.assertIsNone(plan.scenes[0].subject_directive_plan)
        self.assertEqual(scene, plan.to_dicts()[0])

    def test_rejects_unknown_ids_duplicate_subjects_and_invalid_props(self):
        plan = SubjectDirectivePlan(
            shot_id="shot-1",
            temporal_scope=TemporalScope(0, 4),
            subjects=(
                SubjectDirective("singer", "singer", "front", "sings", prop_bindings=()),
                SubjectDirective("singer", "singer", "front", "sings", prop_bindings=()),
            ),
        )
        issues = validate_subject_directive_plan(
            plan, known_subject_ids={"drummer"}, known_prop_ids={"microphone"}
        )
        self.assertTrue(any("duplicate subject" in issue for issue in issues))
        self.assertTrue(any("Unknown subject ID" in issue for issue in issues))

    def test_allows_explicit_environment_subjects_alongside_referenced_actors(self):
        plan = SubjectDirectivePlan(
            shot_id="shot-1",
            temporal_scope=TemporalScope(0, 4),
            subjects=(
                SubjectDirective(
                    "singer", "singer", "front", "sings", temporal_scope=TemporalScope(0, 4)
                ),
                SubjectDirective(
                    "stage_haze",
                    "atmospheric effect",
                    "background",
                    "drifts",
                    temporal_scope=TemporalScope(0, 4),
                ),
            ),
        )
        issues = validate_subject_directive_plan(
            plan,
            known_subject_ids={"singer"},
            known_environment_ids={"festival_stage"},
        )
        self.assertEqual([], issues)

    def test_allows_crowd_roles_and_implicit_visual_effect_relations(self):
        plan = SubjectDirectivePlan(
            shot_id="shot-1",
            temporal_scope=TemporalScope(0, 4),
            subjects=(
                SubjectDirective(
                    "singer", "singer", "stage", "sings", temporal_scope=TemporalScope(0, 4)
                ),
                SubjectDirective(
                    "front_row_crowd",
                    "ambient_population",
                    "foreground",
                    "cheers",
                    temporal_scope=TemporalScope(0, 4),
                ),
            ),
            spatial_relations=(
                SpatialRelation("white_light_and_fog", "framing", "singer"),
            ),
        )
        issues = validate_subject_directive_plan(plan, known_subject_ids={"singer"})
        self.assertEqual([], issues)

    def test_rejects_incomplete_temporal_coverage_and_contradictory_relations(self):
        plan = SubjectDirectivePlan.from_dict(
            {
                "schema_version": "subject-directives/v1",
                "shot_id": "shot-1",
                "temporal_scope": {"start_seconds": 0, "end_seconds": 4},
                "subjects": [
                    {
                        "subject_id": "singer",
                        "role": "singer",
                        "position": "front",
                        "action": "sings",
                        "temporal_scope": {"start_seconds": 1, "end_seconds": 3},
                    }
                ],
                "spatial_relations": [
                    {"subject_id": "singer", "relation": "left_of", "target_id": "band"},
                    {"subject_id": "singer", "relation": "left_of", "target_id": "band", "detail": "right"},
                ],
            }
        )
        issues = validate_subject_directive_plan(plan, known_subject_ids={"singer", "band"})
        self.assertTrue(any("temporal coverage" in issue for issue in issues))
        self.assertTrue(any("contradictory spatial relation" in issue for issue in issues))

    def test_render_plan_validation_skips_legacy_and_rejects_directive_errors(self):
        validate_render_plan_subject_directives([{"scene": 1, "h3": {"prompt": "legacy"}}], render_plan_path="legacy.json")
        with self.assertRaisesRegex(ValueError, "Unknown subject ID"):
            validate_render_plan_subject_directives(
                [{
                    "scene": 2,
                    "subject_directives": SubjectDirectivePlan(
                        shot_id="shot-2", temporal_scope=TemporalScope(0, 1),
                        subjects=(SubjectDirective("missing", "role", "front", "acts", temporal_scope=TemporalScope(0, 1)),),
                    ).to_dict(),
                }],
                known_subject_ids=("known",),
                render_plan_path="plan.json",
            )


if __name__ == "__main__":
    unittest.main()
