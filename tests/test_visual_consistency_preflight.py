import unittest

from feverslop.application.visual_consistency_preflight import (
    preflight_visual_consistency,
)
from feverslop.domain.visual_consistency import PreflightMode, ReferenceAnchor
from feverslop.ports.visual_consistency import ReferenceManifestSnapshot


def _anchor(kind: str, semantic_id: str, look_id: str = "default") -> ReferenceAnchor:
    return ReferenceAnchor(
        id=semantic_id,
        kind=kind,
        look_id=look_id,
        asset_role="identity-reference" if kind == "actor" else "environment-reference",
        asset_sha256="a" * 64,
        prompt_anchor=f"{kind} {semantic_id}",
    )


class VisualConsistencyPreflightTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = ReferenceManifestSnapshot(
            actors={("hero", "default"): _anchor("actor", "hero")},
            locations={("stage", "default"): _anchor("location", "stage")},
            revision="revision",
        )

    def test_strict_missing_actor_blocks(self):
        result = preflight_visual_consistency(
            [{"scene": 1, "references": {"actor_ids": ["missing"], "location_id": "stage"}}],
            self.snapshot,
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode=PreflightMode.STRICT,
        )

        self.assertFalse(result.renderable)
        self.assertEqual(["missing_actor_reference"], [issue.code for issue in result.issues])
        self.assertEqual("error", result.issues[0].severity)

    def test_warn_legacy_unknown_is_nonblocking(self):
        result = preflight_visual_consistency(
            [{"scene": 1, "prompt": "old plan"}],
            self.snapshot,
            mode="ingredients",
            workflow_profile="ingredients-default",
            preflight_mode=PreflightMode.WARN,
        )

        self.assertTrue(result.renderable)
        self.assertEqual(["legacy_contract_unknown"], [issue.code for issue in result.issues])
        self.assertEqual("warning", result.issues[0].severity)

    def test_strict_rejects_malformed_selected_reference_bindings(self):
        cases = (
            {"reference_ids": {"actors": "hero", "location": "stage"}},
            {"references": {"actor_ids": ["hero", 7], "location_id": "stage"}},
            {"actor_ids": ["hero"], "location_id": 7},
        )
        for bindings in cases:
            with self.subTest(bindings=bindings):
                result = preflight_visual_consistency(
                    [{"scene": 1, **bindings}],
                    self.snapshot,
                    mode="i2v",
                    workflow_profile="i2v-default",
                    preflight_mode="strict",
                )

                self.assertFalse(result.renderable)
                self.assertEqual(
                    ["malformed_reference_bindings"],
                    [issue.code for issue in result.issues],
                )
                self.assertEqual("error", result.issues[0].severity)
                self.assertEqual((), result.contracts)

    def test_warn_reports_malformed_bindings_without_blocking(self):
        result = preflight_visual_consistency(
            [{"scene": 1, "references": {"actor_ids": "hero"}}],
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="warn",
        )

        self.assertTrue(result.renderable)
        self.assertEqual("malformed_reference_bindings", result.issues[0].code)
        self.assertEqual("warning", result.issues[0].severity)

    def test_ignores_malformed_lower_priority_bindings_when_movie_fields_selected(self):
        result = preflight_visual_consistency(
            [{
                "scene": 1,
                "reference_ids": {"actors": ["hero"], "location": "stage"},
                "references": {"actor_ids": "stale", "location_id": 9},
            }],
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="strict",
        )

        self.assertTrue(result.renderable)
        self.assertEqual((), result.issues)

    def test_strict_legacy_unknown_blocks(self):
        result = preflight_visual_consistency(
            [{"scene": 1, "prompt": "old plan"}],
            self.snapshot,
            mode="ingredients",
            workflow_profile="ingredients-default",
            preflight_mode="strict",
        )

        self.assertFalse(result.renderable)
        self.assertEqual("error", result.issues[0].severity)

    def test_distinguishes_missing_actor_look_from_missing_actor_id(self):
        result = preflight_visual_consistency(
            [{
                "scene": 1,
                "references": {"actor_ids": ["hero"], "location_id": "stage"},
                "look_ids": {"actors": {"hero": "winter"}},
            }],
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="strict",
        )

        self.assertEqual(["missing_actor_look"], [issue.code for issue in result.issues])

    def test_aggregates_all_missing_reference_bindings_for_scene(self):
        result = preflight_visual_consistency(
            [{
                "scene": 1,
                "references": {
                    "actor_ids": ["missing-a", "missing-b"],
                    "location_id": "missing-place",
                },
            }],
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="strict",
        )

        self.assertEqual(
            [
                "missing_actor_reference",
                "missing_actor_reference",
                "missing_location_reference",
            ],
            [issue.code for issue in result.issues],
        )

    def test_result_is_defensively_immutable(self):
        scenes = [{"scene": 1, "references": {"actor_ids": ["hero"], "location_id": "stage"}}]
        result = preflight_visual_consistency(
            scenes,
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="strict",
        )
        scenes[0]["references"]["actor_ids"].append("other")

        self.assertIsInstance(result.contracts, tuple)
        self.assertIsInstance(result.issues, tuple)
        self.assertEqual(("hero",), tuple(anchor.id for anchor in result.contracts[0].actors))

    def test_reports_duplicate_scene_numbers(self):
        scenes = [
            {"scene": 2, "references": {"actor_ids": ["hero"], "location_id": "stage"}},
            {"scene": 2, "references": {"actor_ids": ["hero"], "location_id": "stage"}},
        ]

        result = preflight_visual_consistency(
            scenes,
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="strict",
        )

        self.assertIn("duplicate_scene_number", [issue.code for issue in result.issues])
        self.assertFalse(result.renderable)

    def test_rejects_malformed_scene_numbers_instead_of_skipping(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            preflight_visual_consistency(
                [{"scene": "2", "references": {"actor_ids": ["hero"]}}],
                self.snapshot,
                mode="i2v",
                workflow_profile="i2v-default",
                preflight_mode="strict",
            )

    def test_uses_scene_cast_policy_for_subject_limit(self):
        snapshot = ReferenceManifestSnapshot(
            actors={
                ("hero", "default"): _anchor("actor", "hero"),
                ("friend", "default"): _anchor("actor", "friend"),
            },
            locations=self.snapshot.locations,
            revision="revision",
        )
        result = preflight_visual_consistency(
            [{"scene": 1, "references": {"actor_ids": ["hero", "friend"], "location_id": "stage"}}],
            snapshot,
            mode="ingredients",
            workflow_profile="ingredients-default",
            preflight_mode="strict",
            subject_mode="single",
            max_scene_actors=4,
        )

        self.assertIn("subject_limit_exceeded", [issue.code for issue in result.issues])

    def test_checks_ingredients_sheet_and_anchor_bindings(self):
        result = preflight_visual_consistency(
            [{
                "scene": 1,
                "references": {"actor_ids": ["hero"], "location_id": "stage"},
                "ingredients": {"anchors": [{"id": "hero"}]},
            }],
            self.snapshot,
            mode="ingredients",
            workflow_profile="ingredients-default",
            preflight_mode="strict",
        )

        self.assertEqual(
            {"missing_ingredients_sheet", "missing_ingredients_anchor"},
            {issue.code for issue in result.issues},
        )

    def test_ingredients_sheet_must_be_a_nonblank_string(self):
        for sheet in (123, " "):
            with self.subTest(sheet=sheet):
                result = preflight_visual_consistency(
                    [{
                        "scene": 1,
                        "references": {
                            "actor_ids": ["hero"],
                            "location_id": "stage",
                        },
                        "ingredients": {
                            "sheet_path": sheet,
                            "anchors": [{"id": "hero"}, {"id": "stage"}],
                        },
                    }],
                    self.snapshot,
                    mode="ingredients",
                    workflow_profile="ingredients-default",
                    preflight_mode="strict",
                )

                self.assertIn(
                    "missing_ingredients_sheet",
                    [issue.code for issue in result.issues],
                )

    def test_checks_msr_actor_and_location_roles(self):
        result = preflight_visual_consistency(
            [{"scene": 1, "references": {"actor_ids": ["hero"], "location_id": "stage"}}],
            self.snapshot,
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode="strict",
        )

        self.assertEqual(
            {"missing_msr_actor_role", "missing_msr_location_role"},
            {issue.code for issue in result.issues},
        )

    def test_rejects_blank_or_non_list_msr_roles(self):
        result = preflight_visual_consistency(
            [{
                "scene": 1,
                "references": {
                    "actor_ids": ["hero"],
                    "location_id": "stage",
                    "actor_msr_paths": "hero.png",
                    "location_msr_path": " ",
                },
            }],
            self.snapshot,
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode="strict",
        )

        self.assertEqual(
            {"missing_msr_actor_role", "missing_msr_location_role"},
            {issue.code for issue in result.issues},
        )

    def test_rejects_unsupported_continuous_transition(self):
        result = preflight_visual_consistency(
            [{
                "scene": 1,
                "transition_from_previous": "continuous",
                "references": {
                    "actor_ids": ["hero"],
                    "location_id": "stage",
                    "actor_msr_paths": ["hero.png"],
                    "location_msr_path": "stage.png",
                },
            }],
            self.snapshot,
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode="strict",
            supports_continuous_transitions=False,
        )

        self.assertIn("unsupported_continuous_transition", [issue.code for issue in result.issues])

    def test_continuous_transition_cannot_skip_an_uncontracted_scene(self):
        bound = {
            "references": {"actor_ids": ["hero"], "location_id": "stage"},
        }
        result = preflight_visual_consistency(
            [
                {"scene": 1, **bound},
                {"scene": 2, "prompt": "legacy"},
                {"scene": 3, "transition_from_previous": "continuous", **bound},
            ],
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="warn",
            supports_continuous_transitions=True,
        )

        self.assertIn(
            "unsupported_continuous_transition",
            [issue.code for issue in result.issues],
        )

    def test_continuous_transition_requires_consecutive_scene_numbers(self):
        bound = {
            "references": {"actor_ids": ["hero"], "location_id": "stage"},
        }
        result = preflight_visual_consistency(
            [
                {"scene": 1, **bound},
                {"scene": 3, "transition_from_previous": "continuous", **bound},
            ],
            self.snapshot,
            mode="i2v",
            workflow_profile="i2v-default",
            preflight_mode="strict",
            supports_continuous_transitions=True,
        )

        self.assertIn(
            "unsupported_continuous_transition",
            [issue.code for issue in result.issues],
        )

    def test_rejects_stale_stored_fingerprint(self):
        result = preflight_visual_consistency(
            [{
                "scene": 1,
                "references": {
                    "actor_ids": ["hero"],
                    "location_id": "stage",
                    "actor_msr_paths": ["hero.png"],
                    "location_msr_path": "stage.png",
                },
                "visual_consistency": {"fingerprint": "0" * 64},
            }],
            self.snapshot,
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode="strict",
        )

        self.assertIn("visual_consistency_fingerprint_mismatch", [issue.code for issue in result.issues])

    def test_stored_visual_consistency_requires_nonblank_matching_fingerprint(self):
        invalid_values = ({}, {"fingerprint": ""}, {"fingerprint": 123})
        for stored in invalid_values:
            with self.subTest(stored=stored):
                result = preflight_visual_consistency(
                    [{
                        "scene": 1,
                        "references": {"actor_ids": ["hero"], "location_id": "stage"},
                        "visual_consistency": stored,
                    }],
                    self.snapshot,
                    mode="i2v",
                    workflow_profile="i2v-default",
                    preflight_mode="strict",
                )

                self.assertIn(
                    "visual_consistency_fingerprint_mismatch",
                    [issue.code for issue in result.issues],
                )

    def test_off_bypasses_contract_checks(self):
        result = preflight_visual_consistency(
            [{"scene": 1, "references": {"actor_ids": ["missing"]}}],
            self.snapshot,
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode="off",
        )

        self.assertTrue(result.renderable)
        self.assertEqual((), result.contracts)
        self.assertEqual((), result.issues)


if __name__ == "__main__":
    unittest.main()
