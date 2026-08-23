import unittest
from dataclasses import replace

from feverslop.domain.visual_consistency import (
    ConsistencyIssue,
    ReferenceAnchor,
    SceneConsistencyContract,
    can_handoff,
)

ACTOR_HASH = "a" * 64
LOCATION_HASH = "b" * 64


def actor(anchor_id: str = "hero") -> ReferenceAnchor:
    return ReferenceAnchor(
        id=anchor_id,
        kind="actor",
        look_id=f"{anchor_id}-look",
        asset_role="identity-reference",
        asset_sha256=ACTOR_HASH,
        prompt_anchor=f"{anchor_id} wears a red leather jacket",
    )


def location(anchor_id: str = "rooftop") -> ReferenceAnchor:
    return ReferenceAnchor(
        id=anchor_id,
        kind="location",
        look_id=f"{anchor_id}-look",
        asset_role="environment-reference",
        asset_sha256=LOCATION_HASH,
        prompt_anchor="rainy neon rooftop at midnight",
    )


def contract(
    *,
    scene: int = 2,
    mode: str = "msr",
    actors: tuple[ReferenceAnchor, ...] = (),
    location_anchor: ReferenceAnchor | None = None,
    transition: str = "continuous",
) -> SceneConsistencyContract:
    return SceneConsistencyContract.create(
        scene=scene,
        mode=mode,
        workflow_profile="ltx-msr-v1",
        actors=actors,
        location=location_anchor,
        transition_from_previous=transition,
    )


class ReferenceAnchorTests(unittest.TestCase):
    def test_rejects_empty_identity_and_prompt_fields(self):
        fields = ("id", "look_id", "asset_role", "prompt_anchor")
        values = {
            "id": "hero",
            "kind": "actor",
            "look_id": "hero-look",
            "asset_role": "identity-reference",
            "asset_sha256": ACTOR_HASH,
            "prompt_anchor": "red jacket",
        }

        for field in fields:
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field.replace("_", " "),
            ):
                ReferenceAnchor(**(values | {field: "  "}))

    def test_rejects_unknown_kind_and_invalid_hash(self):
        with self.assertRaisesRegex(ValueError, "kind must be actor or location"):
            ReferenceAnchor(
                id="hero",
                kind="prop",
                look_id="hero-look",
                asset_role="reference",
                asset_sha256=ACTOR_HASH,
                prompt_anchor="red jacket",
            )
        with self.assertRaisesRegex(ValueError, "64-character hexadecimal"):
            ReferenceAnchor(
                id="hero",
                kind="actor",
                look_id="hero-look",
                asset_role="reference",
                asset_sha256="z" * 64,
                prompt_anchor="red jacket",
            )

    def test_rejects_uppercase_sha256(self):
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            ReferenceAnchor(
                id="hero",
                kind="actor",
                look_id="hero-look",
                asset_role="reference",
                asset_sha256="A" * 64,
                prompt_anchor="red jacket",
            )


class ConsistencyIssueTests(unittest.TestCase):
    def test_validates_required_fields_and_severity(self):
        with self.assertRaisesRegex(ValueError, "severity must be warning or error"):
            ConsistencyIssue(
                code="missing-anchor",
                scene=1,
                severity="info",
                message="No actor anchor",
            )
        with self.assertRaisesRegex(ValueError, "code, scene, and message are required"):
            ConsistencyIssue(
                code="",
                scene=1,
                severity="warning",
                message="No actor anchor",
            )

    def test_rejects_invalid_scene_numbers(self):
        for scene in (True, 0, -1, "1", 1.5):
            with self.subTest(scene=scene), self.assertRaisesRegex(
                ValueError, "scene must be a positive integer",
            ):
                ConsistencyIssue(
                    code="missing-anchor",
                    scene=scene,
                    severity="warning",
                    message="No actor anchor",
                )


class SceneConsistencyContractTests(unittest.TestCase):
    def test_equivalent_contracts_have_equal_fingerprints(self):
        first = contract(actors=(actor("hero"),), location_anchor=location())
        second = contract(actors=(actor("hero"),), location_anchor=location())

        self.assertEqual("feverslop.visual-consistency/v1", first.schema)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(64, len(first.fingerprint))
        self.assertIsInstance(first.to_dict()["scene"], int)

    def test_fingerprint_is_independent_of_mapping_key_order(self):
        original = contract(actors=(actor(),), location_anchor=location())
        payload = original.to_dict()
        reordered = {key: payload[key] for key in reversed(payload)}

        restored = SceneConsistencyContract.from_dict(reordered)

        self.assertEqual(original.fingerprint, restored.fingerprint)

    def test_rejects_duplicate_actor_ids_with_exact_error(self):
        with self.assertRaisesRegex(
            ValueError, r"^duplicate actor id: hero$",
        ):
            contract(actors=(actor("hero"), actor("hero")))

    def test_to_dict_from_dict_round_trip_is_exact(self):
        original = contract(
            actors=(actor("hero"), actor("villain")),
            location_anchor=location(),
        )

        payload = original.to_dict()
        restored = SceneConsistencyContract.from_dict(payload)

        self.assertEqual(original, restored)
        self.assertEqual(payload, restored.to_dict())
        self.assertIsInstance(restored.actors, tuple)

    def test_from_dict_rejects_a_stale_fingerprint(self):
        original = contract(actors=(actor(),), location_anchor=location())
        payload = original.to_dict() | {"scene": 3}

        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            SceneConsistencyContract.from_dict(payload)

    def test_direct_construction_rejects_a_stale_fingerprint(self):
        original = contract(actors=(actor(),), location_anchor=location())

        for stale in (
            lambda: replace(original, scene=3),
            lambda: replace(original, fingerprint="0" * 64),
        ):
            with self.subTest(stale=stale), self.assertRaisesRegex(
                ValueError, "fingerprint does not match canonical payload",
            ):
                stale()

    def test_prompt_anchor_text_is_bounded(self):
        original = contract(
            actors=(actor("hero"), actor("villain")),
            location_anchor=location(),
        )

        full_text = original.prompt_anchor_text()
        bounded = original.prompt_anchor_text(max_chars=24)

        self.assertIn("hero wears a red leather jacket", full_text)
        self.assertIn("rainy neon rooftop at midnight", full_text)
        self.assertEqual(full_text[:24], bounded)
        self.assertEqual(24, len(bounded))

    def test_create_validates_contract_fields_and_anchor_kinds(self):
        invalid_cases = (
            ({"mode": "txt2vid"}, "mode must be ingredients, msr, or i2v"),
            ({"transition": "fade"}, "transition must be cut or continuous"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, message,
            ):
                contract(**overrides)

        with self.assertRaisesRegex(ValueError, "actors must have kind actor"):
            contract(actors=(location(),))
        with self.assertRaisesRegex(ValueError, "location must have kind location"):
            contract(location_anchor=actor())

    def test_rejects_invalid_scene_numbers(self):
        for scene in (True, 0, -1, "1", 1.5):
            with self.subTest(scene=scene), self.assertRaisesRegex(
                ValueError, "scene must be a positive integer",
            ):
                contract(scene=scene)


class HandoffTests(unittest.TestCase):
    def test_allows_compatible_continuous_handoff(self):
        previous = contract(
            scene=1,
            mode="msr",
            actors=(actor("hero"),),
            location_anchor=location(),
            transition="cut",
        )
        current = contract(
            mode="i2v",
            actors=(actor("hero"), actor("villain")),
            location_anchor=location(),
        )

        self.assertTrue(can_handoff(previous, current))

    def test_rejects_handoff_for_each_missing_condition(self):
        previous = contract(
            scene=1,
            mode="msr",
            actors=(actor("hero"),),
            location_anchor=location(),
        )
        candidates = (
            contract(
                actors=(actor("hero"),),
                location_anchor=location(),
                transition="cut",
            ),
            contract(
                mode="ingredients",
                actors=(actor("hero"),),
                location_anchor=location(),
            ),
            contract(actors=(actor("hero"),), location_anchor=None),
            contract(
                actors=(actor("hero"),),
                location_anchor=location("street"),
            ),
            contract(
                actors=(actor("villain"),),
                location_anchor=location(),
            ),
        )

        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertFalse(can_handoff(previous, candidate))


if __name__ == "__main__":
    unittest.main()
