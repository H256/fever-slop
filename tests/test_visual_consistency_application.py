from types import MappingProxyType
import unittest

from feverslop.application.visual_consistency import (
    build_scene_contract,
    normalize_reference_ids,
)
from feverslop.domain.visual_consistency import ReferenceAnchor
from feverslop.ports.visual_consistency import ReferenceManifestSnapshot


def anchor(
    anchor_id: str,
    kind: str,
    *,
    look_id: str = "default",
) -> ReferenceAnchor:
    return ReferenceAnchor(
        id=anchor_id,
        kind=kind,
        look_id=look_id,
        asset_role=(
            "identity-reference" if kind == "actor" else "environment-reference"
        ),
        asset_sha256=("a" if kind == "actor" else "b") * 64,
        prompt_anchor=f"Reference {kind} `{anchor_id}`: stable description",
    )


class NormalizeReferenceIdsTests(unittest.TestCase):
    def test_song_and_movie_shapes_normalize_equally(self):
        song = {
            "references": {
                "actor_ids": [" hero ", "villain"],
                "location_id": " rooftop ",
            }
        }
        movie = {
            "reference_ids": {
                "actors": [" hero ", "villain"],
                "location": " rooftop ",
            }
        }

        self.assertEqual(
            (("hero", "villain"), "rooftop"),
            normalize_reference_ids(song),
        )
        self.assertEqual(normalize_reference_ids(song), normalize_reference_ids(movie))

    def test_preserves_actor_order_and_removes_duplicates_after_trimming(self):
        scene = {
            "references": {
                "actor_ids": [" villain ", "hero", "villain", "", None, " hero "],
                "location_id": " stage ",
            }
        }

        self.assertEqual(
            (("villain", "hero"), "stage"),
            normalize_reference_ids(scene),
        )

    def test_uses_top_level_fallbacks_and_tolerates_malformed_nested_shapes(self):
        self.assertEqual(
            (("hero", "villain"), "street"),
            normalize_reference_ids(
                {
                    "reference_ids": [],
                    "references": "bad",
                    "actor_ids": (" hero ", 7, "villain"),
                    "location_id": " street ",
                }
            ),
        )
        self.assertEqual(((), ""), normalize_reference_ids(None))

    def test_malformed_nested_values_do_not_block_valid_top_level_fallbacks(self):
        self.assertEqual(
            (("hero",), "street"),
            normalize_reference_ids(
                {
                    "reference_ids": {
                        "actors": {"not": "an id list"},
                        "location": ["not", "an id"],
                    },
                    "actor_ids": ["hero"],
                    "location_id": "street",
                }
            ),
        )


class BuildSceneContractTests(unittest.TestCase):
    def setUp(self):
        self.hero = anchor("hero", "actor")
        self.hero_winter = anchor("hero", "actor", look_id="winter")
        self.villain = anchor("villain", "actor")
        self.rooftop = anchor("rooftop", "location")
        self.snapshot = ReferenceManifestSnapshot(
            actors={
                ("hero", "default"): self.hero,
                ("hero", "winter"): self.hero_winter,
                ("villain", "default"): self.villain,
            },
            locations={("rooftop", "default"): self.rooftop},
            revision="c" * 64,
        )

    def test_snapshot_is_defensively_immutable(self):
        source = {("hero", "default"): self.hero}
        snapshot = ReferenceManifestSnapshot(
            actors=source,
            locations={},
            revision="revision",
        )
        source.clear()

        self.assertEqual(self.hero, snapshot.actors[("hero", "default")])
        self.assertIsInstance(snapshot.actors, MappingProxyType)
        with self.assertRaises(TypeError):
            snapshot.actors[("other", "default")] = self.hero

    def test_builds_contract_with_default_looks_and_normalized_transition(self):
        contract = build_scene_contract(
            {
                "scene": 2,
                "references": {
                    "actor_ids": ["hero", "villain"],
                    "location_id": "rooftop",
                },
                "transition_from_previous": "continuous",
            },
            self.snapshot,
            mode="msr",
            workflow_profile="ltx-msr-v1",
        )

        self.assertEqual(2, contract.scene)
        self.assertEqual((self.hero, self.villain), contract.actors)
        self.assertEqual(self.rooftop, contract.location)
        self.assertEqual("continuous", contract.transition_from_previous)

    def test_uses_optional_per_reference_look_ids(self):
        contract = build_scene_contract(
            {
                "scene": 3,
                "reference_ids": {"actors": ["hero"], "location": "rooftop"},
                "look_ids": {
                    "actors": {"hero": " winter "},
                    "location": " default ",
                },
            },
            self.snapshot,
            mode="i2v",
            workflow_profile="ltx-i2v-v1",
        )

        self.assertEqual((self.hero_winter,), contract.actors)
        self.assertEqual(self.rooftop, contract.location)
        self.assertEqual("cut", contract.transition_from_previous)

    def test_partial_current_actor_looks_fall_back_to_existing_actor_look_ids(self):
        contract = build_scene_contract(
            {
                "scene": 3,
                "actor_ids": ["hero"],
                "look_ids": {"actors": {"other": "summer"}},
                "actor_look_ids": {"hero": "winter"},
            },
            self.snapshot,
            mode="i2v",
            workflow_profile="ltx-i2v-v1",
        )

        self.assertEqual((self.hero_winter,), contract.actors)

    def test_malformed_current_location_look_falls_back_to_legacy_location_look(self):
        rooftop_night = anchor("rooftop", "location", look_id="night")
        snapshot = ReferenceManifestSnapshot(
            actors={},
            locations={("rooftop", "night"): rooftop_night},
            revision="revision",
        )
        contract = build_scene_contract(
            {
                "scene": 3,
                "location_id": "rooftop",
                "look_ids": {"location": {"malformed": "value"}},
                "location_look_id": " night ",
            },
            snapshot,
            mode="i2v",
            workflow_profile="ltx-i2v-v1",
        )

        self.assertEqual(rooftop_night, contract.location)

    def test_mode_and_workflow_profile_are_keyword_only(self):
        with self.assertRaises(TypeError):
            build_scene_contract({"scene": 1}, self.snapshot, "msr", "profile")

    def test_rejects_invalid_scene_numbers(self):
        for scene in (None, True, 0, -1, "1"):
            with self.subTest(scene=scene), self.assertRaisesRegex(
                ValueError, "scene must be a positive integer"
            ):
                build_scene_contract(
                    {"scene": scene},
                    self.snapshot,
                    mode="msr",
                    workflow_profile="profile",
                )

    def test_missing_semantic_id_and_look_variant_name_all_context(self):
        cases = (
            (
                {"scene": 4, "actor_ids": ["missing"]},
                r"Scene 4.*actor.*missing.*default",
            ),
            (
                {
                    "scene": 5,
                    "actor_ids": ["hero"],
                    "look_ids": {"actors": {"hero": "summer"}},
                },
                r"Scene 5.*actor.*hero.*summer",
            ),
            (
                {"scene": 6, "location_id": "missing"},
                r"Scene 6.*location.*missing.*default",
            ),
        )
        for scene, message in cases:
            with self.subTest(scene=scene), self.assertRaisesRegex(ValueError, message):
                build_scene_contract(
                    scene,
                    self.snapshot,
                    mode="msr",
                    workflow_profile="profile",
                )


if __name__ == "__main__":
    unittest.main()
