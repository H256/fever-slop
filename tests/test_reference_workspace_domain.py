from __future__ import annotations

import unittest

from feverslop.domain.reference_workspace import (
    ReferenceAsset,
    ReferenceKind,
    ReferenceLook,
    ReferenceProvenance,
    ReferenceWorkspaceSnapshot,
    SceneReferenceAssignment,
)


class ReferenceKindParsingTests(unittest.TestCase):
    def test_all_kinds_exist(self):
        expected = {
            "actor", "location", "background", "style",
            "storyboard_frame", "storyboard_page",
            "msr_sheet", "ingredients_sheet", "continuity",
        }
        actual = {kind.value for kind in ReferenceKind}
        self.assertEqual(expected, actual)

    def test_kind_from_value(self):
        self.assertEqual(ReferenceKind.ACTOR, ReferenceKind("actor"))
        self.assertEqual(ReferenceKind.MSR_SHEET, ReferenceKind("msr_sheet"))


class SceneReferenceAssignmentTests(unittest.TestCase):
    def test_basic_assignment(self):
        a = SceneReferenceAssignment(
            scene_number=3,
            actor_ids=("hero_villain",),
            location_ids=("lab_main",),
        )
        self.assertEqual(3, a.scene_number)
        self.assertEqual(("hero_villain",), a.actor_ids)
        self.assertEqual(("lab_main",), a.location_ids)

    def test_look_ids_default_empty(self):
        a = SceneReferenceAssignment(scene_number=1, actor_ids=("hero",))
        self.assertEqual({}, a.actor_look_ids)

    def test_look_ids_stored(self):
        a = SceneReferenceAssignment(
            scene_number=1,
            actor_ids=("hero",),
            actor_look_ids={"hero": "look_night"},
        )
        self.assertEqual({"hero": "look_night"}, a.actor_look_ids)

    def test_duplicate_actor_ids_deduplicated(self):
        a = SceneReferenceAssignment(
            scene_number=1,
            actor_ids=("hero", "villain", "hero"),
        )
        self.assertEqual(("hero", "villain"), a.actor_ids)

    def test_empty_strings_filtered(self):
        a = SceneReferenceAssignment(
            scene_number=1,
            actor_ids=("hero", "", "  "),
        )
        self.assertEqual(("hero",), a.actor_ids)

    def test_invalid_scene_number_zero(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SceneReferenceAssignment(scene_number=0, actor_ids=())

    def test_invalid_scene_number_negative(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SceneReferenceAssignment(scene_number=-1, actor_ids=())

    def test_invalid_scene_number_bool(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SceneReferenceAssignment(scene_number=True, actor_ids=())

    def test_multi_subject_assignment_respects_existing_limit(self):
        with self.assertRaisesRegex(ValueError, "at most 4 actors"):
            assignment = SceneReferenceAssignment(
                scene_number=3,
                actor_ids=("a", "b", "c", "d", "e"),
            )
            assignment.validate_against(
                known_actor_ids=["a", "b", "c", "d", "e"],
                known_location_ids=[],
                max_scene_actors=4,
            )

    def test_unknown_actor_id_validation(self):
        assignment = SceneReferenceAssignment(
            scene_number=3,
            actor_ids=("hero", "unknown_actor"),
        )
        issues = assignment.validate_against(
            known_actor_ids=["hero"],
            known_location_ids=[],
        )
        self.assertIn("Unknown actor ID: unknown_actor", issues)

    def test_unknown_location_id_validation(self):
        assignment = SceneReferenceAssignment(
            scene_number=3,
            location_ids=("unknown_loc",),
        )
        issues = assignment.validate_against(
            known_actor_ids=[],
            known_location_ids=["lab_main"],
        )
        self.assertIn("Unknown location ID: unknown_loc", issues)

    def test_four_actors_valid(self):
        assignment = SceneReferenceAssignment(
            scene_number=1,
            actor_ids=("a", "b", "c", "d"),
        )
        issues = assignment.validate_against(
            known_actor_ids=["a", "b", "c", "d"],
            known_location_ids=[],
            max_scene_actors=4,
        )
        self.assertEqual([], issues)

    def test_look_id_default_label(self):
        look = ReferenceLook(id="default", reference_id="hero")
        self.assertEqual("", look.label)

    def test_provenance_minimal(self):
        prov = ReferenceProvenance(source="import")
        self.assertEqual("import", prov.source)
        self.assertEqual("", prov.generated_at)


class ReferenceWorkspaceSnapshotTests(unittest.TestCase):
    def _snapshot(self, *assets, **kwargs):
        return ReferenceWorkspaceSnapshot(
            assets=assets,
            assignments=kwargs.get("assignments", ()),
            revision=kwargs.get("revision", "r1"),
            project_id=kwargs.get("project_id", "demo"),
        )

    def test_get_asset_by_id(self):
        hero = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR, label="Hero")
        snap = self._snapshot(hero)
        found = snap.get_asset("hero")
        self.assertIsNotNone(found)
        self.assertEqual("hero", found.id)
        self.assertIsNone(snap.get_asset("nonexistent"))

    def test_filter_kinds(self):
        hero = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR)
        loc = ReferenceAsset(id="loc", kind=ReferenceKind.LOCATION)
        snap = self._snapshot(hero, loc)
        filtered = snap.filter_assets(kinds=[ReferenceKind.LOCATION])
        self.assertEqual(("loc",), tuple(a.id for a in filtered))

    def test_filter_stale_only(self):
        fresh = ReferenceAsset(id="fresh", kind=ReferenceKind.ACTOR, stale=False)
        old = ReferenceAsset(id="old", kind=ReferenceKind.ACTOR, stale=True)
        snap = self._snapshot(fresh, old)
        self.assertEqual(("old",), tuple(a.id for a in snap.filter_assets(stale_only=True)))

    def test_filter_missing_only(self):
        present = ReferenceAsset(id="present", kind=ReferenceKind.ACTOR, exists=True)
        gone = ReferenceAsset(id="gone", kind=ReferenceKind.ACTOR, exists=False)
        snap = self._snapshot(present, gone)
        self.assertEqual(("gone",), tuple(a.id for a in snap.filter_assets(missing_only=True)))

    def test_filter_combination(self):
        present_actor = ReferenceAsset(id="a1", kind=ReferenceKind.ACTOR, exists=True)
        gone_loc = ReferenceAsset(id="l1", kind=ReferenceKind.LOCATION, exists=False)
        snap = self._snapshot(present_actor, gone_loc)
        result = snap.filter_assets(kinds=[ReferenceKind.LOCATION], missing_only=True)
        self.assertEqual(("l1",), tuple(a.id for a in result))

    def test_get_assignments_for_scene(self):
        a1 = SceneReferenceAssignment(scene_number=3, actor_ids=("hero",))
        a2 = SceneReferenceAssignment(scene_number=5, actor_ids=("villain",))
        snap = self._snapshot(assignments=(a1, a2))
        scene3 = snap.get_assignments_for_scene(3)
        self.assertEqual(1, len(scene3))
        self.assertEqual(3, scene3[0].scene_number)
        self.assertEqual((), snap.get_assignments_for_scene(99))

    def test_get_assignments_for_asset(self):
        a1 = SceneReferenceAssignment(scene_number=3, actor_ids=("hero", "villain"))
        a2 = SceneReferenceAssignment(scene_number=5, actor_ids=("villain",))
        snap = self._snapshot(assignments=(a1, a2))
        hero_assignments = snap.get_assignments_for_asset("hero")
        self.assertEqual((3,), tuple(a.scene_number for a in hero_assignments))
        villain_assignments = snap.get_assignments_for_asset("villain")
        self.assertEqual((3, 5), tuple(a.scene_number for a in villain_assignments))

    def test_scenes_using_asset(self):
        a1 = SceneReferenceAssignment(scene_number=5, location_ids=("lab",))
        a2 = SceneReferenceAssignment(scene_number=2, location_ids=("lab",))
        snap = self._snapshot(assignments=(a1, a2))
        self.assertEqual((2, 5), snap.scenes_using_asset("lab"))

    def test_asset_default_fields(self):
        a = ReferenceAsset(id="test", kind=ReferenceKind.ACTOR)
        self.assertEqual("", a.label)
        self.assertEqual("", a.path)
        self.assertEqual(0, a.width)
        self.assertEqual(0, a.height)
        self.assertTrue(a.exists)
        self.assertEqual((), a.looks)
        self.assertFalse(a.stale)
        self.assertEqual("", a.generation_state)


if __name__ == "__main__":
    unittest.main()
