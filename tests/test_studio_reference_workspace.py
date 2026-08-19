from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.domain.reference_workspace import (
    PropInteraction,
    ReferenceAsset,
    ReferenceKind,
    ReferenceProvenance,
    ReferenceWorkspaceSnapshot,
    SceneReferenceAssignment,
)
from feverslop.studio.reference_workspace_service import (
    GenerationCommand,
    ReferenceWorkspaceService,
    _assignment_from_dict,
    _assignment_to_dict,
    _asset_to_dict,
)


class _MockLibrary:
    def __init__(self, props: list[str] | None = None):
        self.snapshots: dict[str, ReferenceWorkspaceSnapshot] = {}
        self.assignments_data: dict[str, list[dict]] = {}
        self.import_results: dict[str, ReferenceAsset] = {}
        self._props = list(props or [])
        self.saved_assignments: dict[str, tuple] = {}

    def load(self, project_id: str) -> ReferenceWorkspaceSnapshot:
        return self.snapshots.get(project_id, ReferenceWorkspaceSnapshot(assets=(), assignments=()))

    def save_assignments(self, project_id: str, assignments: tuple[SceneReferenceAssignment, ...], expected_revision: str) -> str:
        self.saved_assignments[project_id] = assignments
        return "r2"

    def add_asset(self, project_id: str, asset: ReferenceAsset) -> ReferenceAsset:
        return asset

    def get_known_actor_ids(self, project_id: str) -> list[str]:
        return ["hero", "villain"]

    def get_known_location_ids(self, project_id: str) -> list[str]:
        return ["lab", "office"]

    def get_background_ids(self, project_id: str) -> list[str]:
        return []

    def get_known_prop_ids(self, project_id: str) -> list[str]:
        return list(self._props)

    def get_max_scene_actors(self, project_id: str) -> int:
        return 4

    def get_invalidated_artifacts(self, project_id: str, changed_scenes: list[int], changed_actor_ids=None, changed_location_ids=None):
        result = {}
        for sn in changed_scenes:
            result.setdefault("msr_sheets", []).append(f"scene_{sn}_msr.png")
        return result

    def import_asset(self, project_id: str, source_path: Path, asset: ReferenceAsset) -> ReferenceAsset:
        return self.import_results.get(source_path.name, ReferenceAsset(
            id=asset.id, kind=asset.kind, label=asset.label, path=f"movie/references/imported/{asset.id}.png", width=640, height=480
        ))


class AssignmentSerializationTests(unittest.TestCase):
    def test_roundtrip(self):
        a = SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), location_ids=("lab",), actor_look_ids={"hero": "night"})
        d = _assignment_to_dict(a)
        restored = _assignment_from_dict(d)
        self.assertEqual(a.scene_number, restored.scene_number)
        self.assertEqual(a.actor_ids, restored.actor_ids)
        self.assertEqual(a.location_ids, restored.location_ids)

    def test_empty_look_ids(self):
        a = SceneReferenceAssignment(scene_number=1, actor_ids=("hero",))
        d = _assignment_to_dict(a)
        self.assertEqual({}, d["actor_look_ids"])

    def test_roundtrip_preserves_props(self):
        a = SceneReferenceAssignment(
            scene_number=3,
            actor_ids=("hero",),
            prop_ids=("guitar",),
            prop_interactions=(PropInteraction(
                actor_id="hero", prop_id="guitar", action="holds", relationship="grabs",
            ),),
        )
        d = _assignment_to_dict(a)
        self.assertEqual(["guitar"], d["prop_ids"])
        self.assertEqual(
            [{"actor_id": "hero", "prop_id": "guitar", "action": "holds", "relationship": "grabs"}],
            d["prop_interactions"],
        )
        restored = _assignment_from_dict(d)
        self.assertEqual(a.prop_ids, restored.prop_ids)
        self.assertEqual(a.prop_interactions, restored.prop_interactions)

    def test_from_dict_without_prop_keys(self):
        restored = _assignment_from_dict({"scene_number": 1})
        self.assertEqual((), restored.prop_ids)
        self.assertEqual((), restored.prop_interactions)


class AssetDictTests(unittest.TestCase):
    def test_asset_with_provenance(self):
        a = ReferenceAsset(
            id="hero", kind=ReferenceKind.ACTOR, label="Hero",
            provenance=ReferenceProvenance(source="import", generated_at="2026-01-01"),
        )
        d = _asset_to_dict(a)
        self.assertEqual("hero", d["id"])
        self.assertEqual("actor", d["kind"])
        self.assertEqual("import", d["provenance"]["source"])

    def test_asset_without_provenance(self):
        a = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR)
        d = _asset_to_dict(a)
        self.assertEqual("", d["provenance"]["source"])


class ReferenceWorkspaceServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._mock = _MockLibrary()
        hero = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR, label="Hero")
        lab = ReferenceAsset(id="lab", kind=ReferenceKind.LOCATION, label="Lab")
        self._mock.snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(hero, lab),
            assignments=(SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),),
            revision="r1",
            project_id="proj",
        )

    def _service_with_mock(self):
        with patch(
            "feverslop.studio.reference_workspace_service.ProjectReferenceLibrary",
            return_value=self._mock,
        ):
            return ReferenceWorkspaceService(project_root=Path("/tmp/test"))

    def test_load_library(self):
        svc = self._service_with_mock()
        snap = svc.load_library("proj")
        self.assertEqual(2, len(snap.assets))

    def test_filter_by_kind(self):
        svc = self._service_with_mock()
        result = svc.filter_library("proj", kinds=[ReferenceKind.ACTOR])
        self.assertEqual(("hero",), tuple(a.id for a in result))

    def test_get_asset_found(self):
        svc = self._service_with_mock()
        a = svc.get_asset("proj", "hero")
        self.assertIsNotNone(a)
        self.assertEqual("hero", a.id)

    def test_get_asset_not_found(self):
        svc = self._service_with_mock()
        self.assertIsNone(svc.get_asset("proj", "nonexistent"))

    def test_scenes_using_asset(self):
        svc = self._service_with_mock()
        scenes = svc.scenes_using_asset("proj", "hero")
        self.assertEqual((3,), scenes)

    def test_preview_valid_assignment(self):
        svc = self._service_with_mock()
        result = svc.preview_assignment("proj", 5, actor_ids=("hero",))
        self.assertTrue(result.success)

    def test_preview_invalid_actor(self):
        svc = self._service_with_mock()
        result = svc.preview_assignment("proj", 5, actor_ids=("unknown",))
        self.assertFalse(result.success)
        self.assertEqual("validation_errors", result.error.code if result.error else None)

    def test_preview_exceeds_actor_limit(self):
        svc = self._service_with_mock()
        result = svc.preview_assignment("proj", 5, actor_ids=("a", "b", "c", "d", "e"))
        self.assertFalse(result.success)

    def test_save_assignments(self):
        svc = self._service_with_mock()
        result = svc.save_assignments("proj", (
            {"scene_number": 5, "actor_ids": ["hero"]},
        ), "r1")
        self.assertTrue(result.success)
        self.assertEqual("r2", result.data["new_revision"])

    def test_save_assignments_roundtrip_preserves_props(self):
        self._mock = _MockLibrary(props=["guitar"])
        hero = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR, label="Hero")
        lab = ReferenceAsset(id="lab", kind=ReferenceKind.LOCATION, label="Lab")
        self._mock.snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(hero, lab),
            assignments=(SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),),
            revision="r1",
            project_id="proj",
        )
        svc = self._service_with_mock()
        result = svc.save_assignments("proj", (
            {
                "scene_number": 5,
                "actor_ids": ["hero"],
                "prop_ids": ["guitar"],
                "prop_interactions": [{"actor_id": "hero", "prop_id": "guitar", "action": "holds"}],
            },
        ), "r1")
        self.assertTrue(result.success)
        self.assertEqual("r2", result.data["new_revision"])
        saved = self._mock.saved_assignments["proj"]
        a = [item for item in saved if item.scene_number == 5][0]
        self.assertEqual(("guitar",), a.prop_ids)
        self.assertEqual(
            (PropInteraction(actor_id="hero", prop_id="guitar", action="holds"),),
            a.prop_interactions,
        )

    def test_save_assignments_unknown_prop_rejected(self):
        svc = self._service_with_mock()
        result = svc.save_assignments("proj", (
            {
                "scene_number": 5,
                "actor_ids": ["hero"],
                "prop_ids": ["guitar"],
            },
        ), "r1")
        self.assertFalse(result.success)
        self.assertEqual("validation_errors", result.error.code if result.error else None)
        self.assertNotIn("proj", self._mock.saved_assignments)

    def test_import_asset(self):
        svc = self._service_with_mock()
        result = svc.import_asset("proj", Path("/tmp/hero.png"), {"id": "hero", "kind": "actor", "label": "Hero"})
        self.assertTrue(result.success)
        self.assertIn("asset", result.data)

    def test_import_missing_id(self):
        svc = self._service_with_mock()
        result = svc.import_asset("proj", Path("/tmp/hero.png"), {"kind": "actor"})
        self.assertFalse(result.success)
        self.assertEqual("missing_id", result.error.code if result.error else None)

    def test_queue_generation_storyboard(self):
        svc = self._service_with_mock()
        cmd = GenerationCommand(action="storyboard_frame", scene_number=5, reference_ids=("frame_1",))
        result = svc.queue_generation("proj", cmd)
        self.assertTrue(result.success)
        self.assertIn("job_id", result.data)

    def test_queue_generation_msr(self):
        svc = self._service_with_mock()
        cmd = GenerationCommand(action="msr_sheet", scene_number=3, actor_ids=("hero",), location_ids=("lab",))
        result = svc.queue_generation("proj", cmd)
        self.assertTrue(result.success)

    def test_queue_generation_ingredients(self):
        svc = self._service_with_mock()
        cmd = GenerationCommand(action="ingredients_sheet", scene_number=3, actor_ids=("hero",), background_ids=("bg1",))
        result = svc.queue_generation("proj", cmd)
        self.assertTrue(result.success)

    def test_queue_generation_unknown_action(self):
        svc = self._service_with_mock()
        cmd = GenerationCommand(action="unknown_action_xyz")
        result = svc.queue_generation("proj", cmd)
        self.assertFalse(result.success)
        self.assertEqual("unknown_action", result.error.code if result.error else None)

    def test_check_stale(self):
        self._mock.snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(
                ReferenceAsset(id="stale_asset", kind=ReferenceKind.ACTOR, stale=True),
                ReferenceAsset(id="fresh_asset", kind=ReferenceKind.ACTOR, stale=False),
            ),
            assignments=(),
            revision="r1",
            project_id="proj",
        )
        svc = self._service_with_mock()
        stale = svc.check_stale("proj")
        self.assertEqual(("stale_asset",), stale.stale_assets)


if __name__ == "__main__":
    unittest.main()
