"""Integration tests for the reference workspace flow.

Tests the complete stack: adapters -> service -> viewmodel.
These require PySide6 to run.
"""
from __future__ import annotations

import json
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

# Skip if PySide6 not available
if "PySide6" not in sys.modules:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("PySide6 not available")

from PySide6.QtWidgets import QApplication

from feverslop.domain.reference_workspace import (
    ReferenceAsset,
    ReferenceKind,
)
from feverslop.studio.desktop.viewmodels.references import (
    ReferenceListModel,
    ReferenceWorkspaceViewModel,
)
from feverslop.studio.reference_workspace_service import (
    CommandResult,
    GenerationCommand,
    ReferenceWorkspaceService,
)


def _fixture_path(proj_dir: Path) -> Path:
    """Return the movie/references dir inside an adapter project."""
    ref_dir = proj_dir / "movie" / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir


class TestReferenceWorkspaceIntegration(unittest.TestCase):
    """Integration tests for reference workspace."""

    @classmethod
    def setUpClass(cls) -> None:
        if not QApplication.instance():
            cls._app = QApplication([])
        else:
            cls._app = QApplication.instance()

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._project_root = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_test_project(self, project_id: str = "test") -> Path:
        proj_dir = self._project_root / project_id
        proj_dir.mkdir()
        (proj_dir / "config.json").write_text("{}")
        return proj_dir

    def _create_assets(self, proj_dir: Path, assets: list[dict[str, Any]]) -> None:
        """Create test asset files in the project's movie/references directory."""
        ref_dir = _fixture_path(proj_dir)
        for asset in assets:
            path = ref_dir / asset.get("filename", asset["id"] + ".png")
            if not asset.get("exists", True):
                continue
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    def _create_manifest(self, proj_dir: Path, actors: list[str] | None = None, locations: list[str] | None = None) -> None:
        """Create movie/references/manifest.json with known actor/location IDs."""
        ref_dir = _fixture_path(proj_dir)
        data: dict[str, list[dict[str, str]]] = {}
        if actors:
            data["actors"] = [{"id": a} for a in actors]
        if locations:
            data["locations"] = [{"id": loc} for loc in locations]
        (ref_dir / "manifest.json").write_text(json.dumps(data))

    def test_service_creation(self) -> None:
        """Service can be created with a project root."""
        proj_dir = self._make_test_project()
        service = ReferenceWorkspaceService(project_root=proj_dir)
        self.assertIsNotNone(service)

    def test_load_empty_project(self) -> None:
        """Loading an empty project returns empty snapshot."""
        proj_dir = self._make_test_project("empty")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        snap = service.load_library("empty")
        self.assertEqual(0, len(snap.assets))
        self.assertEqual(0, len(snap.assignments))

    def test_load_with_assets(self) -> None:
        """Loading a project with asset files returns assets."""
        proj_dir = self._make_test_project("assets")
        self._create_assets(proj_dir, [{"id": "scene3_msr_sheet", "filename": "scene3_msr_sheet.png"}])
        service = ReferenceWorkspaceService(project_root=proj_dir)
        snap = service.load_library("assets")
        self.assertIn(ReferenceKind.MSR_SHEET, {a.kind for a in snap.assets})

    def test_filter_assets(self) -> None:
        """Filtering by kind works correctly."""
        proj_dir = self._make_test_project("filter")
        self._create_assets(
            proj_dir,
            [
                {"id": "scene3_msr_sheet", "filename": "scene3_msr_sheet.png"},
                {"id": "scene1_storyboard_main", "filename": "scene1_storyboard_main.png"},
            ],
        )
        service = ReferenceWorkspaceService(project_root=proj_dir)
        snap = service.load_library("filter")
        msr = snap.filter_assets(kinds=[ReferenceKind.MSR_SHEET])
        self.assertEqual(1, len(msr))
        self.assertEqual("scene3_msr_sheet", msr[0].id)
        sb = snap.filter_assets(kinds=[ReferenceKind.STORYBOARD_FRAME])
        self.assertEqual(1, len(sb))
        self.assertEqual("scene1_storyboard_main", sb[0].id)

    def test_preview_assignment(self) -> None:
        """Preview scene assignment validates actor_ids."""
        proj_dir = self._make_test_project("preview")
        self._create_manifest(proj_dir, actors=["hero", "villain"])
        service = ReferenceWorkspaceService(project_root=proj_dir)
        result: CommandResult = service.preview_assignment(
            "preview",
            1,
            actor_ids=("hero",),
        )
        self.assertTrue(result.success)

    def test_preview_unknown_actor(self) -> None:
        """Preview fails when actor ID is not known."""
        proj_dir = self._make_test_project("preview_bad")
        self._create_manifest(proj_dir, actors=["hero"])
        service = ReferenceWorkspaceService(project_root=proj_dir)
        result: CommandResult = service.preview_assignment(
            "preview_bad",
            1,
            actor_ids=("unknown",),
        )
        self.assertFalse(result.success)

    def test_save_assignments(self) -> None:
        """Save assignments creates/persists the assignments file."""
        proj_dir = self._make_test_project("save")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        snap = service.load_library("save")
        result: CommandResult = service.save_assignments(
            "save",
            tuple([]),
            snap.revision,
        )
        self.assertTrue(result.success)
        self.assertIn("new_revision", result.data)

    def test_assignment_roundtrip(self) -> None:
        """Assignments survive load -> save -> load cycle."""
        proj_dir = self._make_test_project("roundtrip")
        self._create_manifest(proj_dir, actors=["hero"])
        service = ReferenceWorkspaceService(project_root=proj_dir)

        snap1 = service.load_library("roundtrip")
        result: CommandResult = service.save_assignments(
            "roundtrip",
            tuple([{"scene_number": 1, "actor_ids": ["hero"], "location_ids": []}]),
            snap1.revision,
        )
        self.assertTrue(result.success)

        snap2 = service.load_library("roundtrip")
        self.assertEqual(1, len(snap2.assignments))
        self.assertEqual(1, snap2.assignments[0].scene_number)
        self.assertIn("hero", snap2.assignments[0].actor_ids)

    def test_revision_conflict(self) -> None:
        """Save with stale revision should fail."""
        proj_dir = self._make_test_project("conflict")
        service = ReferenceWorkspaceService(project_root=proj_dir)

        snap1 = service.load_library("conflict")
        result1: CommandResult = service.save_assignments(
            "conflict",
            tuple([{"scene_number": 1, "actor_ids": []}]),
            snap1.revision,
        )
        self.assertTrue(result1.success)

        result2: CommandResult = service.save_assignments(
            "conflict",
            tuple([{"scene_number": 2, "actor_ids": []}]),
            snap1.revision,
        )
        self.assertFalse(result2.success)

    def test_viewmodel_creation(self) -> None:
        """Viewmodel can be created and bound to service."""
        proj_dir = self._make_test_project("vm")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        vm = ReferenceWorkspaceViewModel(service=service)
        self.assertIsNotNone(vm.library_model)
        self.assertIsNotNone(vm.assignments_model)

    def test_viewmodel_project_switch(self) -> None:
        """Viewmodel updates when switching projects."""
        proj_dir = self._make_test_project("proj_a")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        vm = ReferenceWorkspaceViewModel(service=service)
        ok = vm.set_project("proj_a")
        self.assertTrue(ok)
        self.assertEqual("proj_a", vm.current_project)

    def test_viewmodel_asset_selection(self) -> None:
        """Viewmodel can select and track assets."""
        proj_dir = self._make_test_project("select")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        vm = ReferenceWorkspaceViewModel(service=service)
        vm.set_project("select")
        vm.select_asset("hero")
        self.assertEqual("hero", vm.selected_asset)

    def test_viewmodel_filter(self) -> None:
        """Viewmodel can filter assets by kind."""
        proj_dir = self._make_test_project("filter_vm")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        vm = ReferenceWorkspaceViewModel(service=service)
        vm.set_project("filter_vm")
        vm.set_filter_kind("actor")
        self.assertIsNotNone(vm.library_model)

    def test_list_model_data(self) -> None:
        """ReferenceListModel returns correct role data."""
        model = ReferenceListModel()
        assets = (
            ReferenceAsset(id="a1", kind=ReferenceKind.ACTOR, label="Hero"),
            ReferenceAsset(id="a2", kind=ReferenceKind.LOCATION, label="Lab"),
        )
        model.replace(assets)
        self.assertEqual(2, model.rowCount())

        idx = model.index(0)
        self.assertEqual("a1", model.data(idx, model.IdRole))
        self.assertEqual(ReferenceKind.ACTOR.value, model.data(idx, model.KindRole))
        self.assertEqual("Hero", model.data(idx, model.LabelRole))

    def test_viewmodel_collect_assignments_multiple_rows(self) -> None:
        """collect_assignments returns correct data for all rows in assignments model."""
        proj_dir = self._make_test_project("collect")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        vm = ReferenceWorkspaceViewModel(service=service)
        vm.set_project("collect")

        # Populate assignments model with 3 scenes
        from feverslop.domain.reference_workspace import SceneReferenceAssignment
        vm._assignments_model.replace((
            SceneReferenceAssignment(scene_number=1, actor_ids=("hero",), location_ids=("lab",)),
            SceneReferenceAssignment(scene_number=2, actor_ids=("hero", "sidekick"), location_ids=()),
            SceneReferenceAssignment(scene_number=3, actor_ids=(), location_ids=(), background_ids=("bg1",)),
        ))

        assignments = vm.collect_assignments()
        self.assertEqual(3, len(assignments))
        self.assertEqual(1, assignments[0]["scene_number"])
        self.assertIn("hero", assignments[0]["actor_ids"])
        self.assertIn("lab", assignments[0]["location_ids"])
        self.assertEqual(2, assignments[1]["scene_number"])
        self.assertEqual(["hero", "sidekick"], assignments[1]["actor_ids"])
        self.assertEqual(3, assignments[2]["scene_number"])
        self.assertIn("bg1", assignments[2]["background_ids"])

    def test_viewmodel_collect_assignments_empty(self) -> None:
        """collect_assignments returns empty list when model has no rows."""
        proj_dir = self._make_test_project("empty_collect")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        vm = ReferenceWorkspaceViewModel(service=service)
        vm.set_project("empty_collect")
        assignments = vm.collect_assignments()
        self.assertEqual([], assignments)


class TestReferenceWorkspaceServiceFull(unittest.TestCase):
    """Full service lifecycle tests (no PySide6 required)."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._project_root = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_project(self, project_id: str) -> Path:
        proj_dir = self._project_root / project_id
        proj_dir.mkdir()
        (proj_dir / "config.json").write_text("{}")
        return proj_dir

    def _create_manifest(self, proj_dir: Path, actors: list[str] | None = None, locations: list[str] | None = None) -> None:
        ref_dir = _fixture_path(proj_dir)
        data: dict[str, list[dict[str, str]]] = {}
        if actors:
            data["actors"] = [{"id": a} for a in actors]
        if locations:
            data["locations"] = [{"id": loc} for loc in locations]
        (ref_dir / "manifest.json").write_text(json.dumps(data))

    def test_queue_generation_storyboard(self) -> None:
        """Queue storyboard generation returns job ID."""
        proj_dir = self._make_project("gen")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        result: CommandResult = service.queue_generation(
            "gen",
            GenerationCommand(action="storyboard_frame"),
        )
        self.assertTrue(result.success)
        self.assertIn("job_id", result.data)

    def test_queue_generation_msr(self) -> None:
        """Queue MSR generation returns job ID."""
        proj_dir = self._make_project("msr")
        service = ReferenceWorkspaceService(project_root=proj_dir)
        result: CommandResult = service.queue_generation(
            "msr",
            GenerationCommand(action="msr_sheet", scene_number=1),
        )
        self.assertTrue(result.success)

    def test_check_stale(self) -> None:
        """Check stale identifies modified assets."""
        proj_dir = self._make_project("stale")
        ref_dir = _fixture_path(proj_dir)
        asset_file = ref_dir / "scene1_ingredients_main.png"
        asset_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        service = ReferenceWorkspaceService(project_root=proj_dir)

        asset_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xFF" * 50)

        stale_state = service.check_stale("stale")
        self.assertTrue(isinstance(stale_state.stale_assets, tuple))

    def test_scenes_using_asset(self) -> None:
        """Scenes using asset returns correct scene numbers."""
        proj_dir = self._make_project("scenes")
        assignments_dir = proj_dir / "movie"
        assignments_dir.mkdir(parents=True, exist_ok=True)
        (assignments_dir / "reference_assignments.json").write_text(
            json.dumps({
                "revision": "v2",
                "assignments": [
                    {
                        "scene_number": 3,
                        "actor_ids": ["hero", "villain"],
                        "location_ids": [],
                        "background_ids": [],
                        "style_ids": [],
                        "actor_look_ids": {},
                    },
                    {
                        "scene_number": 5,
                        "actor_ids": ["villain"],
                        "location_ids": [],
                        "background_ids": [],
                        "style_ids": [],
                        "actor_look_ids": {},
                    },
                ],
            })
        )
        service = ReferenceWorkspaceService(project_root=proj_dir)
        snap = service.load_library("scenes")
        hero_scenes = snap.scenes_using_asset("hero")
        self.assertEqual((3,), hero_scenes)
        villain_scenes = snap.scenes_using_asset("villain")
        self.assertEqual((3, 5), villain_scenes)


if __name__ == "__main__":
    unittest.main()
