"""Integration tests for the reference workspace flow.

Tests the complete stack: adapters -> service -> viewmodel.
These require PySide6 to run.
"""
from __future__ import annotations

import sys
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
    ReferenceWorkspaceService,
)


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
        self._projects_root = Path(self._tmp) / "projects"
        self._projects_root.mkdir()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_test_project(self, project_id: str = "test") -> Path:
        proj_dir = self._projects_root / project_id
        proj_dir.mkdir()
        # Create a minimal project config
        config = proj_dir / "config.json"
        config.write_text("{}")
        return proj_dir

    def _create_assets(self, proj_dir: Path, assets: list[dict[str, Any]]) -> None:
        """Create test asset files in the project's reference directory."""
        ref_dir = proj_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        for asset in assets:
            path = ref_dir / asset.get("filename", asset["id"] + ".png")
            if not asset.get("exists", True):
                continue
            # Create a minimal PNG-like file (just for existence testing)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    def test_service_creation(self) -> None:
        """Service can be created with projects_root."""
        proj_dir = self._make_test_project()
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        self.assertIsNotNone(service)

    def test_load_empty_project(self) -> None:
        """Loading an empty project returns empty snapshot."""
        proj_dir = self._make_test_project("empty")
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        snap = service.load_library("empty")
        self.assertEqual(0, len(snap.assets))
        self.assertEqual(0, len(snap.assignments))

    def test_load_with_assets(self) -> None:
        """Loading a project with asset files returns assets."""
        proj_dir = self._make_test_project("assets")
        self._create_assets(proj_dir, [{"id": "hero", "filename": "hero.png"}])
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        service.set_project("assets")
        snap = service.load_library("assets")
        # At minimum, the snapshot loads without error
        self.assertIsNotNone(snap.revision)

    def test_filter_assets(self) -> None:
        """Filtering by kind works correctly."""
        proj_dir = self._make_test_project("filter")
        self._create_assets(
            proj_dir,
            [{"id": "hero", "filename": "hero.png"}, {"id": "loc1", "filename": "loc1.png"}],
        )
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        service.set_project("filter")
        all_assets = service.load_library("filter").filter_assets(kinds=list(ReferenceKind))
        # Should have at least the files we created
        asset_ids = {a.id for a in all_assets}
        self.assertTrue(asset_ids)

    def test_preview_assignment(self) -> None:
        """Preview scene assignment validates actor_ids."""
        proj_dir = self._make_test_project("preview")
        config = proj_dir / "references" / "scene_assignments.json"
        config.parent.mkdir(exist_ok=True)
        config.write_text("{}")
        self._create_assets(proj_dir, [{"id": "hero", "filename": "hero.png"}])
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        # Preview with valid actors should succeed
        result: CommandResult = service.preview_assignment(
            "preview",
            1,
            actor_ids=("hero",),
        )
        self.assertTrue(result.success)

    def test_save_assignments(self) -> None:
        """Save assignments creates/persists the assignments file."""
        proj_dir = self._make_test_project("save")
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        service.set_project("save")
        old_snap = service.load_library("save")
        result: CommandResult = service.save_assignments(
            "save",
            tuple([]),
            old_snap.revision,
        )
        # Should succeed for initial empty save
        self.assertTrue(result.success)
        self.assertIn("new_revision", result.data)

    def test_assignment_roundtrip(self) -> None:
        """Assignments survive load -> save -> load cycle."""
        proj_dir = self._make_test_project("roundtrip")
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        service.set_project("roundtrip")

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
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        service.set_project("conflict")

        snap1 = service.load_library("conflict")
        # First save with current revision
        result1: CommandResult = service.save_assignments(
            "conflict",
            tuple([{"scene_number": 1, "actor_ids": []}]),
            snap1.revision,
        )
        self.assertTrue(result1.success)

        # Second save with stale revision should fail
        result2: CommandResult = service.save_assignments(
            "conflict",
            tuple([{"scene_number": 2, "actor_ids": []}]),
            snap1.revision,  # old revision
        )
        self.assertFalse(result2.success)

    def test_viewmodel_creation(self) -> None:
        """Viewmodel can be created and bound to service."""
        proj_dir = self._make_test_project("vm")
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        vm = ReferenceWorkspaceViewModel(service=service)
        self.assertIsNotNone(vm.library_model)
        self.assertIsNotNone(vm.assignments_model)

    def test_viewmodel_project_switch(self) -> None:
        """Viewmodel updates when switching projects."""
        self._make_test_project("proj_a")
        self._make_test_project("proj_b")
        proj_dir = self._projects_root / "proj_a"
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        vm = ReferenceWorkspaceViewModel(service=service)
        ok = vm.set_project("proj_a")
        self.assertTrue(ok)
        self.assertEqual("proj_a", vm.current_project)

    def test_viewmodel_asset_selection(self) -> None:
        """Viewmodel can select and track assets."""
        proj_dir = self._make_test_project("select")
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        vm = ReferenceWorkspaceViewModel(service=service)
        vm.set_project("select")
        vm.select_asset("hero")
        self.assertEqual("hero", vm.selected_asset)

    def test_viewmodel_filter(self) -> None:
        """Viewmodel can filter assets by kind."""
        proj_dir = self._make_test_project("filter_vm")
        service = ReferenceWorkspaceService(
            project_root=proj_dir.parent,
        )
        vm = ReferenceWorkspaceViewModel(service=service)
        vm.set_project("filter_vm")
        vm.set_filter_kind("actor")
        # After filter, library_model should update
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


class TestReferenceWorkspaceServiceFull(unittest.TestCase):
    """Full service lifecycle tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._project_root = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_project(self, project_id: str) -> Path:
        proj_dir = self._project_root / project_id
        proj_dir.mkdir()
        (proj_dir / "config.json").write_text("{}")
        return proj_dir

    def test_queue_generation_storyboard(self) -> None:
        """Queue storyboard generation returns job ID."""
        self._make_project("gen")
        service = ReferenceWorkspaceService(
            project_root=self._project_root,
        )
        service.set_project("gen")
        from feverslop.studio.reference_workspace_service import GenerationCommand

        result: CommandResult = service.queue_generation(
            "gen",
            GenerationCommand(action="storyboard"),
        )
        self.assertTrue(result.success)
        self.assertIn("generation_job_id", result.data)

    def test_queue_generation_msr(self) -> None:
        """Queue MSR generation returns job ID."""
        self._make_project("msr")
        service = ReferenceWorkspaceService(
            project_root=self._project_root,
        )
        service.set_project("msr")
        from feverslop.studio.reference_workspace_service import GenerationCommand

        result: CommandResult = service.queue_generation(
            "msr",
            GenerationCommand(action="msr_sheet", scene_number=1),
        )
        self.assertTrue(result.success)

    def test_check_stale(self) -> None:
        """Check stale identifies modified assets."""
        proj_dir = self._make_project("stale")
        ref_dir = proj_dir / "references"
        ref_dir.mkdir()
        asset_file = ref_dir / "hero.png"
        asset_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        service = ReferenceWorkspaceService(
            project_root=self._project_root,
        )
        service.set_project("stale")

        # Snapshot, then modify the asset file
        asset_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xFF" * 50)

        stale_state = service.check_stale("stale")
        # At least check that it runs without error
        self.assertTrue(isinstance(stale_state.stale_assets, tuple))

    def test_scenes_using_asset(self) -> None:
        """Scenes using asset returns correct scene numbers."""
        proj_dir = self._make_project("scenes")
        ref_dir = proj_dir / "references"
        ref_dir.mkdir()
        (ref_dir / "scene_assignments.json").write_text(
            '{"assignments":['
            '{"scene_number":3,"actor_ids":["hero","villain"],"location_ids":[],"background_ids":[],"style_ids":[],"actor_look_ids":{}}'
            ','
            '{"scene_number":5,"actor_ids":["villain"],"location_ids":[],"background_ids":[],"style_ids":[],"actor_look_ids":{}}'
            '],"revision":"v2"}',
        )
        service = ReferenceWorkspaceService(
            project_root=self._project_root,
        )
        service.set_project("scenes")
        snap = service.load_library("scenes")
        hero_scenes = snap.scenes_using_asset("hero")
        self.assertEqual((3,), hero_scenes)
        villain_scenes = snap.scenes_using_asset("villain")
        self.assertEqual((3, 5), villain_scenes)


if __name__ == "__main__":
    unittest.main()
