from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from feverslop.domain.reference_workspace import (
    PropInteraction,
    ReferenceAsset,
    ReferenceKind,
    ReferenceProvenance,
    ReferenceWorkspaceSnapshot,
    SceneReferenceAssignment,
)
from feverslop.ports.reference_library import (
    ArtifactInvalidationPort,
    GenerationJobPort,
    ImportReferencePort,
    MovieBiblePort,
    ReferenceLibraryPort,
    SceneCastPort,
)

# ---- Fake ports ----


class FakeLibrary(ReferenceLibraryPort):
    def __init__(self):
        self._snapshots: dict[str, ReferenceWorkspaceSnapshot] = {}
        self._asset_counter = 0
        self.load_count = 0

    def load(self, project_id: str) -> ReferenceWorkspaceSnapshot:
        self.load_count += 1
        return self._snapshots.get(project_id, ReferenceWorkspaceSnapshot(assets=(), assignments=()))

    def save_assignments(
        self,
        project_id: str,
        assignments: tuple[SceneReferenceAssignment, ...],
        expected_revision: str,
    ) -> str:
        snap = self._snapshots.get(project_id)
        if snap and snap.revision != expected_revision:
            raise ValueError("Revision mismatch")
        new_revision = "r2" if snap else "r1"
        if snap:
            self._snapshots[project_id] = ReferenceWorkspaceSnapshot(
                assets=snap.assets,
                assignments=assignments,
                revision=new_revision,
                project_id=project_id,
            )
        else:
            self._snapshots[project_id] = ReferenceWorkspaceSnapshot(
                assets=(),
                assignments=assignments,
                revision=new_revision,
                project_id=project_id,
            )
        return new_revision

    def add_asset(self, project_id: str, asset: ReferenceAsset) -> ReferenceAsset:
        snap = self._snapshots.get(project_id)
        if snap is None:
            snap = ReferenceWorkspaceSnapshot(assets=(), assignments=(), project_id=project_id)
        self._snapshots[project_id] = ReferenceWorkspaceSnapshot(
            assets=snap.assets + (asset,),
            assignments=snap.assignments,
            revision=snap.revision,
            project_id=project_id,
        )
        return asset


class FakeBible(MovieBiblePort):
    def __init__(self, actors=None, locations=None, backgrounds=None, props=None):
        self._actors = actors or []
        self._locations = locations or []
        self._backgrounds = backgrounds or []
        self._props = list(props or [])

    def get_known_actor_ids(self, project_id: str) -> list[str]:
        return list(self._actors)

    def get_known_location_ids(self, project_id: str) -> list[str]:
        return list(self._locations)

    def get_background_ids(self, project_id: str) -> list[str]:
        return list(self._backgrounds)

    def get_known_prop_ids(self, project_id: str) -> list[str]:
        return list(self._props)


class FakeSceneCast(SceneCastPort):
    def __init__(self, max_actors=4):
        self._max = max_actors

    def get_max_scene_actors(self, project_id: str) -> int:
        return self._max


class FakeInvalidation(ArtifactInvalidationPort):
    def get_invalidated_artifacts(
        self,
        project_id: str,
        changed_scenes: list[int],
        changed_actor_ids: list[str] | None = None,
        changed_location_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for sn in changed_scenes:
            result.setdefault("msr_sheets", []).append(f"scene_{sn}_msr.png")
            result.setdefault("ingredients_sheets", []).append(f"scene_{sn}_ingredients.png")
        return result


class FakeImporter(ImportReferencePort):
    def import_asset(self, project_id: str, source_path: Path, asset: ReferenceAsset) -> ReferenceAsset:
        return ReferenceAsset(
            id=asset.id,
            kind=asset.kind,
            label=asset.label,
            path=f"movie/references/{source_path.name}",
            width=640,
            height=480,
            provenance=ReferenceProvenance(source="import"),
        )


class FakeJobs(GenerationJobPort):
    def __init__(self):
        self.jobs: list[dict[str, Any]] = []

    def queue_storyboard_frame(self, project_id: str, scene_number: int, reference_ids: tuple[str, ...]) -> str:
        job = {"action": "storyboard_frame", "scene": scene_number, "refs": reference_ids}
        self.jobs.append(job)
        return f"job_{len(self.jobs)}"

    def queue_msr_sheet(self, project_id: str, scene_number: int, actor_ids: tuple[str, ...], location_ids: tuple[str, ...] = ()) -> str:
        job = {"action": "msr_sheet", "scene": scene_number, "actors": actor_ids, "locations": location_ids}
        self.jobs.append(job)
        return f"job_{len(self.jobs)}"

    def queue_ingredients_sheet(self, project_id: str, scene_number: int, actor_ids: tuple[str, ...], location_ids: tuple[str, ...] = (), background_ids: tuple[str, ...] = ()) -> str:
        job = {"action": "ingredients_sheet", "scene": scene_number, "actors": actor_ids, "locations": location_ids, "backgrounds": background_ids}
        self.jobs.append(job)
        return f"job_{len(self.jobs)}"

    def queue_reference_rerender(self, project_id: str, reference_id: str) -> str:
        job = {"action": "rerender", "ref": reference_id}
        self.jobs.append(job)
        return f"job_{len(self.jobs)}"


# ---- Tests ----

class LoadReferenceWorkspaceUseCaseTests(unittest.TestCase):
    def _use_case(self, library, **kwargs):
        from feverslop.application.reference_workspace import (
            LoadReferenceWorkspaceUseCase,
        )
        return LoadReferenceWorkspaceUseCase(library, **kwargs)

    def test_load_empty_project(self):
        lib = FakeLibrary()
        uc = self._use_case(lib)
        result = uc.load("proj")
        self.assertEqual((), result.assets)
        self.assertEqual((), result.assignments)

    def test_load_assets(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(
                ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR),
                ReferenceAsset(id="lab", kind=ReferenceKind.LOCATION),
            ),
            assignments=(),
            revision="r1",
            project_id="proj",
        )
        uc = self._use_case(lib)
        result = uc.load("proj")
        self.assertEqual(2, len(result.assets))

    def test_filter_by_kind(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(
                ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR),
                ReferenceAsset(id="lab", kind=ReferenceKind.LOCATION),
                ReferenceAsset(id="bg1", kind=ReferenceKind.BACKGROUND),
            ),
            assignments=(),
            revision="r1",
            project_id="proj",
        )
        uc = self._use_case(lib)
        result = uc.filter("proj", kinds=[ReferenceKind.LOCATION])
        self.assertEqual(("lab",), tuple(a.id for a in result))

    def test_filter_stale_only(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(
                ReferenceAsset(id="a1", kind=ReferenceKind.ACTOR, stale=False),
                ReferenceAsset(id="a2", kind=ReferenceKind.ACTOR, stale=True),
            ),
            assignments=(),
            revision="r1",
            project_id="proj",
        )
        uc = self._use_case(lib)
        result = uc.filter("proj", stale_only=True)
        self.assertEqual(("a2",), tuple(a.id for a in result))


class PreviewSceneAssignmentUseCaseTests(unittest.TestCase):
    def _use_case(self, **kwargs):
        from feverslop.application.reference_workspace import (
            PreviewSceneAssignmentUseCase,
        )
        return PreviewSceneAssignmentUseCase(**kwargs)

    def test_valid_assignment(self):
        uc = self._use_case(
            bible=FakeBible(actors=["hero", "villain"], locations=["lab"]),
            scene_cast=FakeSceneCast(4),
        )
        assignment = SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), location_ids=("lab",))
        issues = uc.preview(assignment)
        self.assertEqual([], issues)

    def test_unknown_actor_id(self):
        uc = self._use_case(
            bible=FakeBible(actors=["hero"]),
            scene_cast=FakeSceneCast(4),
        )
        assignment = SceneReferenceAssignment(scene_number=3, actor_ids=("unknown",))
        issues = uc.preview(assignment)
        self.assertTrue(any("unknown" in i for i in issues))

    def test_actor_limit_exceeded(self):
        uc = self._use_case(
            bible=FakeBible(actors=["a", "b", "c", "d"]),
            scene_cast=FakeSceneCast(2),
        )
        assignment = SceneReferenceAssignment(scene_number=1, actor_ids=("a", "b", "c"))
        issues = uc.preview(assignment)
        self.assertTrue(any("actor" in i.lower() for i in issues))

    def test_unknown_location_id(self):
        uc = self._use_case(
            bible=FakeBible(locations=["lab"]),
            scene_cast=FakeSceneCast(4),
        )
        assignment = SceneReferenceAssignment(scene_number=1, location_ids=("underground",))
        issues = uc.preview(assignment)
        self.assertTrue(any("unknown" in i.lower() for i in issues))

    def test_known_prop_assignment_valid(self):
        uc = self._use_case(
            bible=FakeBible(actors=["hero"], props=["guitar"]),
            scene_cast=FakeSceneCast(4),
        )
        assignment = SceneReferenceAssignment(
            scene_number=3,
            actor_ids=("hero",),
            prop_ids=("guitar",),
            prop_interactions=(PropInteraction(actor_id="hero", prop_id="guitar", action="holds"),),
        )
        issues = uc.preview(assignment)
        self.assertEqual([], issues)

    def test_unknown_prop_rejected(self):
        uc = self._use_case(
            bible=FakeBible(actors=["hero"], props=["guitar"]),
            scene_cast=FakeSceneCast(4),
        )
        assignment = SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), prop_ids=("mic",))
        issues = uc.preview(assignment)
        self.assertTrue(any("Unknown prop ID: mic" in i for i in issues))

    def test_prop_interaction_unknown_prop_rejected(self):
        uc = self._use_case(
            bible=FakeBible(actors=["hero"], props=["guitar"]),
            scene_cast=FakeSceneCast(4),
        )
        assignment = SceneReferenceAssignment(
            scene_number=3,
            actor_ids=("hero",),
            prop_interactions=(PropInteraction(actor_id="hero", prop_id="mic", action="holds"),),
        )
        issues = uc.preview(assignment)
        self.assertTrue(any("Unknown interaction prop ID: mic" in i for i in issues))

    def test_prop_rejected_when_no_props_configured(self):
        uc = self._use_case(
            bible=FakeBible(actors=["hero"]),
            scene_cast=FakeSceneCast(4),
        )
        assignment = SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), prop_ids=("guitar",))
        issues = uc.preview(assignment)
        self.assertTrue(any("Unknown prop ID: guitar" in i for i in issues))


class SaveSceneAssignmentsUseCaseTests(unittest.TestCase):
    def _use_case(self, **kwargs):
        from feverslop.application.reference_workspace import (
            SaveSceneAssignmentsUseCase,
        )
        return SaveSceneAssignmentsUseCase(**kwargs)

    def test_save_updates_assignments(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(), assignments=(), revision="r1", project_id="proj",
        )
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"], locations=["lab"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),)
        result = uc.save("proj", assignments, "r1")
        self.assertEqual("r2", result.new_revision)
        self.assertIn(3, result.affected_scenes)

    def test_revision_mismatch_raises(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(), assignments=(), revision="r1", project_id="proj",
        )
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),)
        with self.assertRaisesRegex(ValueError, "Revision mismatch"):
            uc.save("proj", assignments, "wrong")

    def test_unknown_actor_rejected(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(), assignments=(), revision="r1", project_id="proj",
        )
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("unknown",)),)
        result = uc.save("proj", assignments, "r1")
        self.assertTrue(any("unknown" in i.lower() for i in result.issues))
        self.assertEqual("r1", result.new_revision)

    def test_save_prop_assignment_persists(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(), assignments=(), revision="r1", project_id="proj",
        )
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"], locations=["lab"], props=["guitar"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        assignments = (
            SceneReferenceAssignment(
                scene_number=3,
                actor_ids=("hero",),
                prop_ids=("guitar",),
                prop_interactions=(PropInteraction(actor_id="hero", prop_id="guitar", action="holds"),),
            ),
        )
        result = uc.save("proj", assignments, "r1")
        self.assertEqual((), result.issues)
        self.assertEqual("r2", result.new_revision)
        self.assertIn(3, result.affected_scenes)
        saved = lib.load("proj").assignments
        self.assertEqual(1, len(saved))
        self.assertEqual(("guitar",), saved[0].prop_ids)
        self.assertEqual(
            (PropInteraction(actor_id="hero", prop_id="guitar", action="holds"),),
            saved[0].prop_interactions,
        )

    def test_save_unknown_prop_not_persisted(self):
        lib = FakeLibrary()
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(), assignments=(), revision="r1", project_id="proj",
        )
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"], locations=["lab"], props=["guitar"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        assignments = (
            SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), prop_ids=("mic",)),
        )
        result = uc.save("proj", assignments, "r1")
        self.assertTrue(result.issues)
        self.assertEqual("r1", result.new_revision)
        self.assertEqual((), lib.load("proj").assignments)

    def test_returns_invalidated_artifacts(self):
        lib = FakeLibrary()
        old_assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),)
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(), assignments=old_assignments, revision="r1", project_id="proj",
        )
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"], locations=["lab"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        new_assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), location_ids=("lab",)),)
        result = uc.save("proj", new_assignments, "r1")
        self.assertIn("msr_sheets", result.invalidated_artifacts)

    def test_save_with_old_snapshot_skips_load(self):
        lib = FakeLibrary()
        old_assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),)
        old_snap = ReferenceWorkspaceSnapshot(
            assets=(), assignments=old_assignments, revision="r1", project_id="proj",
        )
        lib._snapshots["proj"] = old_snap
        lib.load_count = 0
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"], locations=["lab"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        new_assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), location_ids=("lab",)),)
        result = uc.save("proj", new_assignments, "r1", old_snapshot=old_snap)
        self.assertEqual("r2", result.new_revision)
        self.assertIn(3, result.affected_scenes)
        self.assertEqual(0, lib.load_count)

    def test_save_without_old_snapshot_loads_library(self):
        lib = FakeLibrary()
        old_assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),)
        lib._snapshots["proj"] = ReferenceWorkspaceSnapshot(
            assets=(), assignments=old_assignments, revision="r1", project_id="proj",
        )
        uc = self._use_case(
            library=lib,
            bible=FakeBible(actors=["hero"], locations=["lab"]),
            scene_cast=FakeSceneCast(4),
            invalidation=FakeInvalidation(),
        )
        lib.load_count = 0
        new_assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), location_ids=("lab",)),)
        result = uc.save("proj", new_assignments, "r1")
        self.assertEqual("r2", result.new_revision)
        self.assertIn(3, result.affected_scenes)
        self.assertEqual(1, lib.load_count)


class ImportReferenceUseCaseTests(unittest.TestCase):
    def _use_case(self, **kwargs):
        from feverslop.application.reference_workspace import ImportReferenceUseCase
        return ImportReferenceUseCase(**kwargs)

    def test_import_forwards_to_adapter(self):
        importer = FakeImporter()
        uc = self._use_case(importer=importer)
        asset = uc.import_asset(
            "proj",
            Path("/tmp/hero.png"),
            ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR, label="Hero"),
        )
        self.assertIn("hero.png", asset.path)
        self.assertEqual(640, asset.width)


class GenerationJobUseCaseTests(unittest.TestCase):
    def _use_case(self, **kwargs):
        from feverslop.application.reference_workspace import GenerationJobUseCase
        return GenerationJobUseCase(**kwargs)

    def test_queue_storyboard(self):
        jobs = FakeJobs()
        uc = self._use_case(jobs=jobs)
        job_id = uc.queue_storyboard_frame("proj", 5, ("frame_1",))
        self.assertEqual("job_1", job_id)
        self.assertEqual(1, len(jobs.jobs))

    def test_queue_msr_sheet(self):
        jobs = FakeJobs()
        uc = self._use_case(jobs=jobs)
        job_id = uc.queue_msr_sheet("proj", 3, ("hero",), ("lab",))
        self.assertEqual("job_1", job_id)

    def test_queue_ingredients_sheet(self):
        jobs = FakeJobs()
        uc = self._use_case(jobs=jobs)
        job_id = uc.queue_ingredients_sheet("proj", 3, ("hero",), ("lab",), ("bg1",))
        self.assertEqual("job_1", job_id)

    def test_queue_rerender(self):
        jobs = FakeJobs()
        uc = self._use_case(jobs=jobs)
        job_id = uc.queue_reference_rerender("proj", "hero")
        self.assertEqual("job_1", job_id)


if __name__ == "__main__":
    unittest.main()
