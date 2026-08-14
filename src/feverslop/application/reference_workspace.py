from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.domain.reference_workspace import (
    ReferenceAsset,
    ReferenceKind,
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


# ---- Result objects ----


@dataclass(frozen=True)
class SaveAssignmentsResult:
    new_revision: str
    affected_scenes: tuple[int, ...]
    invalidated_artifacts: dict[str, Any]
    issues: tuple[str, ...] = ()


# ---- Use cases ----


class LoadReferenceWorkspaceUseCase:
    """Load and filter reference workspace assets."""

    def __init__(self, library: ReferenceLibraryPort):
        self._library = library

    def load(self, project_id: str) -> ReferenceWorkspaceSnapshot:
        return self._library.load(project_id)

    def filter(
        self,
        project_id: str,
        kinds: list[ReferenceKind] | None = None,
        stale_only: bool = False,
        missing_only: bool = False,
    ) -> tuple[ReferenceAsset, ...]:
        snap = self.load(project_id)
        return snap.filter_assets(
            kinds=kinds or list(ReferenceKind),
            stale_only=stale_only,
            missing_only=missing_only,
        )


class PreviewSceneAssignmentUseCase:
    """Validate a scene assignment without persisting."""

    def __init__(
        self,
        *,
        bible: MovieBiblePort,
        scene_cast: SceneCastPort,
    ):
        self._bible = bible
        self._scene_cast = scene_cast

    def preview(
        self,
        assignment: SceneReferenceAssignment,
        project_id: str = "",
        known_actor_ids: list[str] | None = None,
        known_location_ids: list[str] | None = None,
        known_background_ids: list[str] | None = None,
        max_scene_actors: int | None = None,
    ) -> list[str]:
        if known_actor_ids is None:
            known_actor_ids = self._bible.get_known_actor_ids(project_id)
        if known_location_ids is None:
            known_location_ids = self._bible.get_known_location_ids(project_id)
        if known_background_ids is None:
            known_background_ids = self._bible.get_background_ids(project_id)
        if max_scene_actors is None:
            max_scene_actors = self._scene_cast.get_max_scene_actors(project_id)

        issues = assignment.validate_against(
            known_actor_ids=known_actor_ids,
            known_location_ids=known_location_ids,
            known_background_ids=known_background_ids,
            max_scene_actors=max_scene_actors,
        )
        return issues


class SaveSceneAssignmentsUseCase:
    """Validate and persist scene assignments with revision control."""

    def __init__(
        self,
        *,
        library: ReferenceLibraryPort,
        bible: MovieBiblePort,
        scene_cast: SceneCastPort,
        invalidation: ArtifactInvalidationPort,
    ):
        self._library = library
        self._bible = bible
        self._scene_cast = scene_cast
        self._invalidation = invalidation
        self._preview = PreviewSceneAssignmentUseCase(bible=bible, scene_cast=scene_cast)

    def save(
        self,
        project_id: str,
        assignments: tuple[SceneReferenceAssignment, ...],
        expected_revision: str,
        old_snapshot: ReferenceWorkspaceSnapshot | None = None,
    ) -> SaveAssignmentsResult:
        all_issues: list[str] = []
        for a in assignments:
            issues = self._preview.preview(
                a,
                project_id=project_id,
            )
            all_issues.extend(issues)

        if all_issues:
            return SaveAssignmentsResult(
                new_revision=expected_revision,
                affected_scenes=tuple(),
                invalidated_artifacts={},
                issues=tuple(all_issues),
            )

        if old_snapshot is not None:
            old_snap = old_snapshot
        else:
            old_snap = self._library.load(project_id)
        old_assignments = old_snap.assignments

        new_revision: str
        try:
            new_revision = self._library.save_assignments(project_id, assignments, expected_revision)
        except ValueError:
            raise  # Revision mismatch bubbles up

        changed_scene_numbers = self._find_changed_scenes(assignments, old_assignments)

        changed_actors = self._collect_changed_ids(assignments, old_assignments, field="actor_ids")
        changed_locations = self._collect_changed_ids(assignments, old_assignments, field="location_ids")

        invalidated = self._invalidation.get_invalidated_artifacts(
            project_id=project_id,
            changed_scenes=changed_scene_numbers,
            changed_actor_ids=changed_actors or None,
            changed_location_ids=changed_locations or None,
        )

        return SaveAssignmentsResult(
            new_revision=new_revision,
            affected_scenes=tuple(changed_scene_numbers),
            invalidated_artifacts=invalidated,
            issues=(),
        )

    @staticmethod
    def _collect_changed_ids(
        new_assignments: tuple[SceneReferenceAssignment, ...],
        old_assignments: tuple[SceneReferenceAssignment, ...],
        field: str,
    ) -> list[str]:
        new_ids: set[str] = set()
        for a in new_assignments:
            new_ids.update(getattr(a, field))
        old_ids: set[str] = set()
        for a in old_assignments:
            old_ids.update(getattr(a, field))
        return list(new_ids ^ old_ids)

    @staticmethod
    def _find_changed_scenes(
        new_assignments: tuple[SceneReferenceAssignment, ...],
        old_assignments: tuple[SceneReferenceAssignment, ...],
    ) -> list[int]:
        def _scene_key(a: SceneReferenceAssignment) -> tuple:
            return (
                a.actor_ids,
                a.location_ids,
                a.background_ids,
                a.style_ids,
                tuple(sorted((a.actor_look_ids or {}).items())),
                a.prop_ids,
                tuple(item.to_dict() for item in a.prop_interactions),
            )

        old_map: dict[int, tuple] = {}
        for a in old_assignments:
            old_map[a.scene_number] = _scene_key(a)

        new_map: dict[int, tuple] = {}
        for a in new_assignments:
            new_map[a.scene_number] = _scene_key(a)

        changed: list[int] = []
        all_scenes = set(old_map.keys()) | set(new_map.keys())
        for sn in all_scenes:
            old_key = old_map.get(sn)
            new_key = new_map.get(sn)
            if old_key is None or new_key is None or new_key != old_key:
                changed.append(sn)
        return sorted(changed)


class ImportReferenceUseCase:
    """Import an external image into the project."""

    def __init__(self, *, importer: ImportReferencePort):
        self._importer = importer

    def import_asset(
        self,
        project_id: str,
        source_path: Path,
        asset: ReferenceAsset,
    ) -> ReferenceAsset:
        return self._importer.import_asset(project_id, source_path, asset)


class GenerationJobUseCase:
    """Queue generation jobs for reference assets."""

    def __init__(self, *, jobs: GenerationJobPort):
        self._jobs = jobs

    def queue_storyboard_frame(
        self,
        project_id: str,
        scene_number: int,
        reference_ids: tuple[str, ...],
    ) -> str:
        return self._jobs.queue_storyboard_frame(project_id, scene_number, reference_ids)

    def queue_msr_sheet(
        self,
        project_id: str,
        scene_number: int,
        actor_ids: tuple[str, ...],
        location_ids: tuple[str, ...] = (),
    ) -> str:
        return self._jobs.queue_msr_sheet(project_id, scene_number, actor_ids, location_ids)

    def queue_ingredients_sheet(
        self,
        project_id: str,
        scene_number: int,
        actor_ids: tuple[str, ...],
        location_ids: tuple[str, ...] = (),
        background_ids: tuple[str, ...] = (),
    ) -> str:
        return self._jobs.queue_ingredients_sheet(project_id, scene_number, actor_ids, location_ids, background_ids)

    def queue_reference_rerender(
        self,
        project_id: str,
        reference_id: str,
    ) -> str:
        return self._jobs.queue_reference_rerender(project_id, reference_id)
