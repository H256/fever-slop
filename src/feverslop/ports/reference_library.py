from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from feverslop.domain.reference_workspace import (
    ReferenceAsset,
    SceneReferenceAssignment,
    ReferenceWorkspaceSnapshot,
)


class ReferenceLibraryPort(Protocol):
    """Persistent store for reference workspace assets and assignments."""

    def load(self, project_id: str) -> ReferenceWorkspaceSnapshot:
        """Load the reference workspace snapshot for the given project."""  # noqa: D402
        ...

    def save_assignments(
        self,
        project_id: str,
        assignments: tuple[SceneReferenceAssignment, ...],
        expected_revision: str,
    ) -> str:
        """Save assignments atomically with revision check.

        Returns the new revision string on success.
        Raises ``ValueError`` on revision mismatch.
        """  # noqa: D402
        ...

    def add_asset(
        self,
        project_id: str,
        asset: ReferenceAsset,
    ) -> ReferenceAsset:
        """Persist a new reference asset (typically after import).

        Returns the stored asset with adapter-filled fields (path, dimensions).
        """  # noqa: D402
        ...


class ImportReferencePort(Protocol):
    """Copy external images into a safe project-relative location."""

    def import_asset(
        self,
        project_id: str,
        source_path: Path,
        asset: ReferenceAsset,
    ) -> ReferenceAsset:
        """Copy *source_path* into the project structure.

        The adapter must:
        - Reject directory traversal in *source_path*
        - Preserve the original file unchanged
        - Return a new asset with the resolved path and dimensions
        """  # noqa: D402
        ...


class MovieBiblePort(Protocol):
    """Read-only access to the project's movie bible data."""

    def get_known_actor_ids(self, project_id: str) -> list[str]:
        """Return the authoritative list of actor IDs from the movie bible."""  # noqa: D402
        ...

    def get_known_location_ids(self, project_id: str) -> list[str]:
        """Return the authoritative list of location IDs from the movie bible."""  # noqa: D402
        ...

    def get_background_ids(self, project_id: str) -> list[str]:
        """Return known background IDs from render-plan metadata."""  # noqa: D402
        ...

    def get_known_prop_ids(self, project_id: str) -> list[str]:
        """Return known prop IDs from project global asset config."""  # noqa: D402
        ...


class SceneCastPort(Protocol):
    """Access to scene cast resolution for validation."""

    def get_max_scene_actors(self, project_id: str) -> int:
        """Return the maximum number of actors allowed in a single scene."""  # noqa: D402
        ...


class ArtifactInvalidationPort(Protocol):
    """Determine which downstream artifacts become stale after changes."""

    def get_invalidated_artifacts(
        self,
        project_id: str,
        changed_scenes: list[int],
        changed_actor_ids: list[str] | None = None,
        changed_location_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return artifact classes and paths that need regeneration.

        Returns a mapping like:
        {
            "msr_sheets": ["scene_3_msr_sheet.png"],
            "ingredients_sheets": ["scene_3_ingredients.png"],
            "renders": ["scene_3/shot_1.png"],
        }
        """  # noqa: D402
        ...


class GenerationJobPort(Protocol):
    """Queue generation jobs for reference assets."""

    def queue_storyboard_frame(
        self,
        project_id: str,
        scene_number: int,
        reference_ids: tuple[str, ...],
    ) -> str:
        """Queue a storyboard frame generation job. Returns job ID."""  # noqa: D402
        ...

    def queue_msr_sheet(
        self,
        project_id: str,
        scene_number: int,
        actor_ids: tuple[str, ...],
        location_ids: tuple[str, ...] = (),
    ) -> str:
        """Queue an MSR sheet generation job. Returns job ID."""  # noqa: D402
        ...

    def queue_ingredients_sheet(
        self,
        project_id: str,
        scene_number: int,
        actor_ids: tuple[str, ...],
        location_ids: tuple[str, ...] = (),
        background_ids: tuple[str, ...] = (),
    ) -> str:
        """Queue an Ingredients sheet generation job. Returns job ID."""  # noqa: D402
        ...

    def queue_reference_rerender(
        self,
        project_id: str,
        reference_id: str,
    ) -> str:
        """Queue a rerender for a single reference asset. Returns job ID."""  # noqa: D402
        ...
