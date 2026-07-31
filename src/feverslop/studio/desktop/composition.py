from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feverslop.adapters.project_scene_documents import ProjectSceneDocuments
from feverslop.application.scene_workspace import LoadSceneWorkspaceUseCase, PatchSceneUseCase
from feverslop.infra.sqlite_adapter import SqliteArtifactProvenance, SqliteRevisionStore
from feverslop.ports.reporting import NullReporter
from feverslop.studio.job_service import StudioJobService
from feverslop.studio.jobs import JobRegistry
from feverslop.studio.projects import ProjectStore
from feverslop.studio.reference_workspace_service import ReferenceWorkspaceService
from feverslop.studio.rebuild_service import RebuildService
from feverslop.studio.scene_workspace_service import SceneWorkspaceService
from feverslop.studio.timeline_service import TimelineStudioService


@dataclass(frozen=True)
class StudioContext:
    store: ProjectStore
    jobs: JobRegistry
    job_service: StudioJobService
    scene_service: SceneWorkspaceService
    timeline_service: TimelineStudioService
    rebuild_service: RebuildService
    provenance: SqliteArtifactProvenance
    reference_factory: type[ReferenceWorkspaceService] = ReferenceWorkspaceService


def create_studio_context(projects_root: str | Path) -> StudioContext:
    store = ProjectStore(projects_root, reporter=NullReporter())
    jobs = JobRegistry()
    revisions_db = str(store.projects_root / "render" / "prompt_revisions.db")
    provenance = SqliteArtifactProvenance(revisions_db)
    job_service = StudioJobService(store=store, jobs=jobs, provenance=provenance)
    scene_documents = ProjectSceneDocuments(store.project_root)
    return StudioContext(
        store=store,
        jobs=jobs,
        job_service=job_service,
        scene_service=SceneWorkspaceService(
            load_workspace=LoadSceneWorkspaceUseCase(
                documents=scene_documents,
                media=scene_documents,
            ),
            patch_scene=PatchSceneUseCase(documents=scene_documents),
            jobs=job_service,
            project_type=lambda project_id: str(
                store.project_metadata(project_id).get(
                    "project_type",
                    "standard_music_video",
                )
            ),
        ),
        timeline_service=TimelineStudioService(job_registry=jobs),
        rebuild_service=RebuildService(SqliteRevisionStore(revisions_db)),
        provenance=provenance,
    )
