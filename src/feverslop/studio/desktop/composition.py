from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feverslop.ports.reporting import NullReporter
from feverslop.studio.job_service import StudioJobService
from feverslop.studio.jobs import JobRegistry
from feverslop.studio.projects import ProjectStore


@dataclass(frozen=True)
class StudioContext:
    store: ProjectStore
    jobs: JobRegistry
    job_service: StudioJobService


def create_studio_context(projects_root: str | Path) -> StudioContext:
    store = ProjectStore(projects_root, reporter=NullReporter())
    jobs = JobRegistry()
    return StudioContext(
        store=store,
        jobs=jobs,
        job_service=StudioJobService(store=store, jobs=jobs),
    )

