from __future__ import annotations

from typing import Protocol

from feverslop.domain.rebuild_policy import (
    ArtifactFingerprint,
    ArtifactKind,
    RebuildPlan,
)


class RebuildExecutionPort(Protocol):
    """Port for executing a rebuild plan by dispatching job actions."""

    def request_rebuild(self, project_id: str, plan: RebuildPlan) -> str: ...


class ArtifactProvenancePort(Protocol):
    """Port for recording and reading artifact provenance fingerprints."""

    def record_fingerprint(self, project_id: str, fingerprint: ArtifactFingerprint) -> None: ...

    def load_fingerprint(self, project_id: str, kind: ArtifactKind, scene_number: int | None = None) -> ArtifactFingerprint | None: ...

    def load_fingerprints(self, project_id: str) -> list[ArtifactFingerprint]: ...
