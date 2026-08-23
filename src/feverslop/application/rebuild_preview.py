from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from feverslop.domain.rebuild_policy import (
    ArtifactKind,
    ChangeSet,
    Freshness,
    RebuildPlan,
    preview_rebuild,
)
from feverslop.ports.rebuild_execution import RebuildExecutionPort


class RebuildStage(Enum):
    PLANNING = "planning"
    REFERENCES = "references"
    RENDER = "render"
    FINAL = "final"


@dataclass(frozen=True)
class ArtifactState:
    kind: ArtifactKind
    state: Freshness
    scene_numbers: frozenset[int] = frozenset()


@dataclass(frozen=True)
class RebuildPreviewResult:
    stages: tuple[RebuildStage, ...]
    affected_scenes: frozenset[int]
    reusable_artifacts: tuple[ArtifactState, ...]
    stale_artifacts: tuple[ArtifactState, ...]
    unknown_artifacts: tuple[ArtifactState, ...]
    plan: RebuildPlan


class PreviewRebuildUseCase:
    def __init__(self) -> None:
        pass

    def execute(
        self,
        *,
        change: ChangeSet,
        current_fingerprints: dict[ArtifactKind, Freshness] | None = None,
    ) -> RebuildPreviewResult:
        plan = preview_rebuild(change, current_fingerprints=current_fingerprints)

        stages_map = {
            "planning": RebuildStage.PLANNING,
            "references": RebuildStage.REFERENCES,
            "render": RebuildStage.RENDER,
            "final": RebuildStage.FINAL,
        }
        stages = tuple(stages_map[stage.name] for stage in plan.stages)

        reusable = tuple(
            ArtifactState(kind=kind, state=Freshness.CURRENT)
            for kind in sorted(plan.reuse, key=lambda k: k.value)
        )

        stale = tuple(
            ArtifactState(
                kind=kind,
                state=Freshness.STALE,
                scene_numbers=plan.affected_scenes,
            )
            for kind in sorted(plan.rebuild, key=lambda k: k.value)
        )

        unknown = tuple(
            ArtifactState(kind=kind, state=Freshness.UNKNOWN)
            for kind in sorted(plan.unknown, key=lambda k: k.value)
        )

        return RebuildPreviewResult(
            stages=stages,
            affected_scenes=plan.affected_scenes,
            reusable_artifacts=reusable,
            stale_artifacts=stale,
            unknown_artifacts=unknown,
            plan=plan,
        )


class RequestRebuildUseCase:
    def __init__(self, executor: RebuildExecutionPort) -> None:
        if not callable(getattr(executor, "request_rebuild", None)):
            raise TypeError("executor must implement RebuildExecutionPort.request_rebuild")
        self._executor = executor

    def execute(
        self,
        *,
        project_id: str,
        change: ChangeSet,
        current_fingerprints: dict[ArtifactKind, Freshness] | None = None,
    ) -> str:
        plan = preview_rebuild(change, current_fingerprints=current_fingerprints)
        if not plan.rebuild:
            return ""
        return self._executor.request_rebuild(project_id, plan)
