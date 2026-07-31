from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class ArtifactKind(Enum):
    AUDIO_TIMELINE = "audio_timeline"
    AUDIO_ANALYSIS = "audio_analysis"
    BEAT_MARKERS = "beat_markers"
    SCENE_RENDER = "scene_render"
    SCENE_STORYBOARD = "scene_storyboard"
    REFERENCE_SHEETS = "reference_sheets"
    REFERENCE_SOURCES = "reference_sources"
    PREPARED_WORKFLOW = "prepared_workflow"
    FINAL_VIDEO = "final_video"
    REVIEW_ORDERING = "review_ordering"
    RENDER_PLAN = "render_plan"
    PROMPT_GENERATION = "prompt_generation"


class ChangeKind(Enum):
    PROMPT = "prompt"
    TIMELINE = "timeline"
    REFERENCE_ASSIGNMENT = "reference_assignment"
    WORKFLOW_PROFILE = "workflow_profile"
    DIMENSIONS = "dimensions"
    REVIEW_ORDERING = "review_ordering"


class Freshness(Enum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArtifactFingerprint:
    artifact_kind: ArtifactKind
    scene_number: int | None
    prompt_hash: str | None = None
    workflow_hash: str | None = None
    reference_hash: str | None = None
    timeline_hash: str | None = None
    dimensions_hash: str | None = None


# Dependencies: artifact -> artifacts it depends on
_ARTIFACT_DEPENDENCIES: dict[ArtifactKind, frozenset[ArtifactKind]] = {
    ArtifactKind.AUDIO_ANALYSIS: frozenset(),
    ArtifactKind.BEAT_MARKERS: frozenset({ArtifactKind.AUDIO_ANALYSIS}),
    ArtifactKind.AUDIO_TIMELINE: frozenset(
        {ArtifactKind.AUDIO_ANALYSIS, ArtifactKind.BEAT_MARKERS}
    ),
    ArtifactKind.PREPARED_WORKFLOW: frozenset(
        {ArtifactKind.SCENE_RENDER}
    ),
    ArtifactKind.SCENE_RENDER: frozenset(
        {
            ArtifactKind.AUDIO_TIMELINE,
            ArtifactKind.PROMPT_GENERATION,
            ArtifactKind.REFERENCE_SHEETS,
        }
    ),
    ArtifactKind.SCENE_STORYBOARD: frozenset(
        {ArtifactKind.PROMPT_GENERATION, ArtifactKind.REFERENCE_SHEETS}
    ),
    ArtifactKind.REFERENCE_SHEETS: frozenset(
        {ArtifactKind.REFERENCE_SOURCES}
    ),
    ArtifactKind.REFERENCE_SOURCES: frozenset(),
    ArtifactKind.FINAL_VIDEO: frozenset(
        {ArtifactKind.PREPARED_WORKFLOW, ArtifactKind.REVIEW_ORDERING}
    ),
    ArtifactKind.REVIEW_ORDERING: frozenset(),
    ArtifactKind.RENDER_PLAN: frozenset(
        {ArtifactKind.AUDIO_TIMELINE, ArtifactKind.PROMPT_GENERATION}
    ),
    ArtifactKind.PROMPT_GENERATION: frozenset(
        {ArtifactKind.AUDIO_TIMELINE, ArtifactKind.REFERENCE_SHEETS}
    ),
}

_GLOBAL_ARTIFACTS = frozenset({
    ArtifactKind.FINAL_VIDEO,
    ArtifactKind.REVIEW_ORDERING,
})


@dataclass(frozen=True)
class _RebuildStage:
    name: str
    order: int


PLANNING_STAGE = _RebuildStage("planning", 1)
REFERENCE_STAGE = _RebuildStage("references", 2)
RENDER_STAGE = _RebuildStage("render", 3)
FINAL_STAGE = _RebuildStage("final", 4)


@dataclass(frozen=True)
class RebuildPlan:
    rebuild: FrozenSet[ArtifactKind] = frozenset()
    reuse: FrozenSet[ArtifactKind] = frozenset()
    invalidate: FrozenSet[ArtifactKind] = frozenset()
    unknown: FrozenSet[ArtifactKind] = frozenset()
    affected_scenes: FrozenSet[int] = frozenset()
    stages: tuple[_RebuildStage, ...] = ()


@dataclass(frozen=True)
class ChangeSet:
    change_kinds: FrozenSet[ChangeKind] = frozenset()
    scene_numbers: FrozenSet[int] | None = None

    @classmethod
    def prompt(cls, scene_numbers: set[int]) -> ChangeSet:
        return cls(change_kinds=frozenset({ChangeKind.PROMPT}), scene_numbers=frozenset(scene_numbers))

    @classmethod
    def timeline(cls, scene_numbers: set[int]) -> ChangeSet:
        return cls(change_kinds=frozenset({ChangeKind.TIMELINE}), scene_numbers=frozenset(scene_numbers))

    @classmethod
    def reference_assignment(cls, scene_numbers: set[int]) -> ChangeSet:
        return cls(
            change_kinds=frozenset({ChangeKind.REFERENCE_ASSIGNMENT}),
            scene_numbers=frozenset(scene_numbers),
        )

    @classmethod
    def references(cls, scene_numbers: set[int]) -> ChangeSet:
        return cls.reference_assignment(scene_numbers)

    @classmethod
    def workflow_profile(cls) -> ChangeSet:
        return cls(change_kinds=frozenset({ChangeKind.WORKFLOW_PROFILE}))

    @classmethod
    def dimensions(cls) -> ChangeSet:
        return cls(change_kinds=frozenset({ChangeKind.DIMENSIONS}))

    @classmethod
    def review_ordering(cls) -> ChangeSet:
        return cls(change_kinds=frozenset({ChangeKind.REVIEW_ORDERING}))

    @classmethod
    def empty(cls) -> ChangeSet:
        return cls()

    @staticmethod
    def combine(*changes: ChangeSet) -> ChangeSet:
        kinds: set[ChangeKind] = set()
        scenes: set[int] = set()
        for change in changes:
            kinds.update(change.change_kinds)
            if change.scene_numbers is not None:
                scenes.update(change.scene_numbers)
        return ChangeSet(
            change_kinds=frozenset(kinds),
            scene_numbers=frozenset(scenes) if scenes else None,
        )


def _resolve_downstream(
    kind: ArtifactKind,
    visited: set[ArtifactKind],
    *,
    skip_global: bool = False,
) -> set[ArtifactKind]:
    """Resolve all artifacts that depend on the given kind (direct and indirect)."""
    result = set()
    for artifact, deps in _ARTIFACT_DEPENDENCIES.items():
        if kind in deps and artifact not in visited:
            if skip_global and artifact in _GLOBAL_ARTIFACTS:
                continue
            visited.add(artifact)
            result.add(artifact)
            result.update(_resolve_downstream(artifact, visited, skip_global=skip_global))
    return result


# Map ChangeKind -> directly affected artifact kinds
_CHANGE_TO_ARTIFACT: dict[ChangeKind, set[ArtifactKind]] = {
    ChangeKind.PROMPT: {ArtifactKind.PROMPT_GENERATION},
    ChangeKind.TIMELINE: {ArtifactKind.AUDIO_TIMELINE},
    ChangeKind.REFERENCE_ASSIGNMENT: {ArtifactKind.REFERENCE_SHEETS},
    ChangeKind.WORKFLOW_PROFILE: {ArtifactKind.PREPARED_WORKFLOW},
    ChangeKind.DIMENSIONS: {ArtifactKind.SCENE_RENDER},
    ChangeKind.REVIEW_ORDERING: {ArtifactKind.REVIEW_ORDERING},
}


def preview_rebuild(
    change: ChangeSet,
    current_fingerprints: dict[ArtifactKind, Freshness] | None = None,
) -> RebuildPlan:
    """Compute which artifacts need rebuild given a change set."""
    if not change.change_kinds:
        return RebuildPlan()

    directly_affected: set[ArtifactKind] = set()
    for kind in change.change_kinds:
        directly_affected.update(_CHANGE_TO_ARTIFACT.get(kind, set()))

    is_global = change.scene_numbers is None

    # For changes that affect the whole pipeline (workflow profile, dimensions)
    # also mark all related scene artifacts
    if ChangeKind.WORKFLOW_PROFILE in change.change_kinds:
        directly_affected.update({
            ArtifactKind.PREPARED_WORKFLOW,
            ArtifactKind.SCENE_RENDER,
            ArtifactKind.FINAL_VIDEO,
            ArtifactKind.REFERENCE_SHEETS,
            ArtifactKind.PROMPT_GENERATION,
            ArtifactKind.RENDER_PLAN,
        })

    if ChangeKind.DIMENSIONS in change.change_kinds:
        directly_affected.update({
            ArtifactKind.SCENE_RENDER,
            ArtifactKind.PREPARED_WORKFLOW,
            ArtifactKind.FINAL_VIDEO,
            ArtifactKind.SCENE_STORYBOARD,
            ArtifactKind.REFERENCE_SHEETS,
        })

    # Resolve downstream dependencies
    rebuild: set[ArtifactKind] = set()
    visited: set[ArtifactKind] = set()
    for artifact in directly_affected:
        rebuild.add(artifact)
        visited.add(artifact)
        rebuild.update(_resolve_downstream(artifact, visited, skip_global=not is_global))

    # Special handling: workflow-profile change should keep audio and reference sources
    if ChangeKind.WORKFLOW_PROFILE in change.change_kinds:
        rebuild.discard(ArtifactKind.AUDIO_TIMELINE)
        rebuild.discard(ArtifactKind.AUDIO_ANALYSIS)
        rebuild.discard(ArtifactKind.BEAT_MARKERS)
        rebuild.discard(ArtifactKind.REFERENCE_SOURCES)

    # Special handling: dimensions change should keep audio
    if ChangeKind.DIMENSIONS in change.change_kinds:
        rebuild.discard(ArtifactKind.AUDIO_TIMELINE)
        rebuild.discard(ArtifactKind.AUDIO_ANALYSIS)
        rebuild.discard(ArtifactKind.BEAT_MARKERS)
        rebuild.discard(ArtifactKind.REFERENCE_SOURCES)

    # Use provenance data to filter out artifacts that are already CURRENT
    if current_fingerprints:
        for kind in list(rebuild):
            if current_fingerprints.get(kind) == Freshness.CURRENT:
                rebuild.discard(kind)

    # Determine reuse: artifacts not in rebuild
    all_kinds = set(ArtifactKind)
    reuse = all_kinds - rebuild

    # Determine affected scenes
    affected_scenes = change.scene_numbers if change.scene_numbers else frozenset()

    # Determine stages
    stages: set[_RebuildStage] = set()
    rebuild_set = frozenset(rebuild)
    if ArtifactKind.PROMPT_GENERATION in rebuild_set or ArtifactKind.RENDER_PLAN in rebuild_set:
        stages.add(PLANNING_STAGE)
    if ArtifactKind.REFERENCE_SHEETS in rebuild_set:
        stages.add(REFERENCE_STAGE)
    if ArtifactKind.SCENE_RENDER in rebuild_set or ArtifactKind.SCENE_STORYBOARD in rebuild_set or ArtifactKind.PREPARED_WORKFLOW in rebuild_set:
        stages.add(RENDER_STAGE)
    if ArtifactKind.FINAL_VIDEO in rebuild_set or ArtifactKind.REVIEW_ORDERING in rebuild_set:
        stages.add(FINAL_STAGE)

    # Determine unknown: artifacts with no provenance can't be determined here,
    # left empty (client fills from actual provenance data)
    return RebuildPlan(
        rebuild=frozenset(rebuild),
        reuse=frozenset(reuse),
        invalidate=frozenset(),
        unknown=frozenset(),
        affected_scenes=affected_scenes,
        stages=tuple(sorted(stages, key=lambda s: s.order)),
    )


def compute_freshness(
    fingerprint: ArtifactFingerprint | None,
    *,
    prompt_hash: str | None = None,
    workflow_hash: str | None = None,
    reference_hash: str | None = None,
    timeline_hash: str | None = None,
    dimensions_hash: str | None = None,
) -> Freshness:
    """Determine whether an artifact is current, stale, or unknown."""
    if fingerprint is None:
        return Freshness.UNKNOWN

    checks = [
        ("prompt_hash", prompt_hash, fingerprint.prompt_hash),
        ("workflow_hash", workflow_hash, fingerprint.workflow_hash),
        ("reference_hash", reference_hash, fingerprint.reference_hash),
        ("timeline_hash", timeline_hash, fingerprint.timeline_hash),
        ("dimensions_hash", dimensions_hash, fingerprint.dimensions_hash),
    ]

    for _name, current, recorded in checks:
        if current is not None and recorded is not None and current != recorded:
            return Freshness.STALE
        if current is not None and recorded is None:
            return Freshness.UNKNOWN
    return Freshness.CURRENT
