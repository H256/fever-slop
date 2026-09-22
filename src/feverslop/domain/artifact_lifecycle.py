"""Manifest-governed artifact lifecycle classification (issue #1388, Piece 1).

Pure, path-anchored classification of project artifacts into the
authoritative / resume-cache / review-export lifecycle classes, plus a
pure eligibility decision for prune candidates.

The module performs no filesystem I/O beyond ``pathlib`` path
comparison. All manifest state (``SceneWorkflowManifest.verify`` /
``compare_canonical_dependencies`` results), file-presence booleans,
pending-stage signals, and cross-scene references are injected by the
application layer (Piece 2) via ``ScenePruneContext`` /
``PlanPruneContext``.

Notes:
- ``user_override`` entries live inside ``base.json`` (the canonical
  plan); an override-bearing ``base.json`` is classified as
  ``canonical_plan`` and is protected through it.
- ``other`` (facefix/upscale intermediates, ``h3_prompt.json``, ...) is
  not a candidate in #1388.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ArtifactKind(str, Enum):
    """What an artifact is, anchored to the canonical project layout."""

    config = "config"
    canonical_plan = "canonical_plan"
    #: Override entries live inside ``base.json``; classified as
    #: ``canonical_plan`` (protected) — see module docstring.
    user_override = "user_override"
    reference_asset = "reference_asset"
    reference_manifest = "reference_manifest"
    final_scene_clip = "final_scene_clip"
    assembled_final_video = "assembled_final_video"
    review_export = "review_export"
    scene_workflow = "scene_workflow"
    raw_clip = "raw_clip"
    derived_plan = "derived_plan"
    other = "other"


class ArtifactLifecycleClass(str, Enum):
    """What an artifact is authoritative for (mirrors #1384 ``ArtifactClass``)."""

    authoritative = "authoritative"
    resume_cache = "resume_cache"
    review_export = "review_export"


class IneligibilityReason(str, Enum):
    """Why a prune candidate is not eligible for deletion."""

    manifest_missing = "manifest_missing"
    manifest_invalid = "manifest_invalid"
    fingerprint_mismatch = "fingerprint_mismatch"
    no_final_successor = "no_final_successor"
    facefix_pending = "facefix_pending"
    upscale_pending = "upscale_pending"
    dependencies_missing = "dependencies_missing"
    referenced_by_unfinished = "referenced_by_unfinished"


#: Kinds that are never pruned (issue #1388 acceptance criterion 5).
PROTECTED_KINDS: frozenset[ArtifactKind] = frozenset(
    {
        ArtifactKind.config,
        ArtifactKind.canonical_plan,
        ArtifactKind.user_override,
        ArtifactKind.reference_asset,
        ArtifactKind.reference_manifest,
        ArtifactKind.final_scene_clip,
        ArtifactKind.assembled_final_video,
        ArtifactKind.review_export,
    }
)

#: Kinds that may be pruned when their manifest is valid and matching.
CANDIDATE_KINDS: frozenset[ArtifactKind] = frozenset(
    {
        ArtifactKind.scene_workflow,
        ArtifactKind.raw_clip,
        ArtifactKind.derived_plan,
    }
)

_LIFECYCLE_CLASS_BY_KIND: dict[ArtifactKind, ArtifactLifecycleClass | None] = {
    ArtifactKind.config: ArtifactLifecycleClass.authoritative,
    ArtifactKind.canonical_plan: ArtifactLifecycleClass.authoritative,
    ArtifactKind.user_override: ArtifactLifecycleClass.authoritative,
    ArtifactKind.reference_asset: ArtifactLifecycleClass.authoritative,
    ArtifactKind.reference_manifest: ArtifactLifecycleClass.authoritative,
    ArtifactKind.final_scene_clip: ArtifactLifecycleClass.authoritative,
    ArtifactKind.assembled_final_video: ArtifactLifecycleClass.authoritative,
    ArtifactKind.review_export: ArtifactLifecycleClass.review_export,
    ArtifactKind.scene_workflow: ArtifactLifecycleClass.resume_cache,
    ArtifactKind.raw_clip: ArtifactLifecycleClass.resume_cache,
    ArtifactKind.derived_plan: ArtifactLifecycleClass.resume_cache,
    ArtifactKind.other: None,
}

_PROTECTED_REASON_BY_KIND: dict[ArtifactKind, str] = {
    ArtifactKind.config: "project configuration is authoritative",
    ArtifactKind.canonical_plan: "canonical plan (base.json / story plan) is authoritative",
    ArtifactKind.user_override: "user overrides live inside the canonical plan",
    ArtifactKind.reference_asset: "reference assets are authoritative inputs",
    ArtifactKind.reference_manifest: "reference and scene manifests are authoritative",
    ArtifactKind.final_scene_clip: "final scene clips are authoritative outputs",
    ArtifactKind.assembled_final_video: "assembled final videos are authoritative outputs",
    ArtifactKind.review_export: "review exports are retained for inspection",
}

_FINAL_SCENE_CLIP_NAMES = frozenset({"final.mp4", "final_facefix.mp4", "upscale_final.mp4"})
_DERIVED_PLAN_NAMES = frozenset(
    {"compact.json", "anchored.json", "references.json", "ingredients.json"}
)
_REVIEW_EXPORT_SUFFIXES = frozenset({".md", ".html"})


def _relative_parts(path: Path, project_dir: Path) -> tuple[str, ...] | None:
    """Return ``path`` relative to ``project_dir`` as name parts, or None."""
    for candidate_path, candidate_root in (
        (path, project_dir),
        (Path(path).resolve(), Path(project_dir).resolve()),
    ):
        try:
            return candidate_path.relative_to(candidate_root).parts
        except ValueError:
            continue
    return None


def _is_scene_dir_name(name: str) -> bool:
    suffix = name.removeprefix("scene_")
    return bool(suffix) and suffix.isdigit()


def classify_artifact(path: Path, project_dir: Path) -> ArtifactKind:
    """Classify ``path`` by its position in the canonical project layout.

    The classification is path-anchored (see ``SceneArtifactLayout``):
    only the canonical layout paths map to a known kind; everything
    else — including paths outside the project — is ``other``.
    """
    parts = _relative_parts(path, project_dir)
    if parts is None:
        return ArtifactKind.other
    name = parts[-1]

    if parts == ("config.json",):
        return ArtifactKind.config
    if parts[0] != "output":
        return ArtifactKind.other
    if Path(name).suffix.lower() in _REVIEW_EXPORT_SUFFIXES:
        return ArtifactKind.review_export
    if len(parts) >= 2 and parts[1] == "references":
        return (
            ArtifactKind.reference_manifest
            if name == "manifest.json"
            else ArtifactKind.reference_asset
        )
    if len(parts) >= 2 and parts[1] == "prompts":
        if name.startswith("story_plan_") and name.endswith(".json"):
            return ArtifactKind.canonical_plan
        return ArtifactKind.other
    if len(parts) >= 3 and parts[1] == "render":
        if len(parts) >= 4 and parts[2] == "plans":
            if name == "base.json":
                return ArtifactKind.canonical_plan
            if name in _DERIVED_PLAN_NAMES:
                return ArtifactKind.derived_plan
            return ArtifactKind.other
        if len(parts) >= 4 and parts[2] == "final":
            return ArtifactKind.assembled_final_video
        if len(parts) >= 5 and parts[2] == "scenes" and _is_scene_dir_name(parts[3]):
            if name == "manifest.json":
                return ArtifactKind.reference_manifest
            if name == "workflow.json":
                return ArtifactKind.scene_workflow
            if name == "raw.mp4":
                return ArtifactKind.raw_clip
            if name in _FINAL_SCENE_CLIP_NAMES:
                return ArtifactKind.final_scene_clip
            return ArtifactKind.other
        return ArtifactKind.other
    return ArtifactKind.other


def lifecycle_class_for(kind: ArtifactKind) -> ArtifactLifecycleClass | None:
    """Return the lifecycle class for ``kind`` (None for ``other``)."""
    return _LIFECYCLE_CLASS_BY_KIND[kind]


def is_protected(kind: ArtifactKind) -> bool:
    """Return whether ``kind`` is never pruned."""
    return kind in PROTECTED_KINDS


def is_candidate(kind: ArtifactKind) -> bool:
    """Return whether ``kind`` may be pruned when its manifest is valid."""
    return kind in CANDIDATE_KINDS


def protected_reason_for(kind: ArtifactKind) -> str:
    """Return the human-readable protected reason (empty for non-protected)."""
    return _PROTECTED_REASON_BY_KIND.get(kind, "")


@dataclass(frozen=True)
class ScenePruneContext:
    """I/O-free state for one scene's prune evaluation.

    Gathered by the application layer:

    - ``manifest_present``: ``scene_NNNN/manifest.json`` exists.
    - ``manifest_valid``: the manifest is readable and supports schema v1-v3.
    - ``verify_mismatches``: ``SceneWorkflowManifest.verify()`` result
      (empty = clean; covers missing manifest-referenced files).
    - ``dependency_mismatches``:
      ``SceneWorkflowManifest.compare_canonical_dependencies()`` result
      vs the base plan's ``canonical_projection.dependencies``.
    - ``final_successor_present``: ``scene_NNNN/final.mp4`` exists.
    - ``facefix_pending``: facefix started but incomplete
      (``workflow_facefix.json`` or ``facefix/`` intermediates present
      without ``final_facefix.mp4``).
    - ``upscale_pending``: upscale enabled or started but incomplete
      (``config.upscale.enabled`` or ``upscale_pass_*`` intermediates
      present without ``upscale_final.mp4``).
    - ``referenced_by_unfinished``: another scene's manifest
      (``startframe_source_clip``, ``first_frame_path``,
      ``last_frame_path``, assets) points at the candidate and that
      scene's final chain is incomplete or its manifest is invalid.
    """

    scene_number: int
    manifest_present: bool
    manifest_valid: bool
    verify_mismatches: tuple[str, ...] = ()
    dependency_mismatches: tuple[str, ...] = ()
    final_successor_present: bool = False
    facefix_pending: bool = False
    upscale_pending: bool = False
    referenced_by_unfinished: bool = False


@dataclass(frozen=True)
class PlanPruneContext:
    """I/O-free state for one derived plan's prune evaluation.

    Gathered by the application layer:

    - ``base_plan_present``: ``output/render/plans/base.json`` exists.
    - ``provenance_present``: the plan carries per-scene
      ``canonical_projection.dependencies``.
    - ``provenance_mismatches``: drift of the plan's per-scene
      ``canonical_projection.dependencies`` vs the base plan's
      (the ``canonical_plan_cli`` STALE check).
    """

    base_plan_present: bool = False
    provenance_present: bool = False
    provenance_mismatches: tuple[str, ...] = ()


@dataclass(frozen=True)
class PruneCandidate:
    """One classified artifact with its prune decision."""

    path: Path
    relative_path: str
    kind: ArtifactKind
    lifecycle_class: ArtifactLifecycleClass | None
    size_bytes: int
    eligible: bool
    reasons: tuple[str, ...] = ()
    protected_reason: str = ""


def _scene_reasons(context: ScenePruneContext) -> list[str]:
    reasons: list[str] = []
    if not context.manifest_present:
        reasons.append(IneligibilityReason.manifest_missing.value)
    elif not context.manifest_valid or context.verify_mismatches:
        reasons.append(IneligibilityReason.manifest_invalid.value)
    if context.dependency_mismatches:
        reasons.append(IneligibilityReason.fingerprint_mismatch.value)
    if not context.final_successor_present:
        reasons.append(IneligibilityReason.no_final_successor.value)
    if context.facefix_pending:
        reasons.append(IneligibilityReason.facefix_pending.value)
    if context.upscale_pending:
        reasons.append(IneligibilityReason.upscale_pending.value)
    if context.referenced_by_unfinished:
        reasons.append(IneligibilityReason.referenced_by_unfinished.value)
    return reasons


def _plan_reasons(context: PlanPruneContext) -> list[str]:
    reasons: list[str] = []
    if not context.provenance_present:
        reasons.append(IneligibilityReason.manifest_missing.value)
    if context.provenance_mismatches:
        reasons.append(IneligibilityReason.fingerprint_mismatch.value)
    if not context.base_plan_present:
        reasons.append(IneligibilityReason.dependencies_missing.value)
    return reasons


def evaluate_candidate(
    candidate_path: Path,
    project_dir: Path,
    *,
    scene_context: ScenePruneContext | None = None,
    plan_context: PlanPruneContext | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Pure eligibility decision for one artifact.

    Returns ``(eligible, reasons)``. A candidate is eligible iff it has
    zero ineligibility reasons; all reasons are collected (not
    first-failure) so the report can explain every blocker. Protected
    kinds are never eligible (``protected_reason_for`` explains why);
    ``other`` is not a candidate in #1388.
    """
    kind = classify_artifact(candidate_path, project_dir)
    if kind in PROTECTED_KINDS:
        return False, ()
    if kind is ArtifactKind.other:
        return False, ("not_a_candidate",)
    if kind is ArtifactKind.derived_plan:
        if plan_context is None:
            raise ValueError("derived plan candidate requires plan_context")
        reasons = _plan_reasons(plan_context)
    else:
        if scene_context is None:
            raise ValueError("scene-scoped candidate requires scene_context")
        reasons = _scene_reasons(scene_context)
    return (len(reasons) == 0, tuple(reasons))
