"""Manifest-governed artifact prune service (issue #1388, piece 2).

Application-layer I/O for the ``feverslop artifact prune`` command: candidate
enumeration via ``SceneArtifactLayout``, per-candidate manifest state
collection (``SceneWorkflowManifest.verify`` /
``compare_canonical_dependencies``), eligibility decisions via the pure domain
``evaluate_candidate``, and archive-first deletion.

``upscale.enabled`` is read from the raw ``config.json`` JSON because the
application layer must not import ``feverslop.config``.
"""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from feverslop.domain.artifact_lifecycle import (
    ArtifactKind,
    PlanPruneContext,
    ScenePruneContext,
    classify_artifact,
    evaluate_candidate,
    is_protected,
    lifecycle_class_for,
    protected_reason_for,
)
from feverslop.domain.effective_render_plan import (
    CanonicalSceneDependencies,
    project_effective_plan,
)
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.errors import FeverSlopDataError
from feverslop.ports.reporting import NullReporter, Reporter
from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.tools.project_asset_archive import (
    ArchiveMember,
    build_archive_manifest,
    resolve_available_zip_path,
)
from feverslop.utils.io import read_json_document

__all__ = [
    "PRUNE_REPORT_SCHEMA",
    "PruneReport",
    "apply_prune",
    "create_prune_archive",
    "scan_project",
]

PRUNE_REPORT_SCHEMA = "feverslop.artifact-prune/v1"

_MODE_SAFE = "safe"
_MODE_APPLY = "apply"

# Per-candidate manifest/JSON read failures are ineligibility reasons, never
# fatal. These are the error types the pure readers can raise.
_MANIFEST_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, FeverSlopDataError)


@dataclass
class PruneReport:
    """Machine-readable prune report (``feverslop.artifact-prune/v1``)."""

    created_at: str
    project: str
    mode: str
    archive_path: str | None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    protected: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRUNE_REPORT_SCHEMA,
            "created_at": self.created_at,
            "project": self.project,
            "mode": self.mode,
            "archive_path": self.archive_path,
            "candidates": [dict(entry) for entry in self.candidates],
            "protected": [dict(entry) for entry in self.protected],
            "deleted": [dict(entry) for entry in self.deleted],
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class _SceneState:
    """Per-scene manifest state gathered once for every candidate evaluation."""

    number: int
    manifest_present: bool
    manifest: SceneWorkflowManifest | None
    verify_mismatches: list[str]
    reference_paths: set[Path]


def scan_project(project_dir: str | Path, reporter: Reporter | None = None) -> PruneReport:
    """Scan a project and classify every canonical artifact for pruning.

    Per-candidate manifest problems are ineligibility reasons (never fatal);
    a missing project directory or an I/O failure during the top-level scan
    raises.
    """
    project_dir = Path(project_dir)
    reporter = reporter if reporter is not None else NullReporter()
    if not project_dir.is_dir():
        raise NotADirectoryError(project_dir)

    reporter.step("artifact-prune-scan")
    reporter.message(f"Scanning {project_dir} for prune candidates")

    layout = SceneArtifactLayout(project_dir)
    errors: list[str] = []

    upscale_enabled = _upscale_enabled_from_config(project_dir, errors)
    base_scenes = _load_base_scenes(layout, errors)
    scene_states = _load_scene_states(layout, project_dir, errors)

    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for path in _enumerated_paths(layout):
        if not path.is_file():
            continue
        kind = classify_artifact(path, project_dir)
        entry_path = path.relative_to(project_dir).as_posix()
        try:
            size = path.stat().st_size
            content_sha256 = _sha256_file(path)
        except OSError as exc:
            errors.append(f"cannot stat {entry_path}: {exc}")
            continue
        if is_protected(kind):
            protected.append(
                {
                    "path": entry_path,
                    "class": lifecycle_class_for(kind).value,
                    "size_bytes": size,
                    "reason": protected_reason_for(kind),
                }
            )
            continue
        if kind is ArtifactKind.scene_workflow or kind is ArtifactKind.raw_clip:
            scene_number = _scene_number_from_path(path)
            state = scene_states.get(scene_number)
            context = _scene_context(
                layout, path, scene_number, state, scene_states, base_scenes, upscale_enabled
            )
            eligible, reasons = evaluate_candidate(
                path, project_dir, scene_context=context
            )
        elif kind is ArtifactKind.derived_plan:
            context = _plan_context(path, layout, base_scenes, errors)
            eligible, reasons = evaluate_candidate(
                path, project_dir, plan_context=context
            )
        else:
            # ``other`` is not a candidate in #1388.
            continue
        candidates.append(
            {
                "path": entry_path,
                "class": lifecycle_class_for(kind).value,
                "size_bytes": size,
                "sha256": content_sha256,
                "eligible": eligible,
                "reasons": list(reasons),
            }
        )

    _report_summary(reporter, candidates, protected)

    return PruneReport(
        created_at=_now_iso(),
        project=str(project_dir),
        mode=_MODE_SAFE,
        archive_path=None,
        candidates=candidates,
        protected=protected,
        deleted=[],
        errors=errors,
    )


def create_prune_archive(
    project_dir: str | Path,
    members: list[ArchiveMember],
    output_zip: str | Path,
) -> Path:
    """Write the candidate archive atomically; nothing is deleted on failure.

    The zip (with ``archive_manifest.json``) is fully written before the
    caller may delete anything. A failure removes the partial zip and re-raises.
    """
    project_dir = Path(project_dir)
    output_zip = resolve_available_zip_path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    created_at = _now_iso()
    archive_members, expected_hashes = _snapshot_members(members)
    manifest = build_archive_manifest(project_dir, archive_members, created_at=created_at)
    temporary_zip = output_zip.with_name(f".{output_zip.name}.{uuid4().hex}.tmp")
    try:
        with ZipFile(temporary_zip, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(
                "archive_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)
            )
            for member in archive_members:
                archive.write(member.source, member.arcname)
        _verify_prune_archive(temporary_zip, archive_members, expected_hashes)
        temporary_zip.replace(output_zip)
    except BaseException:
        temporary_zip.unlink(missing_ok=True)
        raise
    return output_zip


def apply_prune(
    project_dir: str | Path,
    archive_path: str | Path,
    report: PruneReport,
    reporter: Reporter | None = None,
) -> PruneReport:
    """Archive the eligible candidates, then delete exactly the archived files.

    The archive is created and verified before any deletion; an archive
    failure raises and leaves every file in place.
    """
    project_dir = Path(project_dir).resolve()
    reporter = reporter if reporter is not None else NullReporter()
    reporter.step("artifact-prune-apply")
    _validate_report_project(project_dir, report)
    reporter.message("Rechecking prune eligibility immediately before archival")
    current_report = scan_project(project_dir, reporter=reporter)
    scanned_by_path = {
        str(entry.get("path")): entry
        for entry in report.candidates
        if entry.get("eligible") is True
    }
    eligible = [
        entry
        for entry in current_report.candidates
        if entry.get("eligible") is True
        and scanned_by_path.get(str(entry.get("path")), {}).get("sha256") == entry.get("sha256")
    ]
    reporter.message(f"Archiving {len(eligible)} eligible candidate(s) to {archive_path}")
    members: list[ArchiveMember] = []
    for entry in eligible:
        source = project_dir / str(entry["path"])
        members.append(
            ArchiveMember(
                source=source, arcname=str(entry["path"]), size=int(entry["size_bytes"])
            )
        )
    archive_path = create_prune_archive(project_dir, members, archive_path)
    expected_hashes = {str(entry["path"]): str(entry["sha256"]) for entry in eligible}
    _verify_prune_archive(archive_path, members, expected_hashes)
    reporter.message(f"Archive complete: {archive_path}")

    for member in members:
        _verify_source_matches_archive(archive_path, member, expected_hashes[member.arcname])

    deleted: list[dict[str, Any]] = []
    for member in members:
        member.source.unlink()
        deleted.append({"path": member.arcname, "size_bytes": member.size})
    reporter.message(f"Deleted {len(deleted)} file(s)")
    reporter.message("Apply complete")

    return PruneReport(
        created_at=current_report.created_at,
        project=current_report.project,
        mode=_MODE_APPLY,
        archive_path=str(archive_path),
        candidates=list(current_report.candidates),
        protected=list(current_report.protected),
        deleted=deleted,
        errors=list(current_report.errors),
    )


def _validate_report_project(project_dir: Path, report: PruneReport) -> None:
    """Reject reports that do not originate from this exact project root."""
    try:
        report_project = Path(report.project).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise FeverSlopDataError("prune report has an invalid project path") from exc
    if report_project != project_dir:
        raise FeverSlopDataError(
            f"prune report belongs to {report_project}, not {project_dir}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_members(
    members: list[ArchiveMember],
) -> tuple[list[ArchiveMember], dict[str, str]]:
    """Capture the exact source sizes and hashes represented by one ZIP."""
    snapshots: list[ArchiveMember] = []
    hashes: dict[str, str] = {}
    for member in members:
        snapshots.append(
            ArchiveMember(
                source=member.source,
                arcname=member.arcname,
                size=member.source.stat().st_size,
            )
        )
        hashes[member.arcname] = _sha256_file(member.source)
    return snapshots, hashes


def _verify_prune_archive(
    archive_path: Path, members: list[ArchiveMember], expected_hashes: dict[str, str]
) -> None:
    """Require a readable archive containing byte-identical selected members."""
    expected_names = {"archive_manifest.json", *(member.arcname for member in members)}
    try:
        with ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise FeverSlopDataError(f"prune archive is corrupt: {archive_path}")
            if set(archive.namelist()) != expected_names:
                raise FeverSlopDataError(f"prune archive members do not match selection: {archive_path}")
            for member in members:
                if hashlib.sha256(archive.read(member.arcname)).hexdigest() != expected_hashes[member.arcname]:
                    raise FeverSlopDataError(f"prune archive content mismatch: {member.arcname}")
    except (BadZipFile, KeyError, OSError) as exc:
        raise FeverSlopDataError(f"prune archive cannot be verified: {archive_path}") from exc


def _verify_source_matches_archive(
    archive_path: Path, member: ArchiveMember, expected_hash: str
) -> None:
    """Do not unlink a file that changed after it was archived."""
    try:
        if _sha256_file(member.source) != expected_hash:
            raise FeverSlopDataError(f"candidate changed after archival: {member.arcname}")
        with ZipFile(archive_path) as archive:
            if hashlib.sha256(archive.read(member.arcname)).hexdigest() != expected_hash:
                raise FeverSlopDataError(f"archive changed before deletion: {member.arcname}")
    except (BadZipFile, KeyError, OSError) as exc:
        raise FeverSlopDataError(f"cannot verify candidate before deletion: {member.arcname}") from exc


def _enumerated_paths(layout: SceneArtifactLayout) -> list[Path]:
    """Bounded, layout-anchored set of candidate + representative protected paths."""
    paths: list[Path] = [layout.project_dir / "config.json", layout.base_plan]
    paths.extend(layout.derived_plan_paths())
    for number in layout.scene_numbers():
        paths.extend(
            [
                layout.scene_manifest(number),
                layout.scene_workflow(number),
                layout.scene_raw_video(number),
                layout.scene_final_video(number),
                layout.scene_final_facefix_video(number),
                layout.scene_upscaled_video(number),
            ]
        )
    prompts_dir = layout.project_dir / "output" / "prompts"
    if prompts_dir.is_dir():
        paths.extend(sorted(prompts_dir.glob("story_plan_*.json")))
    return paths


def _scene_number_from_path(path: Path) -> int:
    # ``path`` is ``<scenes_dir>/scene_NNNN/<file>``; the parent is the scene dir.
    return int(path.parent.name.removeprefix("scene_"))


def _scene_number_of(scene: dict[str, Any]) -> int:
    raw = scene.get("scene")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _load_base_scenes(layout: SceneArtifactLayout, errors: list[str]) -> list[dict[str, Any]]:
    if not layout.base_plan.is_file():
        return []
    data = _read_json(layout.base_plan, errors)
    if not isinstance(data, list):
        return []
    return [scene for scene in data if isinstance(scene, dict)]


def _load_scene_states(
    layout: SceneArtifactLayout, project_dir: Path, errors: list[str]
) -> dict[int, _SceneState]:
    states: dict[int, _SceneState] = {}
    for number in layout.scene_numbers():
        manifest_path = layout.scene_manifest(number)
        manifest: SceneWorkflowManifest | None = None
        verify_mismatches: list[str] = []
        present = manifest_path.is_file()
        if present:
            try:
                manifest = SceneWorkflowManifest.read(manifest_path)
            except _MANIFEST_READ_ERRORS:
                manifest = None
            if manifest is not None:
                try:
                    verify_mismatches = list(manifest.verify(project_dir))
                except _MANIFEST_READ_ERRORS as exc:
                    verify_mismatches = [f"verify failed: {exc}"]
        states[number] = _SceneState(
            number=number,
            manifest_present=present,
            manifest=manifest,
            verify_mismatches=verify_mismatches,
            reference_paths=_manifest_reference_paths(manifest, project_dir),
        )
    return states


def _scene_context(
    layout: SceneArtifactLayout,
    candidate: Path,
    scene_number: int,
    state: _SceneState | None,
    states: dict[int, _SceneState],
    base_scenes: list[dict[str, Any]],
    upscale_enabled: bool,
) -> ScenePruneContext:
    if state is None:
        state = _SceneState(scene_number, False, None, [], set())
    return ScenePruneContext(
        scene_number=scene_number,
        manifest_present=state.manifest_present,
        manifest_valid=state.manifest is not None,
        verify_mismatches=tuple(state.verify_mismatches),
        dependency_mismatches=_dependency_mismatches(state.manifest, base_scenes, scene_number),
        final_successor_present=layout.scene_final_video(scene_number).is_file(),
        facefix_pending=_facefix_pending(layout, scene_number),
        upscale_pending=_upscale_pending(layout, scene_number, upscale_enabled),
        referenced_by_unfinished=_referenced_by_unfinished(
            candidate, scene_number, states, layout
        ),
    )


def _dependency_mismatches(
    manifest: SceneWorkflowManifest | None,
    base_scenes: list[dict[str, Any]],
    scene_number: int,
) -> tuple[str, ...]:
    if manifest is None:
        return ()
    current = _base_scene_dependencies(base_scenes, scene_number)
    if current is None:
        return ("canonical provenance missing",)
    try:
        return tuple(manifest.compare_canonical_dependencies(current))
    except _MANIFEST_READ_ERRORS as exc:
        return (f"dependency comparison failed: {exc}",)


def _base_scene_dependencies(
    base_scenes: list[dict[str, Any]], scene_number: int
) -> CanonicalSceneDependencies | None:
    scene = next((s for s in base_scenes if _scene_number_of(s) == scene_number), None)
    if scene is None:
        return None
    try:
        projected = project_effective_plan([scene])[0]
    except Exception:
        # Per-candidate projection failure: cannot verify provenance, so the
        # candidate is blocked rather than the scan failing.
        return None
    projection = projected.get("canonical_projection")
    if not isinstance(projection, dict):
        return None
    dependencies = projection.get("dependencies")
    if not isinstance(dependencies, dict):
        return None
    try:
        return CanonicalSceneDependencies.from_dict(dependencies)
    except _MANIFEST_READ_ERRORS:
        return None


def _plan_context(
    plan_path: Path,
    layout: SceneArtifactLayout,
    base_scenes: list[dict[str, Any]],
    errors: list[str],
) -> PlanPruneContext:
    base_present = layout.base_plan.is_file()
    data = _read_json(plan_path, errors)
    if not isinstance(data, list):
        return PlanPruneContext(
            base_plan_present=base_present, provenance_present=False
        )
    provenance_present = True
    mismatches: list[str] = []
    for scene in data:
        if not isinstance(scene, dict):
            provenance_present = False
            continue
        number = _scene_number_of(scene)
        stored = _stored_dependencies(scene)
        if stored is None:
            provenance_present = False
            continue
        current = _base_scene_dependencies(base_scenes, number)
        if current is None:
            mismatches.append(f"scene {number}: base provenance missing")
            continue
        if stored.get("workflow_fingerprint") != current.workflow_fingerprint:
            mismatches.append(f"scene {number}: workflow fingerprint changed")
        if stored.get("reference_fingerprint") != current.reference_fingerprint:
            mismatches.append(f"scene {number}: reference fingerprint changed")
    return PlanPruneContext(
        base_plan_present=base_present,
        provenance_present=provenance_present,
        provenance_mismatches=tuple(mismatches),
    )


def _stored_dependencies(scene: dict[str, Any]) -> dict[str, Any] | None:
    projection = scene.get("canonical_projection")
    if not isinstance(projection, dict):
        return None
    dependencies = projection.get("dependencies")
    return dependencies if isinstance(dependencies, dict) else None


def _manifest_reference_paths(
    manifest: SceneWorkflowManifest | None, project_dir: Path
) -> set[Path]:
    if manifest is None:
        return set()
    artifacts = [
        *manifest.assets,
        manifest.startframe_source_clip,
        manifest.first_frame_path,
        manifest.last_frame_path,
    ]
    paths: set[Path] = set()
    for artifact in artifacts:
        if artifact is None:
            continue
        try:
            paths.add(artifact.resolve(project_dir).resolve())
        except (OSError, ValueError):
            continue
    return paths


def _referenced_by_unfinished(
    candidate: Path,
    scene_number: int,
    states: dict[int, _SceneState],
    layout: SceneArtifactLayout,
) -> bool:
    try:
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    for other_number, other in states.items():
        if other_number == scene_number or other.manifest is None:
            continue
        if candidate_resolved not in other.reference_paths:
            continue
        final_ok = layout.scene_final_video(other_number).is_file()
        manifest_ok = not other.verify_mismatches
        if not (final_ok and manifest_ok):
            return True
    return False


def _facefix_pending(layout: SceneArtifactLayout, scene_number: int) -> bool:
    if layout.scene_final_facefix_video(scene_number).is_file():
        return False
    if layout.scene_workflow_facefix(scene_number).is_file():
        return True
    facefix_dir = layout.scene_dir(scene_number) / "facefix"
    if facefix_dir.is_dir() and any(facefix_dir.iterdir()):
        return True
    return False


def _upscale_pending(
    layout: SceneArtifactLayout, scene_number: int, upscale_enabled: bool
) -> bool:
    if layout.scene_upscaled_video(scene_number).is_file():
        return False
    if upscale_enabled:
        return True
    scene_dir = layout.scene_dir(scene_number)
    if any(scene_dir.glob("upscale_pass_*")):
        return True
    return False


def _upscale_enabled_from_config(project_dir: Path, errors: list[str]) -> bool:
    config_path = project_dir / "config.json"
    if not config_path.is_file():
        return False
    data = _read_json(config_path, errors)
    if not isinstance(data, dict):
        return False
    upscale = data.get("upscale")
    if not isinstance(upscale, dict):
        return False
    return bool(upscale.get("enabled", False))


def _read_json(path: Path, errors: list[str]) -> Any:
    try:
        return read_json_document(path)
    except _MANIFEST_READ_ERRORS as exc:
        errors.append(f"cannot read {path.name}: {exc}")
        return None


def _report_summary(
    reporter: Reporter,
    candidates: list[dict[str, Any]],
    protected: list[dict[str, Any]],
) -> None:
    eligible = sum(1 for entry in candidates if entry["eligible"])
    by_class: dict[str, int] = {}
    for entry in [*candidates, *protected]:
        by_class[entry["class"]] = by_class.get(entry["class"], 0) + 1
    summary = ", ".join(f"{name}={count}" for name, count in sorted(by_class.items())) or "none"
    reporter.message(
        f"Scanned {len(candidates)} candidate(s) ({eligible} eligible), "
        f"{len(protected)} protected"
    )
    reporter.message(f"Class summary: {summary}")
    reporter.message("Scan complete")


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()
