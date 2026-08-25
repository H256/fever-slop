from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from feverslop.domain.canonical_render_plan import PromptRole, validate_canonical_plan
from feverslop.domain.canonical_plan_migration import (
    FindingKind,
    MigrationFinding,
    MigrationInput,
    MigrationReport,
    MigrationSource,
    value_hash,
)
from feverslop.errors import FeverSlopDataError

_MISSING = object()


_BASE_FIELDS = (
    (PromptRole.Z_IMAGE, ("z_image", "prompt")),
    (PromptRole.LTX_BASE, ("ltx", "base_prompt")),
    (PromptRole.LTX_I2V, ("ltx", "i2v_prompt_from_t2i")),
    (PromptRole.LTX_RELAY, ("ltx", "prompt_relay")),
    (PromptRole.H3_VIDEO, ("h3", "prompt")),
    (PromptRole.PERFORMANCE_TIMING, ("performance_timing",)),
)


def analyze_canonical_plan_migration(migration_input: MigrationInput) -> MigrationReport:
    sources = [
        MigrationSource(document.path, document.sha256)
        for document in migration_input.documents
    ]
    documents: dict[str, list[dict[str, Any]]] = {}
    findings: list[MigrationFinding] = []
    base_relative = "output/render/plans/base.json"
    for document in migration_input.documents:
        relative = document.path
        if document.error:
            if relative == base_relative:
                raise FeverSlopDataError(f"Malformed canonical base plan: {relative}")
            findings.append(MigrationFinding("unresolved", relative, document.error))
            continue
        value = document.value
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            if relative == base_relative:
                raise FeverSlopDataError(
                    f"Canonical base plan must be a list of objects: {relative}",
                )
            findings.append(MigrationFinding("unresolved", relative, "artifact must be a list of objects"))
            continue
        documents[relative] = value

    if base_relative not in documents:
        raise FeverSlopDataError(f"Canonical base plan does not exist: {base_relative}")
    base = documents[base_relative]
    validate_canonical_plan(base)
    canonical_index = _canonical_index(base)

    _analyze_base_fields(base, base_relative, findings)
    for relative, scenes in documents.items():
        if relative == base_relative:
            continue
        matches = _match_scenes(scenes, relative, canonical_index, findings)
        kind = _artifact_kind(relative)
        if kind == "references":
            baseline = _reference_baseline(documents, relative)
            if baseline is not None:
                _analyze_pass_through(matches, baseline, relative, canonical_index, findings)
            else:
                findings.append(MigrationFinding(
                    "unresolved", relative, "missing comparison baseline",
                ))
        elif kind == "ingredients":
            baseline = _ingredients_baseline(documents, relative)
            if baseline is not None:
                _analyze_ingredients_relay(matches, baseline, relative, canonical_index, findings)
            else:
                findings.append(MigrationFinding(
                    "unresolved", relative, "missing comparison baseline",
                ))
        elif kind == "legacy_base":
            _analyze_matched_base_fields(matches, relative, findings)

    findings = _resolve_candidate_conflicts(findings)
    findings.sort(key=_finding_sort_key)
    sources.sort(key=lambda item: item.path)
    return MigrationReport(
        migration_input.project_id,
        tuple(sources),
        tuple(findings),
        deepcopy(base),
    )


def _canonical_index(base: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene_id: dict[str, dict[str, Any]] = {}
    by_segment_id: dict[str, dict[str, Any]] = {}
    by_number: dict[int, dict[str, Any]] = {}
    for scene in base:
        canonical = scene.get("canonical")
        if not isinstance(canonical, Mapping):
            raise FeverSlopDataError("Every base-plan scene must have canonical identity")
        by_scene_id[str(canonical["scene_id"])] = scene
        by_segment_id[str(canonical["segment_id"])] = scene
        number = scene.get("scene")
        if isinstance(number, int):
            if number in by_number:
                raise FeverSlopDataError(f"duplicate base scene number: {number}")
            by_number[number] = scene
    return {"scene_id": by_scene_id, "segment_id": by_segment_id, "scene": by_number}


def _analyze_base_fields(
    scenes: list[dict[str, Any]], source: str, findings: list[MigrationFinding],
) -> None:
    for scene in scenes:
        canonical = scene["canonical"]
        for role, path in _BASE_FIELDS:
            candidate = _get(scene, path)
            generated = _get(canonical, ("roles", str(role), "generated", "value"))
            if candidate is _MISSING or generated is _MISSING or candidate == generated:
                continue
            findings.append(_candidate_finding(scene, source, role, path, candidate, "scene_id"))


def _analyze_matched_base_fields(
    matches: list[tuple[dict[str, Any], dict[str, Any], str]],
    source: str,
    findings: list[MigrationFinding],
) -> None:
    for derived, base, matched_by in matches:
        for role, path in _BASE_FIELDS:
            candidate = _get(derived, path)
            generated = _get(base["canonical"], ("roles", str(role), "generated", "value"))
            if candidate is _MISSING or generated is _MISSING or candidate == generated:
                continue
            findings.append(_candidate_finding(base, source, role, path, candidate, matched_by))


def _analyze_pass_through(
    matches: list[tuple[dict[str, Any], dict[str, Any], str]],
    baseline_scenes: list[dict[str, Any]],
    source: str,
    canonical_index: dict[str, Any],
    findings: list[MigrationFinding],
) -> None:
    baseline_matches = _match_scenes(baseline_scenes, source, canonical_index, findings, report_errors=False)
    baseline_by_id = {base["canonical"]["scene_id"]: derived for derived, base, _ in baseline_matches}
    for derived, base, matched_by in matches:
        baseline = baseline_by_id.get(base["canonical"]["scene_id"])
        if baseline is None:
            findings.append(_identity_finding("unresolved", source, "missing comparison baseline", derived))
            continue
        for role, path in _BASE_FIELDS:
            # references.json is regenerated before H3 prompts.  Its H3 field
            # can therefore lag behind the canonical plan without being a
            # manual edit that should be migrated back into the canonical plan.
            if role == PromptRole.H3_VIDEO and source.endswith("/references.json"):
                continue
            candidate = _get(derived, path)
            previous = _get(baseline, path)
            if candidate is _MISSING or previous is _MISSING or candidate == previous:
                continue
            findings.append(_candidate_finding(base, source, role, path, candidate, matched_by))


def _analyze_ingredients_relay(
    matches: list[tuple[dict[str, Any], dict[str, Any], str]],
    baseline_scenes: list[dict[str, Any]],
    source: str,
    canonical_index: dict[str, Any],
    findings: list[MigrationFinding],
) -> None:
    baseline_matches = _match_scenes(baseline_scenes, source, canonical_index, findings, report_errors=False)
    baseline_by_id = {base["canonical"]["scene_id"]: derived for derived, base, _ in baseline_matches}
    for derived, base, matched_by in matches:
        baseline = baseline_by_id.get(base["canonical"]["scene_id"])
        candidate = _get(derived, ("ltx", "prompt_relay"))
        previous = _get(baseline or {}, ("ltx", "msr_prompt_relay"))
        if candidate is _MISSING or previous is _MISSING or candidate == previous:
            continue
        findings.append(_candidate_finding(
            base, source, PromptRole.INGREDIENTS_RELAY, ("ltx", "prompt_relay"), candidate, matched_by,
        ))


def _artifact_kind(relative: str) -> str:
    name = relative.rsplit("/", 1)[-1]
    if name == "references.json" or name.endswith("_refs.json"):
        return "references"
    if name == "ingredients.json" or name.endswith("_ingredients.json"):
        return "ingredients"
    if name.startswith("render_plan_"):
        return "legacy_base"
    return "other"


def _reference_baseline(
    documents: dict[str, list[dict[str, Any]]], source: str,
) -> list[dict[str, Any]] | None:
    anchored = next(
        (value for path, value in documents.items() if path.endswith("/plans/anchored.json")),
        None,
    )
    if anchored is not None:
        return anchored
    legacy_base = source.removesuffix("_refs.json") + ".json"
    return documents.get(legacy_base) or documents.get("output/render/plans/base.json")


def _ingredients_baseline(
    documents: dict[str, list[dict[str, Any]]], source: str,
) -> list[dict[str, Any]] | None:
    references = next(
        (value for path, value in documents.items() if path.endswith("/plans/references.json")),
        None,
    )
    if references is not None:
        return references
    legacy_references = source.removesuffix("_ingredients.json") + "_refs.json"
    return documents.get(legacy_references)


def _match_scenes(
    scenes: list[dict[str, Any]],
    source: str,
    canonical_index: dict[str, Any],
    findings: list[MigrationFinding],
    *,
    report_errors: bool = True,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    identities = [_scene_identity(scene) for scene in scenes]
    counts = Counter(identity for identity in identities if identity is not None)
    matches = []
    matched_scene_ids: set[str] = set()
    for scene, identity in zip(scenes, identities, strict=True):
        if identity is not None and counts[identity] > 1:
            if report_errors:
                findings.append(_identity_finding("unresolved", source, "duplicate scene identity", scene))
            continue
        identity_matches: list[tuple[str, dict[str, Any]]] = []
        canonical = scene.get("canonical")
        if isinstance(canonical, Mapping):
            scene_id = canonical.get("scene_id")
            if isinstance(scene_id, str):
                target = canonical_index["scene_id"].get(scene_id)
                if target is not None:
                    identity_matches.append(("scene_id", target))
        segment_id = _segment_id(scene)
        if segment_id:
            target = canonical_index["segment_id"].get(segment_id)
            if target is not None:
                identity_matches.append(("segment_id", target))
        if isinstance(scene.get("scene"), int):
            target = canonical_index["scene"].get(scene["scene"])
            if target is not None:
                identity_matches.append(("scene_number", target))
        target_ids = {
            target["canonical"]["scene_id"]
            for _, target in identity_matches
        }
        if len(target_ids) > 1:
            if report_errors:
                findings.append(_identity_finding("unresolved", source, "conflicting scene identity", scene))
            continue
        if not identity_matches:
            if report_errors:
                findings.append(_identity_finding("unresolved", source, "orphan scene", scene))
            continue
        matched_by, matched = identity_matches[0]
        matched_scene_ids.add(str(matched["canonical"]["scene_id"]))
        matches.append((scene, matched, str(matched_by)))
    if report_errors:
        for scene_id, base_scene in canonical_index["scene_id"].items():
            if scene_id not in matched_scene_ids:
                findings.append(_identity_finding("unresolved", source, "missing scene", base_scene))
    return matches


def _scene_identity(scene: Mapping[str, Any]) -> tuple[str, Any] | None:
    canonical = scene.get("canonical")
    if isinstance(canonical, Mapping) and canonical.get("scene_id"):
        return "scene_id", canonical["scene_id"]
    segment_id = _segment_id(scene)
    if segment_id:
        return "segment_id", segment_id
    if isinstance(scene.get("scene"), int):
        return "scene", scene["scene"]
    return None


def _segment_id(scene: Mapping[str, Any]) -> str | None:
    canonical = scene.get("canonical")
    metadata = scene.get("metadata")
    value = (
        canonical.get("segment_id") if isinstance(canonical, Mapping) else None
    ) or scene.get("segment_id") or (
        metadata.get("segment_id") if isinstance(metadata, Mapping) else None
    )
    return str(value) if value else None


def _candidate_finding(
    scene: Mapping[str, Any],
    source: str,
    role: str,
    path: tuple[str, ...],
    value: Any,
    matched_by: str,
) -> MigrationFinding:
    canonical = scene["canonical"]
    existing = _get(canonical, ("roles", str(role), "override", "value"))
    if _is_empty(value):
        kind: FindingKind = "unresolved"
        reason = "candidate value is empty"
    elif existing is not _MISSING and existing == value:
        kind = "no_op"
        reason = "override already contains candidate"
    elif existing is not _MISSING:
        kind = "unresolved"
        reason = "candidate conflicts with existing override"
    else:
        kind = "importable"
        reason = "legacy value differs from baseline"
    return MigrationFinding(
        kind,
        source,
        reason,
        str(canonical["scene_id"]),
        str(canonical["segment_id"]),
        scene.get("scene") if isinstance(scene.get("scene"), int) else None,
        str(role),
        ".".join(path),
        matched_by,
        deepcopy(value),
    )


def _identity_finding(
    kind: FindingKind, source: str, reason: str, scene: Mapping[str, Any],
) -> MigrationFinding:
    canonical = scene.get("canonical")
    scene_id = canonical.get("scene_id") if isinstance(canonical, Mapping) else None
    return MigrationFinding(
        kind,
        source,
        reason,
        str(scene_id) if scene_id else None,
        _segment_id(scene),
        scene.get("scene") if isinstance(scene.get("scene"), int) else None,
    )


def _resolve_candidate_conflicts(findings: list[MigrationFinding]) -> list[MigrationFinding]:
    groups: dict[tuple[str | None, str | None], list[MigrationFinding]] = {}
    others = []
    for finding in findings:
        if finding.kind == "importable":
            groups.setdefault((finding.scene_id, finding.role), []).append(finding)
        else:
            others.append(finding)
    for group in groups.values():
        unique = {value_hash(item.value) for item in group}
        if len(unique) == 1:
            others.append(group[0])
            continue
        first = group[0]
        others.append(MigrationFinding(
            "unresolved",
            ", ".join(sorted({item.source_path for item in group})),
            "conflicting candidate values",
            first.scene_id,
            first.segment_id,
            first.scene_number,
            first.role,
            first.field_path,
            first.matched_by,
        ))
    return others


def _get(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (list, tuple, dict)) and not value
    )


def _finding_sort_key(finding: MigrationFinding) -> tuple[str, str, str, str]:
    return (
        finding.source_path,
        finding.scene_id or "",
        finding.role or "",
        finding.reason,
    )
