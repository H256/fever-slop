from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from typing import Any

from feverslop.domain.canonical_plan_regeneration import (
    CanonicalRegenerationResult,
    RegenerationDiagnostic,
)
from feverslop.errors import FeverSlopDataError

_REFERENCE_BINDING_KEYS = frozenset({
    "actor_sheet_paths",
    "actor_msr_paths",
    "actor_reference_descriptions",
    "location_sheet_path",
    "location_msr_path",
    "location_reference_description",
    "visual_consistency_sources",
})


class CanonicalPlanRegenerationService:
    def merge(
        self,
        existing_scenes: Sequence[Mapping[str, Any]],
        generated_scenes: Sequence[Mapping[str, Any]],
        *,
        selected_scene_numbers: Set[int] | None = None,
        reference_scenes: Sequence[Mapping[str, Any]] = (),
    ) -> CanonicalRegenerationResult:
        existing = _index_scenes(existing_scenes, "existing")
        generated = _index_scenes(generated_scenes, "generated")
        diagnostics: list[RegenerationDiagnostic] = []
        references = _reference_bindings(reference_scenes, diagnostics)

        if selected_scene_numbers is not None:
            merged = self._merge_selected(
                existing_scenes,
                existing,
                generated,
                selected_scene_numbers,
                references,
                diagnostics,
            )
        else:
            merged = [
                _merge_scene(scene, existing.get(scene_id), references.get(scene_id))
                for scene_id, scene in generated.items()
            ]
            for scene_id, scene in existing.items():
                if scene_id not in generated:
                    diagnostics.append(_orphan_diagnostic(scene_id, scene))
            for scene_id in references.keys() - generated.keys():
                diagnostics.append(RegenerationDiagnostic(
                    "orphaned_reference_scene",
                    "Reference bindings did not match a regenerated canonical identity.",
                    scene_id=scene_id,
                ))

        shared_settings = _shared_project_setting_fields(existing_scenes)
        for scene in merged:
            scene_id = _canonical_identity(scene, "merged")[0]
            if scene_id not in existing:
                _apply_project_setting_fields(scene, shared_settings)

        return CanonicalRegenerationResult(
            tuple(deepcopy(scene) for scene in merged),
            tuple(diagnostics),
        )

    def _merge_selected(
        self,
        existing_scenes: Sequence[Mapping[str, Any]],
        existing: dict[str, Mapping[str, Any]],
        generated: dict[str, Mapping[str, Any]],
        selected: Set[int],
        references: dict[str, tuple[str, dict[str, Any]]],
        diagnostics: list[RegenerationDiagnostic],
    ) -> list[dict[str, Any]]:
        generated_selected = {
            scene_id: scene
            for scene_id, scene in generated.items()
            if _scene_number(scene) in selected
        }
        occupied_scene_numbers = {
            _scene_number(scene)
            for scene in existing_scenes
            if _scene_number(scene) in selected
        }
        merged: list[dict[str, Any]] = []
        handled: set[str] = set()
        for old_scene in existing_scenes:
            scene_id = _canonical_identity(old_scene, "existing")[0]
            if _scene_number(old_scene) not in selected:
                merged.append(deepcopy(dict(old_scene)))
                continue
            replacement = generated_selected.get(scene_id)
            if replacement is None:
                diagnostics.append(RegenerationDiagnostic(
                    "selected_identity_missing",
                    "Selected scene identity was not produced; existing scene was retained unchanged.",
                    scene_id=scene_id,
                    scene_number=_scene_number(old_scene),
                ))
                merged.append(deepcopy(dict(old_scene)))
                continue
            merged.append(_merge_scene(replacement, old_scene, references.get(scene_id)))
            handled.add(scene_id)

        for scene_id, scene in generated_selected.items():
            if (
                scene_id in handled
                or scene_id in existing
                or _scene_number(scene) in occupied_scene_numbers
            ):
                continue
            merged.append(_merge_scene(scene, None, references.get(scene_id)))
        return merged


def _index_scenes(
    scenes: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for scene in scenes:
        scene_id, _segment_id = _canonical_identity(scene, label)
        if scene_id in result:
            raise FeverSlopDataError(f"duplicate {label} canonical scene_id: {scene_id}")
        result[scene_id] = scene
    return result


def _canonical_identity(scene: Mapping[str, Any], label: str) -> tuple[str, str]:
    canonical = scene.get("canonical")
    if not isinstance(canonical, Mapping):
        raise FeverSlopDataError(f"{label} scene is missing canonical identity")
    scene_id = canonical.get("scene_id")
    segment_id = canonical.get("segment_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise FeverSlopDataError(f"{label} scene is missing canonical identity scene_id")
    if not isinstance(segment_id, str) or not segment_id.strip():
        raise FeverSlopDataError(f"{label} scene is missing canonical identity segment_id")
    return scene_id, segment_id


def _merge_scene(
    generated: Mapping[str, Any],
    existing: Mapping[str, Any] | None,
    reference_entry: tuple[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    generated_id = _canonical_identity(generated, "generated")
    if existing is not None:
        existing_id = _canonical_identity(existing, "existing")
        if generated_id != existing_id:
            raise FeverSlopDataError(
                "canonical identity conflict: matching scene_id has a different segment_id",
            )
    if reference_entry is not None and reference_entry[0] != generated_id[1]:
        raise FeverSlopDataError(
            "reference identity conflict: matching scene_id has a different segment_id",
        )
    merged = deepcopy(dict(generated))
    canonical = merged["canonical"]
    roles = canonical.get("roles")
    if not isinstance(roles, dict):
        raise FeverSlopDataError("generated canonical roles must be an object")
    for role_name, role in roles.items():
        if not isinstance(role, dict):
            raise FeverSlopDataError(f"generated canonical role {role_name!r} must be an object")
        if "effective" in role:
            raise FeverSlopDataError(
                f"generated canonical role {role_name!r} must not persist effective",
            )

    if existing is not None:
        old_roles = existing["canonical"].get("roles")
        if not isinstance(old_roles, Mapping):
            raise FeverSlopDataError("existing canonical roles must be an object")
        for role_name, old_role in old_roles.items():
            if not isinstance(old_role, Mapping) or "override" not in old_role:
                continue
            target = roles.setdefault(str(role_name), {})
            if not isinstance(target, dict):
                raise FeverSlopDataError(f"generated canonical role {role_name!r} must be an object")
            target["override"] = deepcopy(old_role["override"])
        old_render_settings = existing.get("render_settings")
        if isinstance(old_render_settings, Mapping):
            merged["render_settings"] = deepcopy(dict(old_render_settings))
        old_references = existing.get("references")
        generator_fingerprint = (
            old_references.get("generator_fingerprint")
            if isinstance(old_references, Mapping)
            else None
        )
        if generator_fingerprint is not None:
            target_references = merged.setdefault("references", {})
            if not isinstance(target_references, dict):
                raise FeverSlopDataError("generated scene references must be an object")
            target_references["generator_fingerprint"] = deepcopy(generator_fingerprint)

    reference_bindings = reference_entry[1] if reference_entry is not None else None
    if reference_bindings:
        target_references = merged.setdefault("references", {})
        if not isinstance(target_references, dict):
            raise FeverSlopDataError("generated scene references must be an object")
        target_references.update(deepcopy(dict(reference_bindings)))
    return merged


def _shared_project_setting_fields(
    scenes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not scenes:
        return {}
    result: dict[str, Any] = {}
    render_settings = [scene.get("render_settings") for scene in scenes]
    if (
        all(isinstance(value, Mapping) for value in render_settings)
        and all(value == render_settings[0] for value in render_settings[1:])
    ):
        result["render_settings"] = deepcopy(dict(render_settings[0]))
    generators = [
        references.get("generator_fingerprint")
        if isinstance((references := scene.get("references")), Mapping)
        else None
        for scene in scenes
    ]
    if generators[0] is not None and all(value == generators[0] for value in generators[1:]):
        result["generator_fingerprint"] = deepcopy(generators[0])
    return result


def _apply_project_setting_fields(scene: dict[str, Any], fields: Mapping[str, Any]) -> None:
    render_settings = fields.get("render_settings")
    if render_settings is not None:
        scene.setdefault("render_settings", deepcopy(render_settings))
    generator = fields.get("generator_fingerprint")
    if generator is None:
        return
    references = scene.setdefault("references", {})
    if not isinstance(references, dict):
        raise FeverSlopDataError("generated scene references must be an object")
    references.setdefault("generator_fingerprint", deepcopy(generator))


def _reference_bindings(
    scenes: Sequence[Mapping[str, Any]],
    diagnostics: list[RegenerationDiagnostic],
) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for scene in scenes:
        try:
            scene_id, segment_id = _canonical_identity(scene, "reference")
        except FeverSlopDataError:
            diagnostics.append(RegenerationDiagnostic(
                "orphaned_reference_scene",
                "Reference-enriched scene has no usable canonical identity; bindings were not copied.",
                scene_number=_scene_number(scene),
            ))
            continue
        references = scene.get("references")
        if not isinstance(references, Mapping):
            continue
        if scene_id in result:
            raise FeverSlopDataError(f"duplicate reference canonical scene_id: {scene_id}")
        result[scene_id] = (
            segment_id,
            {
                key: deepcopy(value)
                for key, value in references.items()
                if key in _REFERENCE_BINDING_KEYS
            },
        )
    return result


def _orphan_diagnostic(scene_id: str, scene: Mapping[str, Any]) -> RegenerationDiagnostic:
    roles = scene["canonical"].get("roles")
    has_override = isinstance(roles, Mapping) and any(
        isinstance(role, Mapping) and "override" in role
        for role in roles.values()
    )
    code = "orphaned_override_scene" if has_override else "deleted_canonical_scene"
    return RegenerationDiagnostic(
        code,
        "Existing canonical scene was not regenerated; its identity was not reattached.",
        scene_id=scene_id,
        scene_number=_scene_number(scene),
    )


def _scene_number(scene: Mapping[str, Any]) -> int | None:
    value = scene.get("scene")
    return value if isinstance(value, int) else None
