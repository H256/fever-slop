from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from feverslop.domain.canonical_render_plan import (
    PromptRole,
    resolve_effective_role,
    validate_canonical_plan,
)
from feverslop.domain.artifact_hash import fingerprint_json
from feverslop.errors import FeverSlopDataError

PROJECTION_SCHEMA = "feverslop.canonical-projection/v1"
DEPENDENCY_SCHEMA = "feverslop.canonical-dependencies/v1"
CANONICAL_SOURCE = "output/render/plans/base.json"
_MISSING = object()

_OPERATIONAL_FIELDS = (
    "scene",
    "fps",
    "frame_count",
    "render_frame_count",
    "trim_front_frames",
    "width",
    "height",
    "duration",
    "duration_seconds",
    "seed",
    "render_settings",
)
_WORKFLOW_FIELDS = (
    "z_image",
    "ltx",
    "h3",
    "performance_timing",
    "keyframes",
)


@dataclass(frozen=True)
class CanonicalSceneDependencies:
    schema: str
    source: str
    source_revision: str
    scene_id: str
    workflow_fingerprint: str
    reference_fingerprint: str

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "source": self.source,
            "source_revision": self.source_revision,
            "scene_id": self.scene_id,
            "workflow_fingerprint": self.workflow_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalSceneDependencies:
        if payload.get("schema") != DEPENDENCY_SCHEMA:
            raise FeverSlopDataError(
                f"canonical dependencies schema must be {DEPENDENCY_SCHEMA!r}",
            )
        values = {
            field: str(payload.get(field) or "")
            for field in (
                "source",
                "source_revision",
                "scene_id",
                "workflow_fingerprint",
                "reference_fingerprint",
            )
        }
        for field, value in values.items():
            if not value:
                raise FeverSlopDataError(
                    f"canonical dependencies {field} must not be empty",
                )
        for field in ("source_revision", "workflow_fingerprint", "reference_fingerprint"):
            value = values[field]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise FeverSlopDataError(
                    f"canonical dependencies {field} must be a SHA-256 hex digest",
                )
        return cls(schema=DEPENDENCY_SCHEMA, **values)

_ROLE_PATHS: tuple[tuple[PromptRole, tuple[str, ...]], ...] = (
    (PromptRole.Z_IMAGE, ("z_image", "prompt")),
    (PromptRole.LTX_BASE, ("ltx", "base_prompt")),
    (PromptRole.LTX_I2V, ("ltx", "i2v_prompt_from_t2i")),
    (PromptRole.LTX_STATIC, ("ltx", "static_prompt")),
    (PromptRole.LTX_RELAY, ("ltx", "prompt_relay")),
    (PromptRole.LTX_MSR_GLOBAL, ("ltx", "msr_global_prompt")),
    (PromptRole.LTX_MSR_RELAY, ("ltx", "msr_prompt_relay")),
    (PromptRole.INGREDIENTS_GLOBAL, ("ingredients", "global_prompt")),
    (PromptRole.INGREDIENTS_RELAY, ("ltx", "prompt_relay")),
    (PromptRole.H3_VIDEO, ("h3", "prompt")),
    (PromptRole.PERFORMANCE_TIMING, ("performance_timing",)),
)


def canonical_plan_revision(scenes: Sequence[Mapping[str, Any]]) -> str:
    canonical_scenes = []
    for scene in scenes:
        canonical = scene.get("canonical")
        if isinstance(canonical, Mapping):
            authoritative = deepcopy(dict(scene))
            authoritative.pop("canonical_projection", None)
            canonical_scenes.append(authoritative)
    canonical_scenes.sort(
        key=lambda item: str((item.get("canonical") or {}).get("scene_id") or ""),
    )
    return fingerprint_json(canonical_scenes, ensure_ascii=False)


def canonical_scene_dependencies(
    scene: Mapping[str, Any],
    *,
    canonical_scene: Mapping[str, Any] | None = None,
    source_revision: str | None = None,
) -> CanonicalSceneDependencies:
    source = canonical_scene or scene
    canonical = source.get("canonical")
    if not isinstance(canonical, Mapping):
        raise FeverSlopDataError("canonical scene dependencies require canonical metadata")
    scene_id = str(canonical.get("scene_id") or "")
    if not scene_id:
        raise FeverSlopDataError("canonical scene dependencies require canonical.scene_id")

    workflow_payload: dict[str, Any] = {}
    for field in _OPERATIONAL_FIELDS:
        if field in source:
            workflow_payload[field] = deepcopy(source[field])
        elif field in scene:
            workflow_payload[field] = deepcopy(scene[field])
    for field in _WORKFLOW_FIELDS:
        if field in scene:
            workflow_payload[field] = deepcopy(scene[field])
    ingredients = scene.get("ingredients")
    if isinstance(ingredients, Mapping):
        prompt_fields = {
            str(key): deepcopy(value)
            for key, value in ingredients.items()
            if "prompt" in str(key) or "relay" in str(key)
        }
        if prompt_fields:
            workflow_payload["ingredients"] = prompt_fields

    reference_payload = _reference_dependency_payload(source.get("references"))
    revision = source_revision or canonical_plan_revision([source])
    return CanonicalSceneDependencies(
        schema=DEPENDENCY_SCHEMA,
        source=CANONICAL_SOURCE,
        source_revision=revision,
        scene_id=scene_id,
        workflow_fingerprint=_fingerprint(workflow_payload),
        reference_fingerprint=_fingerprint(reference_payload),
    )


def project_effective_plan(
    scenes: Sequence[Mapping[str, Any]],
    canonical_scenes: Sequence[Mapping[str, Any]] | None = None,
    *,
    source_revision: str | None = None,
) -> list[dict[str, Any]]:
    canonical_input = list(canonical_scenes or ())
    validate_canonical_plan(canonical_input)
    by_scene_id = _canonical_index(canonical_input)
    revision = source_revision or canonical_plan_revision(canonical_input or scenes)
    projected = []
    for scene in scenes:
        source = _matching_canonical_scene(scene, by_scene_id)
        projected.append(
            project_effective_scene(
                scene,
                canonical_scene=source,
                source_revision=revision,
            ),
        )
    return projected


def project_effective_scene(
    scene: Mapping[str, Any],
    *,
    canonical_scene: Mapping[str, Any] | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    projected = deepcopy(dict(scene))
    source = canonical_scene or scene
    canonical = source.get("canonical")
    if not isinstance(canonical, Mapping):
        return projected

    projected["canonical"] = deepcopy(dict(canonical))
    for field in _OPERATIONAL_FIELDS:
        if field in source:
            projected[field] = deepcopy(source[field])
    source_references = source.get("references")
    if isinstance(source_references, Mapping):
        existing_references = projected.get("references")
        merged_references = (
            deepcopy(dict(existing_references))
            if isinstance(existing_references, Mapping)
            else {}
        )
        for key in list(merged_references):
            if not _is_derived_reference_key(key):
                merged_references.pop(key)
        merged_references.update(
            deepcopy(dict(_reference_dependency_payload(source_references))),
        )
        projected["references"] = merged_references
    for role, path in _ROLE_PATHS:
        if not _has_role(source, role):
            continue
        legacy_value = _get(projected, path)
        effective = (
            resolve_effective_role(source, role)
            if legacy_value is _MISSING
            else resolve_effective_role(
                source,
                role,
                legacy_value=legacy_value,
            )
        )
        _set(projected, path, effective)
        if role is PromptRole.LTX_I2V:
            _set(projected, ("ltx", "original_style_i2v_prompt"), effective)

    existing_projection = scene.get("canonical_projection")
    existing_revision = (
        str(existing_projection.get("source_revision") or "")
        if isinstance(existing_projection, Mapping)
        else ""
    )
    revision = source_revision or existing_revision or canonical_plan_revision([source])
    projected["canonical_projection"] = {
        "schema": PROJECTION_SCHEMA,
        "scene_id": str(canonical["scene_id"]),
        "source": CANONICAL_SOURCE,
        "source_revision": revision,
        "dependencies": canonical_scene_dependencies(
            projected,
            canonical_scene=source,
            source_revision=revision,
        ).to_dict(),
    }
    return projected


def _canonical_index(scenes: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for scene in scenes:
        canonical = scene.get("canonical")
        if not isinstance(canonical, Mapping):
            continue
        scene_id = str(canonical["scene_id"])
        if scene_id in result:
            raise FeverSlopDataError(f"duplicate canonical scene_id: {scene_id}")
        result[scene_id] = scene
    return result


def _matching_canonical_scene(
    scene: Mapping[str, Any],
    by_scene_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not by_scene_id:
        return None
    canonical = scene.get("canonical")
    if not isinstance(canonical, Mapping):
        return None
    scene_id = str(canonical.get("scene_id") or "")
    match = by_scene_id.get(scene_id)
    if match is None:
        raise FeverSlopDataError(
            f"derived canonical scene_id has no match in base plan: {scene_id or '<empty>'}",
        )
    return match


def _has_role(scene: Mapping[str, Any], role: PromptRole) -> bool:
    canonical = scene.get("canonical")
    roles = canonical.get("roles") if isinstance(canonical, Mapping) else None
    return isinstance(roles, Mapping) and str(role) in roles


def _get(source: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _set(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = deepcopy(value)


def _reference_dependency_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _reference_dependency_payload(item)
            for key, item in value.items()
            if not _is_derived_reference_key(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_reference_dependency_payload(item) for item in value]
    return deepcopy(value)


def _is_derived_reference_key(key: Any) -> bool:
    normalized = str(key).lower()
    return any(
        marker in normalized
        for marker in ("path", "sha", "sheet", "anchor")
    )


def _fingerprint(value: Any) -> str:
    return fingerprint_json(value, ensure_ascii=False)
