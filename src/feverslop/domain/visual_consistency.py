from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any

SCHEMA = "feverslop.visual-consistency/v1"
_MODES = {"ingredients", "msr", "i2v"}
_HANDOFF_MODES = {"msr", "i2v"}
_KINDS = {"actor", "location"}
_TRANSITIONS = {"cut", "continuous"}
_SEVERITIES = {"warning", "error"}


class PreflightMode(StrEnum):
    STRICT = "strict"
    WARN = "warn"
    OFF = "off"

    @classmethod
    def parse(cls, value: PreflightMode | str) -> PreflightMode:
        try:
            return value if isinstance(value, cls) else cls(str(value).strip().lower())
        except ValueError:
            raise ValueError("preflight mode must be strict, warn, or off") from None


def _required(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")


def _validate_sha256(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{field} must be a lowercase SHA-256, 64-character hexadecimal hash"
        )


def _validate_scene(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("scene must be a positive integer")


@dataclass(frozen=True)
class ReferenceAnchor:
    id: str
    kind: str
    look_id: str
    asset_role: str
    asset_sha256: str
    prompt_anchor: str

    def __post_init__(self) -> None:
        _required(self.id, "id")
        if self.kind not in _KINDS:
            raise ValueError("kind must be actor or location")
        _required(self.look_id, "look id")
        _required(self.asset_role, "asset role")
        _validate_sha256(self.asset_sha256, "asset sha256")
        _required(self.prompt_anchor, "prompt anchor")

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "kind": self.kind,
            "look_id": self.look_id,
            "asset_role": self.asset_role,
            "asset_sha256": self.asset_sha256,
            "prompt_anchor": self.prompt_anchor,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceAnchor:
        return cls(
            id=payload["id"],
            kind=payload["kind"],
            look_id=payload["look_id"],
            asset_role=payload["asset_role"],
            asset_sha256=payload["asset_sha256"],
            prompt_anchor=payload["prompt_anchor"],
        )


@dataclass(frozen=True)
class ConsistencyIssue:
    code: str
    scene: int
    severity: str
    message: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (self.code, self.message)):
            raise ValueError("code, scene, and message are required")
        _validate_scene(self.scene)
        if self.severity not in _SEVERITIES:
            raise ValueError("severity must be warning or error")


@dataclass(frozen=True)
class SceneConsistencyContract:
    schema: str
    scene: int
    mode: str
    workflow_profile: str
    actors: tuple[ReferenceAnchor, ...]
    location: ReferenceAnchor | None
    transition_from_previous: str
    fingerprint: str

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        _validate_scene(self.scene)
        if self.mode not in _MODES:
            raise ValueError("mode must be ingredients, msr, or i2v")
        _required(self.workflow_profile, "workflow profile")
        if not isinstance(self.actors, tuple):
            object.__setattr__(self, "actors", tuple(self.actors))
        seen_actor_ids: set[str] = set()
        for anchor in self.actors:
            if not isinstance(anchor, ReferenceAnchor) or anchor.kind != "actor":
                raise ValueError("actors must have kind actor")
            if anchor.id in seen_actor_ids:
                raise ValueError(f"duplicate actor id: {anchor.id}")
            seen_actor_ids.add(anchor.id)
        if self.location is not None and (
            not isinstance(self.location, ReferenceAnchor)
            or self.location.kind != "location"
        ):
            raise ValueError("location must have kind location")
        if self.transition_from_previous not in _TRANSITIONS:
            raise ValueError("transition must be cut or continuous")
        _validate_sha256(self.fingerprint, "fingerprint")
        expected_fingerprint = _fingerprint(
            _canonical_payload(
                scene=self.scene,
                mode=self.mode,
                workflow_profile=self.workflow_profile,
                actors=self.actors,
                location=self.location,
                transition_from_previous=self.transition_from_previous,
            )
        )
        if self.fingerprint != expected_fingerprint:
            raise ValueError("fingerprint does not match canonical payload")

    @classmethod
    def create(
        cls,
        *,
        scene: int,
        mode: str,
        workflow_profile: str,
        actors: tuple[ReferenceAnchor, ...],
        location: ReferenceAnchor | None,
        transition_from_previous: str,
    ) -> SceneConsistencyContract:
        _validate_scene(scene)
        resolved_actors = tuple(actors)
        payload = _canonical_payload(
            scene=scene,
            mode=mode,
            workflow_profile=workflow_profile,
            actors=resolved_actors,
            location=location,
            transition_from_previous=transition_from_previous,
        )
        fingerprint = _fingerprint(payload)
        return cls(
            schema=SCHEMA,
            scene=scene,
            mode=mode,
            workflow_profile=workflow_profile,
            actors=resolved_actors,
            location=location,
            transition_from_previous=transition_from_previous,
            fingerprint=fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **_canonical_payload(
                scene=self.scene,
                mode=self.mode,
                workflow_profile=self.workflow_profile,
                actors=self.actors,
                location=self.location,
                transition_from_previous=self.transition_from_previous,
            ),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneConsistencyContract:
        if payload["schema"] != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        restored = cls.create(
            scene=payload["scene"],
            mode=payload["mode"],
            workflow_profile=payload["workflow_profile"],
            actors=tuple(
                ReferenceAnchor.from_dict(anchor) for anchor in payload["actors"]
            ),
            location=(
                ReferenceAnchor.from_dict(payload["location"])
                if payload["location"] is not None
                else None
            ),
            transition_from_previous=payload["transition_from_previous"],
        )
        if payload["fingerprint"] != restored.fingerprint:
            raise ValueError("fingerprint does not match canonical payload")
        return restored

    def prompt_anchor_text(self, max_chars: int = 700) -> str:
        if type(max_chars) is not int or max_chars < 0:
            raise ValueError("max_chars must be a non-negative integer")
        anchors = (*self.actors, *(() if self.location is None else (self.location,)))
        return "\n".join(anchor.prompt_anchor for anchor in anchors)[:max_chars]


def _canonical_payload(
    *,
    scene: int,
    mode: str,
    workflow_profile: str,
    actors: tuple[ReferenceAnchor, ...],
    location: ReferenceAnchor | None,
    transition_from_previous: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scene": scene,
        "mode": mode,
        "workflow_profile": workflow_profile,
        "actors": [anchor.to_dict() for anchor in actors],
        "location": location.to_dict() if location is not None else None,
        "transition_from_previous": transition_from_previous,
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def can_handoff(
    previous: SceneConsistencyContract,
    current: SceneConsistencyContract,
) -> bool:
    if current.transition_from_previous != "continuous":
        return False
    if previous.mode not in _HANDOFF_MODES or current.mode not in _HANDOFF_MODES:
        return False
    if previous.location is None or current.location is None:
        return False
    if previous.location.id != current.location.id:
        return False
    previous_actor_ids = {anchor.id for anchor in previous.actors}
    current_actor_ids = {anchor.id for anchor in current.actors}
    return bool(previous_actor_ids & current_actor_ids)


def validate_scene_sequence(
    scenes: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate that scene numbers are positive integers in consecutive order.

    Consecutive order is relative to the first scene number (e.g., [5, 6, 7] is valid).
    Does not enforce a specific start number (1-based or otherwise).
    """
    items = tuple(scenes)
    numbers = [scene.get("scene") for scene in items]
    if any(type(number) is not int or number <= 0 for number in numbers):
        raise ValueError(
            "Scenes must use positive integers in consecutive order"
        )
    if numbers and numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError(
            "Scenes must be in consecutive order without duplicates or gaps"
        )
    return items


def expand_handoff_selection(
    contracts: Iterable[SceneConsistencyContract],
    selected: set[int],
) -> set[int]:
    expanded = set(selected)
    items = tuple(contracts)
    for previous, current in zip(items, items[1:]):
        if (
            previous.scene in expanded
            and previous.scene + 1 == current.scene
            and can_handoff(previous, current)
        ):
            expanded.add(current.scene)
    return expanded
