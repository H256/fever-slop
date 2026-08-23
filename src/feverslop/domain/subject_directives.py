from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "subject-directives/v1"
PROP_STATES = frozenset({"held", "played", "attached", "placed", "absent"})


@dataclass(frozen=True)
class TemporalScope:
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("temporal scope requires end_seconds > start_seconds >= 0")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalScope":
        return cls(float(payload.get("start_seconds", 0)), float(payload.get("end_seconds", 0)))

    def to_dict(self) -> dict[str, float]:
        return {"start_seconds": self.start_seconds, "end_seconds": self.end_seconds}


@dataclass(frozen=True)
class PropBinding:
    prop_id: str
    state: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.prop_id.strip():
            raise ValueError("prop binding prop_id is required")
        if self.state not in PROP_STATES:
            raise ValueError(f"unsupported prop state: {self.state}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropBinding":
        return cls(str(payload.get("prop_id", "")).strip(), str(payload.get("state", "")).strip(), str(payload.get("detail", "")).strip())

    def to_dict(self) -> dict[str, str]:
        result = {"prop_id": self.prop_id, "state": self.state}
        if self.detail:
            result["detail"] = self.detail
        return result


@dataclass(frozen=True)
class SpatialRelation:
    subject_id: str
    relation: str
    target_id: str
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialRelation":
        return cls(*(str(payload.get(key, "")).strip() for key in ("subject_id", "relation", "target_id", "detail")))

    def to_dict(self) -> dict[str, str]:
        result = {"subject_id": self.subject_id, "relation": self.relation, "target_id": self.target_id}
        if self.detail:
            result["detail"] = self.detail
        return result


@dataclass(frozen=True)
class SubjectDirective:
    subject_id: str
    role: str
    position: str
    action: str
    interactions: tuple[str, ...] = ()
    prop_bindings: tuple[PropBinding, ...] = ()
    visibility: str = "visible"
    cardinality: int = 1
    temporal_scope: TemporalScope | None = None
    gaze_direction: str = ""

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject directive subject_id is required")
        if self.cardinality < 1:
            raise ValueError("subject directive cardinality must be positive")
        object.__setattr__(self, "interactions", tuple(str(item).strip() for item in self.interactions if str(item).strip()))
        object.__setattr__(self, "prop_bindings", tuple(self.prop_bindings))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubjectDirective":
        scope = payload.get("temporal_scope")
        return cls(
            subject_id=str(payload.get("subject_id", "")).strip(),
            role=str(payload.get("role", "")).strip(),
            position=str(payload.get("position", "")).strip(),
            action=str(payload.get("action", "")).strip(),
            interactions=tuple(payload.get("interactions") or ()),
            gaze_direction=str(payload.get("gaze_direction", "")).strip(),
            prop_bindings=tuple(PropBinding.from_dict(item) for item in payload.get("prop_bindings") or ()),
            visibility=str(payload.get("visibility", "visible")).strip(),
            cardinality=int(payload.get("cardinality", 1)),
            temporal_scope=TemporalScope.from_dict(scope) if scope else None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "subject_id": self.subject_id,
            "role": self.role,
            "position": self.position,
            "action": self.action,
            "prop_bindings": [item.to_dict() for item in self.prop_bindings],
            "visibility": self.visibility,
            "cardinality": self.cardinality,
        }
        if self.interactions:
            result["interactions"] = list(self.interactions)
        if self.gaze_direction:
            result["gaze_direction"] = self.gaze_direction
        if self.temporal_scope:
            result["temporal_scope"] = self.temporal_scope.to_dict()
        return result


@dataclass(frozen=True)
class SubjectDirectivePlan:
    shot_id: str
    temporal_scope: TemporalScope
    subjects: tuple[SubjectDirective, ...] = ()
    spatial_relations: tuple[SpatialRelation, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported subject directive schema: {self.schema_version}")
        if not self.shot_id.strip():
            raise ValueError("subject directive plan shot_id is required")
        object.__setattr__(self, "subjects", tuple(self.subjects))
        object.__setattr__(self, "spatial_relations", tuple(self.spatial_relations))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubjectDirectivePlan":
        return cls(
            shot_id=str(payload.get("shot_id", "")).strip(),
            temporal_scope=TemporalScope.from_dict(payload.get("temporal_scope") or {}),
            subjects=tuple(SubjectDirective.from_dict(item) for item in payload.get("subjects") or ()),
            spatial_relations=tuple(SpatialRelation.from_dict(item) for item in payload.get("spatial_relations") or ()),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shot_id": self.shot_id,
            "temporal_scope": self.temporal_scope.to_dict(),
            "subjects": [item.to_dict() for item in self.subjects],
            **({"spatial_relations": [item.to_dict() for item in self.spatial_relations]} if self.spatial_relations else {}),
        }


def validate_subject_directive_plan(
    plan: SubjectDirectivePlan,
    *,
    known_subject_ids: Iterable[str] = (),
    known_environment_ids: Iterable[str] = (),
    known_prop_ids: Iterable[str] = (),
) -> list[str]:
    issues: list[str] = []
    seen: set[str] = set()
    for subject in plan.subjects:
        if subject.subject_id in seen:
            issues.append(f"duplicate subject ID: {subject.subject_id}")
        seen.add(subject.subject_id)
        if subject.visibility == "visible" and (not subject.position or not subject.action):
            issues.append(f"subject {subject.subject_id} needs position and action")
        scope = subject.temporal_scope
        if scope is None or scope.start_seconds > plan.temporal_scope.start_seconds or scope.end_seconds < plan.temporal_scope.end_seconds:
            issues.append(f"subject {subject.subject_id} has incomplete temporal coverage")
    relation_values: dict[tuple[str, str, str], str] = {}
    for relation in plan.spatial_relations:
        key = (relation.subject_id, relation.relation, relation.target_id)
        previous = relation_values.setdefault(key, relation.detail)
        if previous != relation.detail:
            issues.append(
                "contradictory spatial relation: "
                f"{relation.subject_id} {relation.relation} {relation.target_id}"
            )
    return issues
