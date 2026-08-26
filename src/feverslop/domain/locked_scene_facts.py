from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LockedFact:
    category: str
    key: str
    value: str
    source_id: str
    provenance: str = "canonical"

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "source_id": self.source_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class LockedSceneFacts:
    """Immutable canonical facts that a creative model may not rewrite."""

    scene_id: str
    facts: tuple[LockedFact, ...]

    @classmethod
    def create(cls, *, scene_id: str, facts: Any) -> "LockedSceneFacts":
        resolved_scene_id = str(scene_id).strip()
        if not resolved_scene_id:
            raise ValueError("scene_id is required")
        try:
            raw_facts = tuple(facts)
        except TypeError as exc:
            raise ValueError("facts must be iterable") from exc
        normalized: list[LockedFact] = []
        for raw in raw_facts:
            if not isinstance(raw, Mapping):
                raise ValueError("each locked fact must be an object")
            values = {
                field: str(raw.get(field) or "").strip()
                for field in ("category", "key", "value", "source_id")
            }
            if any(not value for value in values.values()):
                raise ValueError("locked facts require category, key, value, and source_id")
            normalized.append(
                LockedFact(
                    category=values["category"].lower(),
                    key=values["key"].lower(),
                    value=values["value"],
                    source_id=values["source_id"],
                    provenance=str(raw.get("provenance") or "canonical").strip() or "canonical",
                )
            )
        normalized.sort(key=lambda fact: (fact.category.lower(), fact.key.lower(), fact.source_id.lower()))
        by_identity: dict[tuple[str, str], LockedFact] = {}
        for fact in normalized:
            identity = (fact.category.casefold(), fact.key.casefold())
            previous = by_identity.get(identity)
            if previous is not None and previous.value.casefold() != fact.value.casefold():
                raise ValueError(
                    f"contradictory locked fact {fact.category}/{fact.key} "
                    f"from {previous.source_id} and {fact.source_id}"
                )
            by_identity.setdefault(identity, fact)
        return cls(scene_id=resolved_scene_id, facts=tuple(by_identity.values()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LockedSceneFacts":
        if not isinstance(payload, Mapping):
            raise ValueError("locked scene facts must be an object")
        return cls.create(scene_id=payload.get("scene_id", ""), facts=payload.get("facts", []))

    def to_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "facts": [fact.to_dict() for fact in self.facts]}
