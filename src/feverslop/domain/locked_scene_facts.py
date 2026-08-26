from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from feverslop.domain.canonical_render_plan import PromptRole, resolve_effective_role
from feverslop.domain.subject_directives import SubjectDirectivePlan, validate_subject_directive_plan


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


def locked_scene_facts_from_scene(scene: Mapping[str, Any]) -> LockedSceneFacts:
    """Project effective scene inputs into an immutable prompt contract.

    The projector reads canonical values after override resolution and keeps
    creative prose out of the fact set. Legacy ``locked_facts`` remain a
    compatibility input, but are checked together with projected values so
    contradictions fail before a model call.
    """
    if not isinstance(scene, Mapping):
        raise ValueError("scene must be an object")
    canonical = scene.get("canonical")
    scene_id = (
        canonical.get("scene_id")
        if isinstance(canonical, Mapping)
        else scene.get("segment_id") or scene.get("scene_id") or scene.get("scene")
    )
    facts: list[dict[str, str]] = []

    def add(category: str, key: str, value: Any, source_id: str, provenance: str = "canonical") -> None:
        text = _fact_value(value)
        if text:
            facts.append({
                "category": category,
                "key": key,
                "value": text,
                "source_id": source_id,
                "provenance": provenance,
            })

    for raw in scene.get("locked_facts") or ():
        if not isinstance(raw, Mapping):
            raise ValueError("each locked fact must be an object")
        facts.append({
            field: str(raw.get(field) or "").strip()
            for field in ("category", "key", "value", "source_id")
        } | {"provenance": str(raw.get("provenance") or "legacy").strip() or "legacy"})

    references = scene.get("references")
    if isinstance(references, Mapping):
        for actor_id in references.get("actor_ids") or ():
            actor = str(actor_id).strip()
            add("cast", actor, actor, f"references:actor:{actor}")
        for item in references.get("actor_reference_descriptions") or ():
            if isinstance(item, Mapping):
                actor = str(item.get("id") or item.get("name") or "").strip()
                description = item.get("description") or item.get("details")
                if actor:
                    add("identity", actor, description, f"references:actor-description:{actor}")
        add("location", "id", references.get("location_id"), "references:location")
        add("location", "description", references.get("location_reference_description"), "references:location-description")
        for prop_id in references.get("prop_ids") or ():
            prop = str(prop_id).strip()
            add("props", prop, prop, f"references:prop:{prop}")
        for subject_id, bindings in _mapping_items(references.get("prop_bindings")):
            add("props", subject_id, bindings, f"references:prop-bindings:{subject_id}")
        for stem, binding in _mapping_items(references.get("audio_subject_bindings")):
            add("audio_binding", stem, binding, f"references:audio-binding:{stem}")

    directive_payload = scene.get("subject_directives")
    if isinstance(directive_payload, SubjectDirectivePlan):
        directive_plan = directive_payload
    elif isinstance(directive_payload, Mapping):
        directive_plan = SubjectDirectivePlan.from_dict(directive_payload)
    else:
        directive_plan = None
    if directive_plan is not None:
        issues = validate_subject_directive_plan(directive_plan)
        if issues:
            raise ValueError("Invalid subject directives: " + "; ".join(issues))
        for subject in directive_plan.subjects:
            prefix = f"directive:{directive_plan.shot_id}:{subject.subject_id}"
            add("directive", f"{subject.subject_id}.role", subject.role, f"{prefix}:role", "directive")
            add("directive", f"{subject.subject_id}.position", subject.position, f"{prefix}:position", "directive")
            add("directive", f"{subject.subject_id}.action", subject.action, f"{prefix}:action", "directive")
            add("directive", f"{subject.subject_id}.props", [binding.to_dict() for binding in subject.prop_bindings], f"{prefix}:props", "directive")
        for index, relation in enumerate(directive_plan.spatial_relations):
            add("spatial", str(index), relation.to_dict(), f"directive:{directive_plan.shot_id}:relation:{index}", "directive")
        add("timing", "directive", directive_plan.temporal_scope.to_dict(), f"directive:{directive_plan.shot_id}:timing", "directive")

    timing = _effective_role(scene, PromptRole.PERFORMANCE_TIMING, scene.get("performance_timing"))
    add("timing", "performance", timing, "canonical.roles.performance.timing")
    add("timing", "duration", scene.get("duration_seconds") or scene.get("duration"), "scene:duration")
    return LockedSceneFacts.create(scene_id=str(scene_id or ""), facts=facts)


def _effective_role(scene: Mapping[str, Any], role: PromptRole, fallback: Any) -> Any:
    canonical = scene.get("canonical")
    roles = canonical.get("roles") if isinstance(canonical, Mapping) else None
    if isinstance(roles, Mapping) and str(role) in roles:
        return resolve_effective_role(scene, role, legacy_value=fallback)
    return fallback


def _mapping_items(value: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        return tuple((str(key).strip(), item) for key, item in value.items() if str(key).strip())
    if isinstance(value, list):
        return tuple(
            (str(item.get("stem") or "").strip(), item)
            for item in value
            if isinstance(item, Mapping) and str(item.get("stem") or "").strip()
        )
    return ()


def _fact_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).strip()
