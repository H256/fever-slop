"""Versioned, language-neutral contracts for preserving explicit user intent.

This module defines the *semantic intent ledger* (issue #543): generic,
versioned domain models that capture who/what a project is about -- people,
creatures, objects, groups, locations, and abstract narrative entities --
independently of any fixed singer/band ontology, music role, or human-cast
assumption.

Scope (per #543):
- Stable entity and constraint IDs.
- Free role labels and generic entity kinds (people, creatures, objects,
  groups, locations, abstract narrative entities).
- Identity attributes, relations, cardinality, recurrence, scene obligations,
  and required/optional status.
- Provenance linking each explicit constraint to source text and language.
- Persistence in project artifacts/config with schema versioning.
- A compatibility representation for current actor/location dictionaries.

Non-goals (per #543):
- Fixed singer/band ontology.
- DSPy extraction or judging.
- Downstream render enforcement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

#: Version of the ledger serialization contract.
SCHEMA_VERSION = "semantic-intent/v1"

#: Supported generic entity kinds. Role labels stay free; kinds are the only
#: constrained taxonomy so non-human leads (creatures, objects, groups) are
#: representable without a fixed singer/band ontology.
ENTITY_KINDS = frozenset({"person", "creature", "object", "group", "location", "abstract"})

#: How often an entity recurs across the project.
RECURRENCES = frozenset({"once", "recurring", "always"})

#: Whether an entity/relation/constraint is required or optional.
OBLIGATION_STATUS = frozenset({"required", "optional"})

#: The kind of a cross-cutting constraint (scene obligation, etc.).
CONSTRAINT_KINDS = frozenset({"identity", "relation", "cardinality", "recurrence", "scene_obligation"})

#: Whether a constraint was stated explicitly by the user or inferred.
PROVENANCE_ORIGINS = frozenset({"explicit", "inferred"})

_MAX_ID = 128


def _validate_id(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if len(normalized) > _MAX_ID:
        raise ValueError(f"{field_name} exceeds {_MAX_ID} characters")
    return normalized


class ConstraintProvenance(BaseModel):
    """Links a constraint to its origin: explicit source text or inference."""

    origin: str = Field(description="explicit or inferred")
    source_text: str = Field(default="", max_length=2000)
    language: str = Field(default="", max_length=16)
    source_ref: str = Field(default="", max_length=_MAX_ID)

    @model_validator(mode="after")
    def _check(self) -> "ConstraintProvenance":
        if self.origin not in PROVENANCE_ORIGINS:
            raise ValueError(f"origin must be one of {sorted(PROVENANCE_ORIGINS)}")
        if self.origin == "explicit" and not self.source_text.strip():
            raise ValueError("explicit provenance requires non-empty source_text")
        return self


class IntentEntity(BaseModel):
    """A generic project entity: person, creature, object, group, location, or abstract."""

    id: str
    kind: str
    role: str = Field(default="", max_length=256, description="Free role label; no fixed ontology.")
    identity: dict[str, str] = Field(default_factory=dict, description="Identity attributes (free keys).")
    recurrence: str = "once"
    status: str = "required"
    description: str = Field(default="", max_length=2000)

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "entity id")

    @model_validator(mode="after")
    def _check(self) -> "IntentEntity":
        if self.kind not in ENTITY_KINDS:
            raise ValueError(f"kind must be one of {sorted(ENTITY_KINDS)}")
        if self.recurrence not in RECURRENCES:
            raise ValueError(f"recurrence must be one of {sorted(RECURRENCES)}")
        if self.status not in OBLIGATION_STATUS:
            raise ValueError(f"status must be one of {sorted(OBLIGATION_STATUS)}")
        return self


class IntentRelation(BaseModel):
    """A directed relation between two entities, with a free label and cardinality."""

    id: str
    subject_id: str
    relation: str = Field(description="Free relation label.")
    target_id: str
    cardinality: int = Field(default=1, ge=1)
    status: str = "required"
    provenance: ConstraintProvenance | None = None

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "relation id")

    @model_validator(mode="after")
    def _check(self) -> "IntentRelation":
        if self.status not in OBLIGATION_STATUS:
            raise ValueError(f"status must be one of {sorted(OBLIGATION_STATUS)}")
        if not self.relation.strip():
            raise ValueError("relation label is required")
        return self


class IntentConstraint(BaseModel):
    """A cross-cutting constraint (scene obligation, identity, etc.) on an entity."""

    id: str
    entity_id: str
    kind: str
    statement: str = Field(max_length=1000)
    status: str = "required"
    provenance: ConstraintProvenance | None = None

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "constraint id")

    @model_validator(mode="after")
    def _check(self) -> "IntentConstraint":
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"kind must be one of {sorted(CONSTRAINT_KINDS)}")
        if self.status not in OBLIGATION_STATUS:
            raise ValueError(f"status must be one of {sorted(OBLIGATION_STATUS)}")
        if not self.statement.strip():
            raise ValueError("statement is required")
        return self


class IntentLedger(BaseModel):
    """A versioned, self-consistent ledger of entities, relations, and constraints."""

    schema_version: str = SCHEMA_VERSION
    entities: list[IntentEntity] = Field(default_factory=list)
    relations: list[IntentRelation] = Field(default_factory=list)
    constraints: list[IntentConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "IntentLedger":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported semantic intent schema: {self.schema_version}")
        entity_ids = _unique_ids([e.id for e in self.entities], "entity")
        for relation in self.relations:
            _unique_ids([relation.id], "relation")
            _require(entity_ids, relation.subject_id, f"relation {relation.id} subject")
            _require(entity_ids, relation.target_id, f"relation {relation.id} target")
        for constraint in self.constraints:
            _unique_ids([constraint.id], "constraint")
            _require(entity_ids, constraint.entity_id, f"constraint {constraint.id} entity")
        return self

    def entity_ids(self) -> frozenset[str]:
        return frozenset(entity.id for entity in self.entities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entities": [entity.model_dump() for entity in self.entities],
            "relations": [relation.model_dump() for relation in self.relations],
            "constraints": [constraint.model_dump() for constraint in self.constraints],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntentLedger":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            entities=[IntentEntity(**item) for item in payload.get("entities") or []],
            relations=[IntentRelation(**item) for item in payload.get("relations") or []],
            constraints=[IntentConstraint(**item) for item in payload.get("constraints") or []],
        )


def _unique_ids(ids: list[str], kind: str) -> set[str]:
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise ValueError(f"duplicate {kind} id: {value}")
        seen.add(value)
    return seen


def _require(known: set[str], value: str, label: str) -> None:
    if value not in known:
        raise ValueError(f"{label} references unknown entity: {value}")


def ledger_from_legacy_cast(
    actors: list[dict[str, Any]] | None,
    locations: list[dict[str, Any]] | None = None,
) -> IntentLedger:
    """Compatibility path: map current actor/location dictionaries to a ledger.

    Legacy project configs carry ``actors`` (id/name/role/visual_description)
    and ``locations``/``structured_locations`` (id/name/visual_description).
    This converts them into generic ledger entities without assuming a singer
    or band ontology: actors become ``person`` entities with a free role label,
    and locations become ``location`` entities. The result is a documented,
    lossless bridge so legacy projects can adopt the ledger without a fixed
    singer/band model.

    Args:
        actors: List of actor dicts (keys: id, name, role, visual_description).
        locations: Optional list of location dicts (keys: id, name, visual_description).

    Returns:
        An IntentLedger with one entity per actor/location.
    """
    entities: list[IntentEntity] = []
    for actor in actors or []:
        actor_id = str(actor.get("id", "")).strip()
        if not actor_id:
            continue
        entities.append(
            IntentEntity(
                id=actor_id,
                kind="person",
                role=str(actor.get("role", "")).strip(),
                identity={"name": str(actor.get("name", "")).strip()} if actor.get("name") else {},
                description=str(actor.get("visual_description", "")).strip(),
            )
        )
    for location in locations or []:
        location_id = str(location.get("id", "")).strip()
        if not location_id:
            continue
        entities.append(
            IntentEntity(
                id=location_id,
                kind="location",
                role=str(location.get("name", "")).strip(),
                description=str(location.get("visual_description", "")).strip(),
            )
        )
    return IntentLedger(entities=entities)


def entity_by_id(ledger: IntentLedger, entity_id: str) -> IntentEntity | None:
    """Return the entity with the given id, or None when absent.

    Stable-ID lookup is the foundation of provenance: a downstream artifact
    can name the exact entity it retained without re-deriving identity from
    free text.
    """
    for entity in ledger.entities:
        if entity.id == entity_id:
            return entity
    return None


def entity_constraint_ids(ledger: IntentLedger, entity_id: str) -> list[str]:
    """Return the constraint IDs bound to an entity, in ledger order.

    This is the retained-constraint set a reference/scene artifact must carry
    so a pipeline boundary can prove what it kept. A targeted repair that
    drops one of these IDs (or binds another entity's constraint) is detectable
    by comparing against this list.
    """
    return [
        constraint.id
        for constraint in ledger.constraints
        if constraint.entity_id == entity_id
    ]


def ledger_for_project(
    project_dir: str | Path,
    song_id: str,
    actors: list[dict[str, Any]] | None = None,
    locations: list[dict[str, Any]] | None = None,
) -> IntentLedger:
    """Resolve the semantic intent ledger for a project.

    Prefers a persisted ledger artifact at
    ``<project_dir>/output/prompts/semantic_intent_<song_id>.json`` (written by
    the extraction stage). Falls back to the legacy cast/location compatibility
    path so existing projects adopt the ledger without a fixed singer/band
    ontology. A missing or unreadable artifact degrades to the legacy path
    rather than failing the pipeline.
    """
    base = Path(project_dir)
    candidate = base / "output" / "prompts" / f"semantic_intent_{song_id}.json"
    if candidate.exists():
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
            return IntentLedger.from_dict(payload)
        except (ValueError, OSError):
            pass
    return ledger_from_legacy_cast(actors, locations)
