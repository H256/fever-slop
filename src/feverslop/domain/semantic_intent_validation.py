"""Deterministic validation and bounded legacy migration for the intent ledger.

Implements issue #546 on top of the versioned ledger contract from #543
(``feverslop.domain.semantic_intent``).

Design notes:
- ``validate_intent_ledger`` operates on the *raw persisted dict*, not on a
  constructed ``IntentLedger``. That lets it report findings for malformed or
  dangling records by stable ID instead of raising, which is what a
  deterministic, repeatable validator needs.
- Findings carry a stable ``code`` and ``subject_id`` so callers can report
  missing/contradictory required constraints without dumping full prompts.
- Policies (strict / warn / compat) are applied in ``blocking_findings`` so the
  validator itself stays policy-free and repeatable.
- ``migrate_ledger_payload`` is bounded and reversible: supported v1 payloads
  pass through untouched; legacy/v0 payloads are converted through the lossless
  cast bridge with the original preserved under ``migrated_from``.

Non-goals (per #546): prose reinterpretation, replacing DSPy extraction, and
downstream constraint propagation (#547).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from feverslop.domain.semantic_intent import (
    SCHEMA_VERSION,
    ledger_from_legacy_cast,
)

#: Validation policy levels.
MODES = frozenset({"strict", "warn", "compat"})

#: Severity levels for a finding.
SEVERITIES = frozenset({"error", "warning"})


@dataclass(frozen=True)
class IntentFinding:
    """A single, stable, actionable validation finding."""

    code: str
    severity: str
    subject_id: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "subject_id": self.subject_id,
            "message": self.message,
        }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _record_id(record: Any, fallback: str) -> str:
    if isinstance(record, dict):
        value = record.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _entity_ids(entities: list[Any]) -> set[str]:
    ids: set[str] = set()
    for index, entity in enumerate(entities):
        if isinstance(entity, dict):
            value = entity.get("id")
            if isinstance(value, str) and value.strip():
                ids.add(value.strip())
        else:
            ids.add(f"entity[{index}]")
    return ids


def _validate_entities(payload: dict[str, Any]) -> list[IntentFinding]:
    """Validate schema version and entity records; report by stable ID."""
    findings: list[IntentFinding] = []
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        findings.append(
            IntentFinding(
                code="unsupported_schema",
                severity="error",
                subject_id=str(version),
                message=f"unsupported semantic intent schema: {version}",
            )
        )

    entities = _as_list(payload.get("entities"))
    seen: set[str] = set()
    for index, entity in enumerate(entities):
        eid = _record_id(entity, f"entity[{index}]")
        if isinstance(entity, dict):
            value = entity.get("id")
            if not (isinstance(value, str) and value.strip()):
                findings.append(
                    IntentFinding(
                        code="malformed_entity_id",
                        severity="error",
                        subject_id=eid,
                        message="entity has a missing or empty id",
                    )
                )
            elif len(value.strip()) > 128:
                findings.append(
                    IntentFinding(
                        code="malformed_entity_id",
                        severity="error",
                        subject_id=eid,
                        message="entity id exceeds 128 characters",
                    )
                )
        if eid in seen:
            findings.append(
                IntentFinding(
                    code="duplicate_entity_id",
                    severity="error",
                    subject_id=eid,
                    message=f"duplicate entity id: {eid}",
                )
            )
        seen.add(eid)
        if isinstance(entity, dict):
            kind = entity.get("kind")
            if kind not in {
                "person",
                "creature",
                "object",
                "group",
                "location",
                "abstract",
            }:
                findings.append(
                    IntentFinding(
                        code="unknown_entity_kind",
                        severity="warning",
                        subject_id=eid,
                        message=f"entity kind is not a supported kind: {kind!r}",
                    )
                )
    return findings


def _validate_relations(
    payload: dict[str, Any], entity_ids: set[str]
) -> list[IntentFinding]:
    findings: list[IntentFinding] = []
    for index, relation in enumerate(_as_list(payload.get("relations"))):
        rid = _record_id(relation, f"relation[{index}]")
        if not isinstance(relation, dict):
            findings.append(
                IntentFinding(
                    code="dangling_relation",
                    severity="error",
                    subject_id=rid,
                    message="relation record is malformed",
                )
            )
            continue
        for field_name, code in (
            ("subject_id", "dangling_relation_subject"),
            ("target_id", "dangling_relation_target"),
        ):
            ref = relation.get(field_name)
            if not (isinstance(ref, str) and ref.strip()) or ref.strip() not in entity_ids:
                findings.append(
                    IntentFinding(
                        code=code,
                        severity="error",
                        subject_id=rid,
                        message=f"relation {field_name} references unknown entity: {ref!r}",
                    )
                )
    return findings


def _validate_constraints(
    payload: dict[str, Any], entity_ids: set[str]
) -> list[IntentFinding]:
    findings: list[IntentFinding] = []
    for index, constraint in enumerate(_as_list(payload.get("constraints"))):
        cid = _record_id(constraint, f"constraint[{index}]")
        if not isinstance(constraint, dict):
            findings.append(
                IntentFinding(
                    code="dangling_constraint",
                    severity="error",
                    subject_id=cid,
                    message="constraint record is malformed",
                )
            )
            continue
        ref = constraint.get("entity_id")
        if not (isinstance(ref, str) and ref.strip()) or ref.strip() not in entity_ids:
            findings.append(
                IntentFinding(
                    code="dangling_constraint",
                    severity="error",
                    subject_id=cid,
                    message=f"constraint entity_id references unknown entity: {ref!r}",
                )
            )
    return findings


def _validate_contradictions(payload: dict[str, Any]) -> list[IntentFinding]:
    """Contradictory required identity constraints on the same entity are reported by ID."""
    findings: list[IntentFinding] = []
    seen: dict[str, tuple[str, str]] = {}
    for index, constraint in enumerate(_as_list(payload.get("constraints"))):
        if not isinstance(constraint, dict):
            continue
        cid = _record_id(constraint, f"constraint[{index}]")
        if constraint.get("kind") != "identity":
            continue
        entity_id = constraint.get("entity_id")
        statement = str(constraint.get("statement", "")).strip().lower()
        if not isinstance(entity_id, str) or not statement:
            continue
        prior_cid, prior_statement = seen.get(entity_id, (None, None))
        if prior_statement is not None and statement != prior_statement:
            findings.append(
                IntentFinding(
                    code="contradictory_constraint",
                    severity="error",
                    subject_id=cid,
                    message=(
                        f"contradicts identity constraint {prior_cid} on entity "
                        f"{entity_id!r}"
                    ),
                )
            )
        else:
            seen[entity_id] = (cid, statement)
    return findings


def validate_intent_ledger(payload: dict[str, Any], mode: str = "warn") -> list[IntentFinding]:
    """Deterministically validate a persisted ledger payload.

    Operates on the raw dict so malformed or dangling records are reported by
    stable ID rather than raised. Results are repeatable for identical input.
    The ``mode`` argument is accepted for API symmetry but does not filter the
    returned findings; use :func:`blocking_findings` for policy application.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    findings = _validate_entities(payload)
    entity_ids = _entity_ids(_as_list(payload.get("entities")))
    findings.extend(_validate_relations(payload, entity_ids))
    findings.extend(_validate_constraints(payload, entity_ids))
    findings.extend(_validate_contradictions(payload))
    return findings


def blocking_findings(findings: list[IntentFinding], mode: str) -> list[IntentFinding]:
    """Apply a policy to a set of findings.

    - strict: error and warning findings block.
    - warn: only error findings block.
    - compat: no findings block (legacy projects retain current behavior).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if mode == "compat":
        return []
    if mode == "strict":
        return list(findings)
    return [finding for finding in findings if finding.severity == "error"]


#: Key under which the original pre-migration payload is preserved.
MIGRATED_FROM_KEY = "migrated_from"


def legacy_cast_to_ledger(
    actors: list[dict[str, Any]] | None,
    locations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a legacy cast config into a ledger payload without fabricating constraints.

    The bridge maps actors to ``person`` entities and locations to
    ``location`` entities only. No relations or constraints are invented, so no
    unspecified demographic or identity attribute is introduced.
    """
    return ledger_from_legacy_cast(actors, locations).to_dict()


def _is_v1_ledger(payload: dict[str, Any]) -> bool:
    return (
        payload.get("schema_version", SCHEMA_VERSION) == SCHEMA_VERSION
        and "entities" in payload
    )


def migrate_ledger_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Bounded, reversible migration of a persisted ledger payload to v1.

    - Already-v1 payloads pass through untouched (idempotent).
    - Legacy payloads (v0 or a raw cast config) are converted through the
      lossless cast bridge; the original is preserved under ``migrated_from``
      so the migration can be reversed.
    """
    if _is_v1_ledger(payload):
        return dict(payload)
    migrated = legacy_cast_to_ledger(
        payload.get("actors"), payload.get("locations")
    )
    migrated[MIGRATED_FROM_KEY] = dict(payload)
    return migrated


def restore_ledger_original(payload: dict[str, Any]) -> dict[str, Any]:
    """Reverse a migration: return the preserved original, or the payload itself."""
    original = payload.get(MIGRATED_FROM_KEY)
    if isinstance(original, dict):
        return dict(original)
    return dict(payload)
