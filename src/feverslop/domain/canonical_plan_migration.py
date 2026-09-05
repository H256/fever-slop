from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from feverslop.domain.artifact_hash import fingerprint_json

FindingKind = Literal["importable", "unresolved", "no_op"]


@dataclass(frozen=True)
class MigrationDocument:
    path: str
    sha256: str
    value: Any = field(default=None, repr=False, compare=False)
    error: str | None = None


@dataclass(frozen=True)
class MigrationSource:
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class MigrationInput:
    project_id: str
    documents: tuple[MigrationDocument, ...]


@dataclass(frozen=True)
class MigrationFinding:
    kind: FindingKind
    source_path: str
    reason: str
    scene_id: str | None = None
    segment_id: str | None = None
    scene_number: int | None = None
    role: str | None = None
    field_path: str | None = None
    matched_by: str | None = None
    value: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "source_path": self.source_path,
            "reason": self.reason,
            "scene_id": self.scene_id,
            "segment_id": self.segment_id,
            "scene_number": self.scene_number,
            "role": self.role,
            "field_path": self.field_path,
            "matched_by": self.matched_by,
        }
        if self.value is not None:
            result["value_sha256"] = _value_hash(self.value)
        return result


@dataclass(frozen=True)
class MigrationReport:
    project_id: str = field(repr=False, compare=False)
    sources: tuple[MigrationSource, ...]
    findings: tuple[MigrationFinding, ...]
    base_plan: list[dict[str, Any]] = field(repr=False, compare=False)

    @property
    def importable(self) -> tuple[MigrationFinding, ...]:
        return tuple(item for item in self.findings if item.kind == "importable")

    @property
    def unresolved(self) -> tuple[MigrationFinding, ...]:
        return tuple(item for item in self.findings if item.kind == "unresolved")

    @property
    def no_op(self) -> tuple[MigrationFinding, ...]:
        return tuple(item for item in self.findings if item.kind == "no_op")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "feverslop.canonical-plan-migration.v1",
            "sources": [source.to_dict() for source in self.sources],
            "summary": {
                "importable": len(self.importable),
                "unresolved": len(self.unresolved),
                "no_op": len(self.no_op),
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }


def value_hash(value: Any) -> str:
    return _value_hash(value)


def _value_hash(value: Any) -> str:
    return fingerprint_json(value, ensure_ascii=False)
