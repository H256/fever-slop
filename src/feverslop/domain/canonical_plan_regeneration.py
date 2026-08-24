from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class CanonicalPlanSnapshot:
    artifact_id: str
    exists: bool
    sha256: str | None
    scenes: tuple[dict[str, Any], ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class RegenerationDiagnostic:
    code: str
    message: str
    severity: Literal["warning", "error"] = "warning"
    scene_id: str | None = None
    scene_number: int | None = None


@dataclass(frozen=True)
class CanonicalRegenerationResult:
    scenes: tuple[dict[str, Any], ...]
    diagnostics: tuple[RegenerationDiagnostic, ...] = ()
