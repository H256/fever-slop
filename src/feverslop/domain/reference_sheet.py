"""Backend-neutral contracts for turning generated sequences into reference sheets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from feverslop.domain.global_library import AssetKind


class AnchorKind(StrEnum):
    SINGLE_VIEW = "single_view"
    SHEET = "sheet"


@dataclass(frozen=True, slots=True)
class CompiledReferenceSheetPlan:
    kind: str
    view_count: int
    view_labels: tuple[str, ...]
    framing: str
    coverage: str
    rotation: str
    backdrop: str
    duration_seconds: float
    anchor_rule: str
    identity_constraints: str
    negative_constraints: str
    anchor_description: str = ""

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "view_count": self.view_count,
            "view_labels": list(self.view_labels),
            "framing": self.framing,
            "coverage": self.coverage,
            "rotation": self.rotation,
            "backdrop": self.backdrop,
            "duration_seconds": self.duration_seconds,
            "anchor_rule": self.anchor_rule,
            "identity_constraints": self.identity_constraints,
            "negative_constraints": self.negative_constraints,
            "anchor_description": self.anchor_description,
        }


class ReferenceSheetPlan(BaseModel):
    kind: str
    anchor_description: str = ""
    view_count: int = 0
    view_labels: list[str] = Field(default_factory=list)
    framing: str = ""
    coverage: str = ""
    rotation: str = ""
    backdrop: str = ""
    duration_seconds: float = 0.0
    anchor_rule: str = ""
    identity_constraints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise ValueError(f"{field_name} is required")
    return result


def _safe_relative_path(value: str, field_name: str) -> str:
    result = _text(value, field_name)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or "\\" in result:
        raise ValueError(f"{field_name} must be a safe relative path")
    return result


@dataclass(frozen=True, slots=True)
class ReferenceSheetRequest:
    asset_kind: AssetKind
    asset_id: str
    look_id: str
    anchor_image: str
    backend: str
    profile: str
    anchor_kind: AnchorKind = AnchorKind.SINGLE_VIEW
    allow_sheet_anchor: bool = False

    def __post_init__(self) -> None:
        try:
            asset_kind = self.asset_kind if isinstance(self.asset_kind, AssetKind) else AssetKind(self.asset_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid asset kind: {self.asset_kind!r}") from exc
        try:
            anchor_kind = self.anchor_kind if isinstance(self.anchor_kind, AnchorKind) else AnchorKind(self.anchor_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid anchor kind: {self.anchor_kind!r}") from exc
        if not isinstance(self.allow_sheet_anchor, bool):
            raise ValueError("allow_sheet_anchor must be a boolean")
        if anchor_kind is AnchorKind.SHEET and not self.allow_sheet_anchor:
            raise ValueError("sheet anchors require allow_sheet_anchor=True")
        object.__setattr__(self, "asset_kind", asset_kind)
        object.__setattr__(self, "anchor_kind", anchor_kind)
        object.__setattr__(self, "asset_id", _text(self.asset_id, "asset id"))
        object.__setattr__(self, "look_id", _text(self.look_id, "look id"))
        object.__setattr__(self, "anchor_image", _safe_relative_path(self.anchor_image, "anchor image"))
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        object.__setattr__(self, "profile", _text(self.profile, "profile"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_kind": self.asset_kind.value,
            "asset_id": self.asset_id,
            "look_id": self.look_id,
            "anchor_image": self.anchor_image,
            "backend": self.backend,
            "profile": self.profile,
            "anchor_kind": self.anchor_kind.value,
            "allow_sheet_anchor": self.allow_sheet_anchor,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceSheetRequest":
        if not isinstance(payload, dict):
            raise ValueError("reference sheet request must be an object")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ReferenceArtifact:
    kind: str
    path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, "artifact kind"))
        object.__setattr__(self, "path", _safe_relative_path(self.path, "artifact path"))

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceArtifact":
        if not isinstance(payload, dict):
            raise ValueError("reference artifact must be an object")
        return cls(kind=payload.get("kind", ""), path=payload.get("path", ""))


@dataclass(frozen=True, slots=True)
class ReferenceSheetProvenance:
    backend: str
    profile: str
    seed: int | None = None
    prompt_revision: str = ""

    def __post_init__(self) -> None:
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("provenance seed must be an integer or null")
        object.__setattr__(self, "backend", _text(self.backend, "provenance backend"))
        object.__setattr__(self, "profile", _text(self.profile, "provenance profile"))
        object.__setattr__(self, "prompt_revision", _text(self.prompt_revision, "prompt revision", required=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "profile": self.profile,
            "seed": self.seed,
            "prompt_revision": self.prompt_revision,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceSheetProvenance":
        if not isinstance(payload, dict):
            raise ValueError("reference sheet provenance must be an object")
        return cls(
            backend=payload.get("backend", ""),
            profile=payload.get("profile", ""),
            seed=payload.get("seed"),
            prompt_revision=payload.get("prompt_revision", ""),
        )


@dataclass(frozen=True, slots=True)
class ReferenceSheetResult:
    request_fingerprint: str
    artifacts: tuple[ReferenceArtifact, ...]
    provenance: ReferenceSheetProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_fingerprint", _text(self.request_fingerprint, "request fingerprint"))
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ReferenceArtifact) for artifact in artifacts):
            raise ValueError("artifacts must contain ReferenceArtifact objects")
        if not isinstance(self.provenance, ReferenceSheetProvenance):
            raise ValueError("provenance must be a ReferenceSheetProvenance")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_fingerprint": self.request_fingerprint,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceSheetResult":
        if not isinstance(payload, dict):
            raise ValueError("reference sheet result must be an object")
        raw_artifacts = payload.get("artifacts", [])
        if not isinstance(raw_artifacts, (list, tuple)):
            raise ValueError("reference sheet artifacts must be an array")
        return cls(
            request_fingerprint=payload.get("request_fingerprint", ""),
            artifacts=tuple(ReferenceArtifact.from_dict(item) for item in raw_artifacts),
            provenance=ReferenceSheetProvenance.from_dict(payload.get("provenance", {})),
        )
