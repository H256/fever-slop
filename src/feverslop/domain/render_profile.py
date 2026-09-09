from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Real
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from feverslop.domain.artifact_hash import is_sha256_hex
from feverslop.domain.duration_capability import DurationCapability
from feverslop.domain.artifact_hash import fingerprint_json


class RenderProfileSchemaError(ValueError):
    """Raised when a render profile does not satisfy schema version 1."""


class RenderMode(StrEnum):
    T2V = "t2v"
    I2V = "i2v"
    R2V = "r2v"
    MSR = "msr"
    INGREDIENTS = "ingredients"


class QualityProfile(StrEnum):
    DRAFT = "draft"
    STANDARD = "standard"
    FINAL = "final"


class RenderPassStrategy(StrEnum):
    SINGLE_PASS = "single_pass"
    TWO_PASS = "two_pass"


class PostprocessStrategy(StrEnum):
    NONE = "none"
    SEEDVR = "seedvr"


def _enum_value(enum_type: type[StrEnum], value: str | StrEnum, field: str) -> StrEnum:
    try:
        return enum_type(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise RenderProfileSchemaError(f"{field} must be one of: {allowed}") from exc


@dataclass(frozen=True)
class RenderProfile:
    """Versioned, model-agnostic capabilities for one render configuration."""

    schema_version: int
    profile_id: str
    model_family: str
    mode: RenderMode
    quality: QualityProfile
    pass_strategy: RenderPassStrategy
    postprocess: PostprocessStrategy
    capabilities: tuple[str, ...]
    max_duration_seconds: float | None = None
    duration_capability: DurationCapability | None = None

    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        model_family: str,
        mode: str | RenderMode,
        quality: str | QualityProfile,
        pass_strategy: str | RenderPassStrategy,
        postprocess: str | PostprocessStrategy,
        capabilities: Any,
        max_duration_seconds: Real | None = None,
        schema_version: int = 1,
        duration_capability: DurationCapability | Mapping[str, Any] | None = None,
    ) -> RenderProfile:
        if type(schema_version) is not int or schema_version != 1:
            raise RenderProfileSchemaError("schema_version must be 1")

        resolved_id = str(profile_id).strip().lower()
        resolved_family = str(model_family).strip().lower()
        if not resolved_id:
            raise RenderProfileSchemaError("profile_id is required")
        if not resolved_family:
            raise RenderProfileSchemaError("model_family is required")

        if isinstance(capabilities, (str, bytes)):
            raise RenderProfileSchemaError("capabilities must be an iterable of names")
        try:
            normalized_capabilities = sorted({str(item).strip().lower() for item in capabilities})
        except TypeError as exc:
            raise RenderProfileSchemaError("capabilities must be an iterable of names") from exc
        if any(not item for item in normalized_capabilities):
            raise RenderProfileSchemaError("capability names cannot be blank")

        resolved_duration: float | None
        if max_duration_seconds is None:
            resolved_duration = None
        elif isinstance(max_duration_seconds, bool) or not isinstance(max_duration_seconds, Real):
            raise RenderProfileSchemaError("max_duration_seconds must be greater than zero")
        else:
            resolved_duration = float(max_duration_seconds)
            if not isfinite(resolved_duration) or resolved_duration <= 0:
                raise RenderProfileSchemaError("max_duration_seconds must be greater than zero")

        if isinstance(duration_capability, Mapping):
            try:
                duration_capability = DurationCapability.create(**dict(duration_capability))
            except (TypeError, ValueError) as exc:
                raise RenderProfileSchemaError(f"invalid duration_capability: {exc}") from exc
        elif duration_capability is not None and not isinstance(duration_capability, DurationCapability):
            raise RenderProfileSchemaError("duration_capability must be an object")

        return cls(
            schema_version=schema_version,
            profile_id=resolved_id,
            model_family=resolved_family,
            mode=_enum_value(RenderMode, mode, "mode"),
            quality=_enum_value(QualityProfile, quality, "quality"),
            pass_strategy=_enum_value(RenderPassStrategy, pass_strategy, "pass_strategy"),
            postprocess=_enum_value(PostprocessStrategy, postprocess, "postprocess"),
            capabilities=tuple(normalized_capabilities),
            max_duration_seconds=resolved_duration,
            duration_capability=duration_capability,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RenderProfile:
        if not isinstance(payload, Mapping):
            raise RenderProfileSchemaError("render profile must be a mapping")
        try:
            return cls.create(**dict(payload))
        except TypeError as exc:
            raise RenderProfileSchemaError("render profile has invalid or missing fields") from exc

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "model_family": self.model_family,
            "mode": self.mode.value,
            "quality": self.quality.value,
            "pass_strategy": self.pass_strategy.value,
            "postprocess": self.postprocess.value,
            "capabilities": list(self.capabilities),
            "max_duration_seconds": self.max_duration_seconds,
        }
        if self.duration_capability is not None:
            result["duration_capability"] = self.duration_capability.to_dict()
        return result


@dataclass(frozen=True)
class RegisteredRenderProfile:
    profile: RenderProfile
    workflow_path: str

    def __post_init__(self) -> None:
        raw_path = str(self.workflow_path).strip().replace("\\", "/")
        path = PurePosixPath(raw_path)
        if not raw_path or PureWindowsPath(raw_path).drive or path.is_absolute() or ".." in path.parts:
            raise RenderProfileSchemaError("workflow_path must be repository-relative")
        object.__setattr__(self, "workflow_path", path.as_posix())


class RenderProfileRegistry:
    """Deterministic lookup for validated profiles and their workflow assets."""

    def __init__(self, entries: Any) -> None:
        try:
            normalized = tuple(entries)
        except TypeError as exc:
            raise RenderProfileSchemaError("registry entries must be iterable") from exc
        ids = [entry.profile.profile_id for entry in normalized if isinstance(entry, RegisteredRenderProfile)]
        if len(ids) != len(normalized):
            raise RenderProfileSchemaError("registry entries must be RegisteredRenderProfile values")
        if len(ids) != len(set(ids)):
            raise RenderProfileSchemaError("duplicate render profile IDs are not allowed")
        self._entries = {entry.profile.profile_id: entry for entry in normalized}

    def resolve(self, *, profile_id: str, required_capabilities: Any = ()) -> RegisteredRenderProfile:
        key = str(profile_id).strip().lower()
        entry = self._entries.get(key)
        if entry is None:
            raise RenderProfileSchemaError(f"unknown render profile: {key}")
        try:
            required = {str(item).strip().lower() for item in required_capabilities}
        except TypeError as exc:
            raise RenderProfileSchemaError("required_capabilities must be iterable") from exc
        if any(not item for item in required):
            raise RenderProfileSchemaError("required capability names cannot be blank")
        missing = sorted(required - set(entry.profile.capabilities))
        if missing:
            raise RenderProfileSchemaError(
                f"render profile '{key}' lacks capabilities: {', '.join(missing)}"
            )
        return entry


@dataclass(frozen=True)
class RenderProfileResolution:
    """The fully resolved, fingerprintable render-profile provenance."""

    requested_profile_id: str
    entry: RegisteredRenderProfile
    workflow_sha256: str
    model_assets: tuple[str, ...]
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        requested_profile_id: str,
        entry: RegisteredRenderProfile,
        workflow_sha256: str,
        model_assets: Any = (),
    ) -> RenderProfileResolution:
        requested = str(requested_profile_id).strip().lower()
        if not requested:
            raise RenderProfileSchemaError("requested_profile_id is required")
        digest = str(workflow_sha256).strip().lower()
        if not is_sha256_hex(digest):
            raise RenderProfileSchemaError("workflow_sha256 must be a SHA-256 hex digest")
        if not isinstance(entry, RegisteredRenderProfile):
            raise RenderProfileSchemaError("entry must be a RegisteredRenderProfile")
        try:
            assets = tuple(sorted({str(asset).strip() for asset in model_assets}))
        except TypeError as exc:
            raise RenderProfileSchemaError("model_assets must be iterable") from exc
        if any(not asset for asset in assets):
            raise RenderProfileSchemaError("model asset names cannot be blank")
        semantic = {
            "requested_profile_id": requested,
            "profile": entry.profile.to_dict(),
            "workflow_path": entry.workflow_path,
            "workflow_sha256": digest,
            "model_assets": list(assets),
        }
        fingerprint = fingerprint_json(semantic, ensure_ascii=False)
        return cls(
            requested_profile_id=requested,
            entry=entry,
            workflow_sha256=digest,
            model_assets=assets,
            fingerprint=fingerprint,
        )

    @property
    def profile(self) -> RenderProfile:
        return self.entry.profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_profile_id": self.requested_profile_id,
            "resolved_profile_id": self.profile.profile_id,
            "profile": self.profile.to_dict(),
            "workflow_path": self.entry.workflow_path,
            "workflow_sha256": self.workflow_sha256,
            "capabilities": list(self.profile.capabilities),
            "model_assets": list(self.model_assets),
            "fingerprint": self.fingerprint,
        }
