from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Real
from typing import Any, Mapping


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
        return {
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
