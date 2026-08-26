from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CapabilityValidation:
    missing_models: tuple[str, ...]
    missing_nodes: tuple[str, ...]
    legacy_fallback_allowed: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing_models and not self.missing_nodes


@dataclass(frozen=True)
class WorkflowCapabilityManifest:
    manifest_id: str
    model_family: str
    model_version: str
    required_models: tuple[str, ...]
    required_nodes: tuple[str, ...]
    optional_models: tuple[str, ...] = ()
    optional_nodes: tuple[str, ...] = ()

    @classmethod
    def create(
        cls, *, manifest_id: str, model_family: str, model_version: str,
        required_models: Iterable[str], required_nodes: Iterable[str],
        optional_models: Iterable[str] = (), optional_nodes: Iterable[str] = (),
    ) -> "WorkflowCapabilityManifest":
        def names(values: Iterable[str], field: str) -> tuple[str, ...]:
            if isinstance(values, (str, bytes)):
                raise ValueError(f"{field} must be iterable")
            result = tuple(sorted({str(value).strip() for value in values}))
            if any(not value for value in result):
                raise ValueError(f"{field} cannot contain blank names")
            return result

        identity = tuple(str(value).strip() for value in (manifest_id, model_family, model_version))
        if any(not value for value in identity):
            raise ValueError("manifest_id, model_family, and model_version are required")
        return cls(
            *identity,
            names(required_models, "required_models"),
            names(required_nodes, "required_nodes"),
            names(optional_models, "optional_models"),
            names(optional_nodes, "optional_nodes"),
        )

    def validate_inventory(self, *, models: Iterable[str], nodes: Iterable[str]) -> CapabilityValidation:
        available_models = {str(value).strip() for value in models}
        available_nodes = {str(value).strip() for value in nodes}
        return CapabilityValidation(
            missing_models=tuple(item for item in self.required_models if item not in available_models),
            missing_nodes=tuple(item for item in self.required_nodes if item not in available_nodes),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "required_models": list(self.required_models),
            "required_nodes": list(self.required_nodes),
            "optional_models": list(self.optional_models),
            "optional_nodes": list(self.optional_nodes),
        }
