"""Canonical, filesystem-independent contracts for global assets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any


SCHEMA_VERSION = 1


class AssetKind(StrEnum):
    CHARACTER = "character"
    LOCATION = "location"
    STYLE = "style"
    PROP = "prop"


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
class AssetLook:
    id: str
    name: str
    description: str = ""
    hero_image: str = ""
    sheet_image: str = ""
    references: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "look id"))
        object.__setattr__(self, "name", _text(self.name, "look name"))
        object.__setattr__(self, "description", _text(self.description, "look description", required=False))
        for field_name in ("hero_image", "sheet_image"):
            value = getattr(self, field_name)
            if value:
                object.__setattr__(self, field_name, _safe_relative_path(value, field_name))
        normalized_refs = tuple(_safe_relative_path(path, "reference path") for path in self.references)
        object.__setattr__(self, "references", normalized_refs)
        object.__setattr__(self, "metadata", tuple((str(key), str(value)) for key, value in self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "hero_image": self.hero_image,
            "sheet_image": self.sheet_image,
            "references": list(self.references),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AssetLook":
        if not isinstance(payload, dict):
            raise ValueError("look must be an object")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("look metadata must be an object")
        raw_references = payload.get("references", ())
        if not isinstance(raw_references, (list, tuple)):
            raise ValueError("look references must be an array")
        return cls(
            id=payload.get("id", ""), name=payload.get("name", ""),
            description=payload.get("description", ""),
            hero_image=payload.get("hero_image", ""), sheet_image=payload.get("sheet_image", ""),
            references=tuple(raw_references), metadata=tuple(raw_metadata.items()),
        )


@dataclass(frozen=True, slots=True)
class GlobalAsset:
    id: str
    kind: AssetKind
    name: str
    description: str = ""
    looks: tuple[AssetLook, ...] = ()
    revision: int = 1
    schema_version: int = SCHEMA_VERSION
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "asset id"))
        object.__setattr__(self, "name", _text(self.name, "asset name"))
        object.__setattr__(self, "description", _text(self.description, "asset description", required=False))
        try:
            kind = self.kind if isinstance(self.kind, AssetKind) else AssetKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid asset kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("asset revision must be a positive integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported asset schema version: {self.schema_version}")
        looks = tuple(self.looks)
        if any(not isinstance(look, AssetLook) for look in looks):
            raise ValueError("looks must contain AssetLook objects")
        look_ids = [look.id for look in looks]
        if len(look_ids) != len(set(look_ids)):
            raise ValueError("duplicate look id")
        object.__setattr__(self, "looks", looks)
        object.__setattr__(self, "metadata", tuple((str(key), str(value)) for key, value in self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
            "looks": [look.to_dict() for look in self.looks],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalAsset":
        if not isinstance(payload, dict):
            raise ValueError("asset manifest must be an object")
        raw_looks = payload.get("looks", [])
        if not isinstance(raw_looks, list):
            raise ValueError("asset looks must be an array")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("asset metadata must be an object")
        return cls(
            id=payload.get("id", ""), kind=payload.get("kind", ""), name=payload.get("name", ""),
            description=payload.get("description", ""),
            looks=tuple(AssetLook.from_dict(item) for item in raw_looks),
            revision=payload.get("revision", 1), schema_version=payload.get("schema_version", SCHEMA_VERSION),
            metadata=tuple(raw_metadata.items()),
        )


def asset_manifest_path(root: str, asset: GlobalAsset) -> str:
    """Return the canonical manifest path without touching the filesystem."""
    return str(PurePosixPath(root) / asset.kind.value / asset.id / "manifest.json")
