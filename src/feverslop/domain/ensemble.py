"""Declarative ensemble/group-cast constraints for recurring performance scenes.

An ensemble is a named, fixed set of actor IDs (with optional roles) that must
all be present together in scenes of the configured ``required_scene_types``.
Narrative scenes (whose type is not in ``required_scene_types``) may use any
subset of the ensemble. Missing members are never silently added; validation
returns the missing member IDs so the caller can emit an explicit message.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnsembleMember:
    actor_id: str
    role: str = ""


@dataclass(frozen=True)
class EnsembleConfig:
    id: str
    members: tuple[EnsembleMember, ...] = field(default_factory=tuple)
    required_scene_types: tuple[str, ...] = field(default_factory=tuple)

    @property
    def member_actor_ids(self) -> tuple[str, ...]:
        return tuple(member.actor_id for member in self.members)

    def requires_all_members(self, scene_type: object) -> bool:
        normalized = str(scene_type or "").strip().lower()
        return normalized in {
            str(item).strip().lower() for item in self.required_scene_types
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EnsembleConfig:
        ensemble_id = str(raw.get("id") or "").strip()
        if not ensemble_id:
            raise ValueError("ensemble id is required")
        members_raw = raw.get("members") or []
        if not isinstance(members_raw, list):
            raise ValueError(f"ensemble {ensemble_id!r} members must be an array")
        members: list[EnsembleMember] = []
        seen: set[str] = set()
        for index, item in enumerate(members_raw):
            if not isinstance(item, dict):
                raise ValueError(f"ensemble {ensemble_id!r} members[{index}] must be an object")
            actor_id = str(item.get("actor_id") or "").strip()
            if not actor_id:
                raise ValueError(f"ensemble {ensemble_id!r} members[{index}] actor_id is required")
            if actor_id in seen:
                raise ValueError(f"ensemble {ensemble_id!r} has duplicate member {actor_id!r}")
            seen.add(actor_id)
            members.append(
                EnsembleMember(
                    actor_id=actor_id,
                    role=str(item.get("role") or "").strip(),
                )
            )
        required_raw = raw.get("required_scene_types") or []
        if not isinstance(required_raw, list):
            raise ValueError(f"ensemble {ensemble_id!r} required_scene_types must be an array")
        required = tuple(
            str(item).strip().lower()
            for item in required_raw
            if str(item).strip()
        )
        return cls(id=ensemble_id, members=tuple(members), required_scene_types=required)


def load_ensembles(raw: object) -> tuple[EnsembleConfig, ...]:
    """Parse the optional ``ensembles`` configuration array."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("ensembles must be an array")
    ensembles: list[EnsembleConfig] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"ensembles[{index}] must be an object")
        ensemble = EnsembleConfig.from_dict(item)
        if ensemble.id in seen_ids:
            raise ValueError(f"ensembles has duplicate id {ensemble.id!r}")
        seen_ids.add(ensemble.id)
        ensembles.append(ensemble)
    return tuple(ensembles)


def validate_ensemble_cast(
    ensemble: EnsembleConfig,
    *,
    scene_actor_ids: Iterable[object],
    scene_type: object,
) -> tuple[str, ...]:
    """Return the ensemble member IDs missing from a scene's resolved cast.

    Only scenes whose ``scene_type`` is in ``required_scene_types`` must contain
    every member. For other scene types (narrative) the result is always empty:
    subsets are allowed and nothing is ever auto-added.
    """
    if not ensemble.requires_all_members(scene_type):
        return ()
    present = {
        str(value).strip()
        for value in (scene_actor_ids or ())
        if str(value).strip()
    }
    return tuple(
        member.actor_id
        for member in ensemble.members
        if member.actor_id not in present
    )
