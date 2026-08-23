from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from feverslop.domain.scene_cast import resolve_scene_cast


class ReferenceKind(Enum):
    ACTOR = "actor"
    LOCATION = "location"
    BACKGROUND = "background"
    STYLE = "style"
    STORYBOARD_FRAME = "storyboard_frame"
    STORYBOARD_PAGE = "storyboard_page"
    MSR_SHEET = "msr_sheet"
    INGREDIENTS_SHEET = "ingredients_sheet"
    CONTINUITY = "continuity"


@dataclass(frozen=True)
class ReferenceProvenance:
    source: str
    generated_at: str = ""
    job_action: str = ""
    generated_by: str = ""


@dataclass(frozen=True)
class ReferenceLook:
    id: str
    reference_id: str
    label: str = ""


@dataclass(frozen=True)
class ReferenceAsset:
    id: str
    kind: ReferenceKind
    label: str = ""
    description: str = ""
    path: str = ""
    width: int = 0
    height: int = 0
    exists: bool = True
    provenance: ReferenceProvenance | None = None
    looks: tuple[ReferenceLook, ...] = ()
    stale: bool = False
    generation_state: str = ""


@dataclass(frozen=True)
class PropInteraction:
    actor_id: str
    prop_id: str
    action: str
    relationship: str = ""
    actor_look_id: str = ""
    prop_look_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("actor_id", "prop_id", "action"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"prop interaction {field_name} is required")

    def to_dict(self) -> dict[str, str]:
        return {key: value for key, value in {
            "actor_id": self.actor_id, "prop_id": self.prop_id, "action": self.action,
            "relationship": self.relationship, "actor_look_id": self.actor_look_id, "prop_look_id": self.prop_look_id,
        }.items() if value}

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> PropInteraction:
        return cls(
            actor_id=str(payload.get("actor_id", "")), prop_id=str(payload.get("prop_id", "")),
            action=str(payload.get("action", "")), relationship=str(payload.get("relationship", "")),
            actor_look_id=str(payload.get("actor_look_id", "")), prop_look_id=str(payload.get("prop_look_id", "")),
        )


@dataclass(frozen=True)
class SceneReferenceAssignment:
    scene_number: int
    actor_ids: tuple[str, ...] = ()
    location_ids: tuple[str, ...] = ()
    background_ids: tuple[str, ...] = ()
    style_ids: tuple[str, ...] = ()
    actor_look_ids: dict[str, str] | None = None
    prop_ids: tuple[str, ...] = ()
    prop_interactions: tuple[PropInteraction, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.scene_number, bool) or not isinstance(self.scene_number, int) or self.scene_number <= 0:
            raise ValueError("Scene number must be a positive integer")
        object.__setattr__(self, "actor_ids", tuple(dict.fromkeys(str(a).strip() for a in self.actor_ids if str(a).strip())))
        object.__setattr__(self, "location_ids", tuple(dict.fromkeys(str(loc).strip() for loc in self.location_ids if str(loc).strip())))
        object.__setattr__(self, "background_ids", tuple(dict.fromkeys(str(b).strip() for b in self.background_ids or () if str(b).strip())))
        object.__setattr__(self, "style_ids", tuple(dict.fromkeys(str(s).strip() for s in self.style_ids if str(s).strip())))
        object.__setattr__(self, "prop_ids", tuple(dict.fromkeys(str(p).strip() for p in self.prop_ids if str(p).strip())))
        interactions = tuple(self.prop_interactions)
        if any(not isinstance(item, PropInteraction) for item in interactions):
            raise ValueError("prop_interactions must contain PropInteraction objects")
        object.__setattr__(self, "prop_interactions", interactions)
        object.__setattr__(
            self,
            "actor_look_ids",
            {str(k): str(v) for k, v in (self.actor_look_ids or {}).items() if str(k) and str(v)},
        )
        if len(self.actor_ids) > 4:
            raise ValueError(f"Scene {self.scene_number} has {len(self.actor_ids)} actors; at most 4 actors allowed")

    def validate_against(
        self,
        *,
        known_actor_ids: Iterable[str],
        known_location_ids: Iterable[str],
        known_background_ids: Iterable[str] | None = None,
        known_prop_ids: Iterable[str] | None = None,
        max_scene_actors: int = 4,
    ) -> list[str]:
        known_actors = set(known_actor_ids)
        known_locations = set(known_location_ids)
        known_backgrounds = set(known_background_ids or ())
        known_props = set(known_prop_ids or ())
        issues: list[str] = []

        for aid in self.actor_ids:
            if aid not in known_actors:
                issues.append(f"Unknown actor ID: {aid}")

        for lid in self.location_ids:
            if lid not in known_locations:
                issues.append(f"Unknown location ID: {lid}")

        for bgid in self.background_ids:
            if known_backgrounds and bgid not in known_backgrounds:
                issues.append(f"Unknown background ID: {bgid}")

        for prop_id in self.prop_ids:
            if prop_id not in known_props:
                issues.append(f"Unknown prop ID: {prop_id}")
        for interaction in self.prop_interactions:
            if interaction.actor_id not in known_actors:
                issues.append(f"Unknown interaction actor ID: {interaction.actor_id}")
            if interaction.prop_id not in known_props:
                issues.append(f"Unknown interaction prop ID: {interaction.prop_id}")

        resolved = resolve_scene_cast(
            selected_actor_ids=self.actor_ids,
            available_actors=[{"id": aid} for aid in self.actor_ids if aid in known_actors],
            subject_mode="multi",
            max_scene_actors=max_scene_actors,
        )
        if len(self.actor_ids) > len(resolved.actors):
            issues.append(f"Scene {self.scene_number} has {len(self.actor_ids)} actors; at most {max_scene_actors} actors allowed")

        return issues


@dataclass(frozen=True)
class ReferenceWorkspaceSnapshot:
    assets: tuple[ReferenceAsset, ...]
    assignments: tuple[SceneReferenceAssignment, ...]
    revision: str = ""
    project_id: str = ""

    def get_asset(self, asset_id: str) -> ReferenceAsset | None:
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None

    def get_assignments_for_scene(self, scene_number: int) -> tuple[SceneReferenceAssignment, ...]:
        return tuple(a for a in self.assignments if a.scene_number == scene_number)

    def get_assignments_for_asset(self, asset_id: str) -> tuple[SceneReferenceAssignment, ...]:
        return tuple(
            a
            for a in self.assignments
            if asset_id in (
                tuple(a.actor_ids)
                + tuple(a.location_ids)
                + tuple(a.background_ids)
                + tuple(a.style_ids)
                + tuple(a.prop_ids)
            )
        )

    def filter_assets(
        self,
        *,
        kinds: Iterable[ReferenceKind] | None = None,
        stale_only: bool = False,
        missing_only: bool = False,
    ) -> tuple[ReferenceAsset, ...]:
        kinds_set = set(kinds) if kinds is not None else None
        results = []
        for asset in self.assets:
            if kinds_set is not None and asset.kind not in kinds_set:
                continue
            if stale_only and not asset.stale:
                continue
            if missing_only and asset.exists:
                continue
            results.append(asset)
        return tuple(results)

    def scenes_using_asset(self, asset_id: str) -> tuple[int, ...]:
        return tuple(sorted(a.scene_number for a in self.get_assignments_for_asset(asset_id)))
