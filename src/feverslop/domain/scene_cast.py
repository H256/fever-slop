from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SceneActor:
    id: str
    name: str
    role: str = ""
    visual_description: str = ""


@dataclass(frozen=True)
class SceneCast:
    actors: tuple[SceneActor, ...]
    primary_actor_id: str = ""

    @property
    def visible_actor_ids(self) -> tuple[str, ...]:
        return tuple(actor.id for actor in self.actors)


def resolve_scene_cast(
    *,
    selected_actor_ids: Iterable[object],
    available_actors: Iterable[dict[str, Any]],
    subject_mode: str = "multi",
    max_scene_actors: int = 4,
) -> SceneCast:
    actors_by_id = {
        actor.id: actor
        for item in available_actors
        if (actor := _scene_actor(item)) is not None
    }
    limit = 1 if str(subject_mode).strip().lower() == "single" else min(4, max(1, int(max_scene_actors)))
    selected = tuple(dict.fromkeys(str(value).strip() for value in selected_actor_ids if str(value).strip()))
    valid = tuple(actor_id for actor_id in selected if actor_id in actors_by_id)[:limit]
    if not valid and actors_by_id:
        valid = (next(iter(actors_by_id)),)
    actors = tuple(actors_by_id[actor_id] for actor_id in valid)
    return SceneCast(actors=actors, primary_actor_id=actors[0].id if actors else "")


def scene_cast_to_prompt_payload(cast: SceneCast) -> dict[str, Any]:
    return {
        "visible_actor_ids": list(cast.visible_actor_ids),
        "primary_actor_id": cast.primary_actor_id,
        "requires_group_staging": len(cast.actors) > 1,
        "actors": [asdict(actor) for actor in cast.actors],
    }


def _scene_actor(item: dict[str, Any]) -> SceneActor | None:
    actor_id = str(item.get("id") or "").strip()
    if not actor_id:
        return None
    return SceneActor(
        id=actor_id,
        name=str(item.get("name") or actor_id).strip(),
        role=str(item.get("role") or "").strip(),
        visual_description=str(item.get("visual_description") or "").strip(),
    )
