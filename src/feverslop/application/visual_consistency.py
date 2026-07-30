from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from feverslop.domain.movie_utils import transition_from_previous
from feverslop.domain.visual_consistency import (
    ReferenceAnchor,
    SceneConsistencyContract,
)
from feverslop.ports.visual_consistency import ReferenceManifestSnapshot


def normalize_reference_ids(scene: object) -> tuple[tuple[str, ...], str]:
    if not isinstance(scene, Mapping):
        return (), ""

    movie = scene.get("reference_ids")
    song = scene.get("references")
    movie = movie if isinstance(movie, Mapping) else {}
    song = song if isinstance(song, Mapping) else {}
    actor_values = next(
        (
            values
            for candidate in (
                movie.get("actors"),
                song.get("actor_ids"),
                scene.get("actor_ids"),
            )
            if (values := _string_values(candidate))
        ),
        (),
    )
    location_value = next(
        (
            value.strip()
            for candidate in (
                movie.get("location"),
                song.get("location_id"),
                scene.get("location_id"),
            )
            if isinstance(candidate, str) and (value := candidate).strip()
        ),
        "",
    )

    actors: list[str] = []
    seen: set[str] = set()
    for value in actor_values:
        if value not in seen:
            actors.append(value)
            seen.add(value)
    return tuple(actors), location_value


def build_scene_contract(
    scene: Mapping[str, Any],
    snapshot: ReferenceManifestSnapshot,
    *,
    mode: str,
    workflow_profile: str,
) -> SceneConsistencyContract:
    scene_number = scene.get("scene") if isinstance(scene, Mapping) else None
    if type(scene_number) is not int or scene_number <= 0:
        raise ValueError("scene must be a positive integer")

    actor_ids, location_id = normalize_reference_ids(scene)
    actors = tuple(
        _anchor(
            snapshot.actors,
            scene_number=scene_number,
            kind="actor",
            semantic_id=actor_id,
            look_id=_actor_look_id(scene, actor_id),
        )
        for actor_id in actor_ids
    )
    location = (
        _anchor(
            snapshot.locations,
            scene_number=scene_number,
            kind="location",
            semantic_id=location_id,
            look_id=_location_look_id(scene),
        )
        if location_id
        else None
    )
    return SceneConsistencyContract.create(
        scene=scene_number,
        mode=mode,
        workflow_profile=workflow_profile,
        actors=actors,
        location=location,
        transition_from_previous=transition_from_previous(
            scene.get("transition_from_previous")
        ),
    )


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = (value,)
    elif isinstance(value, (list, tuple)):
        candidates = tuple(value)
    else:
        return ()
    return tuple(
        text
        for item in candidates
        if isinstance(item, str) and (text := item.strip())
    )


def _look_ids(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    value = scene.get("look_ids")
    return value if isinstance(value, Mapping) else {}


def _actor_look_id(scene: Mapping[str, Any], actor_id: str) -> str:
    look_ids = _look_ids(scene)
    actors = look_ids.get("actors")
    if isinstance(actors, Mapping):
        value = actors.get(actor_id)
    else:
        value = None
    if not isinstance(value, str) or not value.strip():
        actor_look_ids = scene.get("actor_look_ids")
        value = (
            actor_look_ids.get(actor_id)
            if isinstance(actor_look_ids, Mapping)
            else None
        )
    return str(value or "default").strip() or "default"


def _location_look_id(scene: Mapping[str, Any]) -> str:
    look_ids = _look_ids(scene)
    value = look_ids.get("location")
    if not isinstance(value, str) or not value.strip():
        value = scene.get("location_look_id")
    return str(value or "default").strip() or "default"


def _anchor(
    anchors: Mapping[tuple[str, str], ReferenceAnchor],
    *,
    scene_number: int,
    kind: str,
    semantic_id: str,
    look_id: str,
) -> ReferenceAnchor:
    try:
        return anchors[(semantic_id, look_id)]
    except KeyError:
        raise ValueError(
            f"Scene {scene_number} missing {kind} reference "
            f"id {semantic_id!r} with look id {look_id!r}"
        ) from None
