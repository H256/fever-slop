from __future__ import annotations

import json
from dataclasses import asdict, replace

from feverslop.application.movie_common import (
    MovieInput,
    _planner_source_text,
)
from feverslop.application.movie_continuity import (
    movie_continuity_plan_from_dict,
)
from feverslop.application.movie_references import (
    build_movie_actor_reference_prompt,
    build_movie_actor_visual_description,
)
from feverslop.domain.movie import (
    CinematicShot,
    MovieActor,
    MovieBible,
    MovieContinuityPlan,
    MovieContinuityRule,
    MovieLocation,
    MovieProject,
    StoryArch,
)
from feverslop.domain.movie_utils import clean_visual_description, safe_id
from feverslop.ports.movie import ScenePlanningPort


def _render_plan(movie: MovieProject, *, shot_cards: tuple = ()) -> dict:
    cards_by_id = {card.shot_id: card for card in shot_cards}
    return {
        "project_type": "movie",
        "title": movie.name,
        "duration_seconds": movie.duration_seconds,
        "resolution": {"width": movie.width, "height": movie.height},
        "audio_policy": "ltx_native",
        "visual_backends": ["krea2", "ltx_msr"],
        "movie_screenplay_path": "movie/screenplay.json",
        "movie_story_design_path": "movie/story_design.json",
        "movie_narrative_plan_path": "movie/narrative_plan.json",
        "movie_scene_cards_path": "movie/scene_cards.json",
        "movie_shot_cards_path": "movie/shot_cards.json",
        "shots": [_render_plan_shot(shot, movie_config(movie), shot_card=cards_by_id.get(shot.shot_id)) for shot in movie.shots],
    }


def _render_plan_shot(shot, config: dict, *, shot_card=None) -> dict:
    data = asdict(shot)
    data["acting"] = data.get("expression", "")
    data["continuity_notes"] = data.get("continuity_notes", "")
    data["reference_ids"] = {
        "actors": list(shot.actor_ids) or [_default_actor_id(config)],
        "location": shot.location_id or _default_location_id(config),
    }
    if shot_card:
        data["shot_card"] = asdict(shot_card)
        data["start_frame_brief"] = shot_card.start_frame_brief
        data["end_frame_brief"] = shot_card.end_frame_brief
        data["transition_from_previous"] = shot_card.transition_from_previous
        data["transition_reason"] = shot_card.transition_reason
    return data


def _reference_manifest(movie: MovieProject) -> dict:
    return {
        "project_type": "movie",
        "actors": [
            {
                "id": actor.id,
                "name": actor.name,
                "role": actor.role,
                "visual_description": actor.visual_description,
                "image_prompt": build_movie_actor_reference_prompt(actor.name, actor.visual_description),
                "prompt": build_movie_actor_reference_prompt(actor.name, actor.visual_description),
                "status": "required",
                "msr_sheet_path": "",
            }
            for actor in movie.bible.actors
        ],
        "locations": [
            {
                "id": location.id,
                "name": location.name,
                "visual_description": location.visual_description,
                "image_prompt": location.image_prompt or location.visual_description,
                "prompt": location.image_prompt or location.visual_description,
                "status": "required",
                "msr_sheet_path": "",
            }
            for location in movie.bible.locations
        ],
    }


def generate_movie_bible(*, planner: ScenePlanningPort, request: MovieInput, story_arch, config: dict) -> MovieBible:
    bible = planner.generate_movie_bible(
        title=request.name,
        source_type=request.source_type,
        story_text=_planner_source_text(request, config),
        desired_length=float(request.desired_length),
        story_arch=story_arch,
        config=config,
    )
    if isinstance(bible, MovieBible):
        return _normalize_movie_bible(bible, story_arch=story_arch, config=config, request=request)
    return _movie_bible_from_config(request=request, story_arch=story_arch, config=config)


def plan_movie_shots_from_bible(*, planner: ScenePlanningPort, bible: MovieBible, screenplay, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> tuple[CinematicShot, ...]:
    return tuple(
        planner.plan_shots_from_bible(
            bible=bible,
            screenplay=screenplay,
            desired_length=desired_length,
            width=width,
            height=height,
            min_duration=min_duration,
            max_duration=max_duration,
        ),
    )


def generate_movie_continuity_plan(*, planner: ScenePlanningPort, request: MovieInput, bible: MovieBible, shots: tuple[CinematicShot, ...], config: dict) -> MovieContinuityPlan:
    raw = planner.generate_movie_continuity_plan(
        title=request.name,
        source_type=request.source_type,
        story_text=_planner_source_text(request, config),
        desired_length=float(request.desired_length),
        bible=bible,
        shots=shots,
        config=config,
    )
    if isinstance(raw, MovieContinuityPlan):
        return raw.normalize(bible=bible, shots=shots)
    if isinstance(raw, dict):
        return movie_continuity_plan_from_dict(raw, bible=bible, shots=shots)
    return MovieContinuityPlan.fallback(bible=bible, shots=shots)


def constrain_movie_shots_to_bible(shots: tuple[CinematicShot, ...], bible: MovieBible) -> tuple[CinematicShot, ...]:
    return bible.constrain(shots)


def augment_movie_bible_from_shot_references(bible: MovieBible, shots: tuple[CinematicShot, ...], *, config: dict) -> MovieBible:
    augment_actors = not bool(_configured_movie_actors(config))
    augment_locations = not bool(_configured_movie_locations(config))
    return bible.augment_from_shots(shots, actors=augment_actors, locations=augment_locations)


def _bible_dict(bible: MovieBible) -> dict:
    return {
        "title": bible.title,
        "premise": bible.premise,
        "story_arch": asdict(bible.story_arch),
        "actors": [asdict(actor) for actor in bible.actors],
        "locations": [asdict(location) for location in bible.locations],
        "continuity": [asdict(rule) for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
        "runtime_constraints": dict(bible.runtime_constraints),
    }


def movie_bible_from_dict(data: dict) -> MovieBible:
    story_data = data.get("story_arch") or {}
    story_arch = _story_arch_from_dict(story_data, title=str(data.get("title") or "Movie"), premise=str(data.get("premise") or ""))
    return MovieBible(
        title=str(data.get("title") or story_arch.title),
        premise=str(data.get("premise") or story_arch.premise),
        story_arch=story_arch,
        actors=tuple(_actor_from_dict(actor, index) for index, actor in enumerate(data.get("actors") or [], start=1) if isinstance(actor, dict)),
        locations=tuple(_location_from_dict(location, index) for index, location in enumerate(data.get("locations") or [], start=1) if isinstance(location, dict)),
        continuity=tuple(
            MovieContinuityRule(
                id=safe_id(rule.get("id") or rule.get("description"), f"continuity_{index}"),
                description=str(rule.get("description") or "").strip(),
            )
            for index, rule in enumerate(data.get("continuity") or [], start=1)
            if isinstance(rule, dict)
        ),
        style_constraints=tuple(str(item).strip() for item in data.get("style_constraints") or [] if str(item).strip()),
        runtime_constraints=dict(data.get("runtime_constraints") or {}),
    )


def _normalize_movie_bible(bible: MovieBible, *, story_arch, config: dict, request: MovieInput) -> MovieBible:
    configured_actors = _configured_movie_actors(config)
    configured_locations = _configured_movie_locations(config)
    actors = tuple(configured_actors) if configured_actors else tuple(bible.actors) or (_default_movie_actor(request),)
    locations = tuple(configured_locations) if configured_locations else tuple(bible.locations) or (_default_movie_location(),)
    runtime_constraints = _runtime_constraints(request, config)
    runtime_constraints.update(dict(bible.runtime_constraints or {}))
    if "max_scene_actors" in config:
        runtime_constraints["max_scene_actors"] = min(4, max(1, int(config.get("max_scene_actors") or 4)))
    return replace(
        bible,
        title=bible.title or request.name,
        premise=bible.premise or story_arch.premise,
        story_arch=story_arch,
        actors=actors,
        locations=locations,
        runtime_constraints=runtime_constraints,
    )


def _movie_bible_from_config(*, request: MovieInput, story_arch, config: dict) -> MovieBible:
    actors = tuple(_configured_movie_actors(config)) or (_default_movie_actor(request),)
    locations = tuple(_configured_movie_locations(config)) or (_default_movie_location(),)
    continuity = (
        MovieContinuityRule(id="visual_continuity", description="Keep actor wardrobe, locations, lighting logic, and story geography consistent across shots."),
    )
    return MovieBible(
        title=story_arch.title,
        premise=story_arch.premise,
        story_arch=story_arch,
        actors=actors,
        locations=locations,
        continuity=continuity,
        style_constraints=_style_constraints(config),
        runtime_constraints=_runtime_constraints(request, config),
    )


def movie_config(movie: MovieProject) -> dict:
    return dict(movie.config or {})


def _configured_movie_actors(config: dict) -> list[MovieActor]:
    configured = config.get("actors") if isinstance(config.get("actors"), list) else []
    actors = []
    for index, actor in enumerate(configured, start=1):
        if not isinstance(actor, dict):
            continue
        actors.append(_actor_from_dict(actor, index))
    return actors


def _configured_movie_locations(config: dict) -> list[MovieLocation]:
    raw_locations = config.get("structured_locations")
    if not isinstance(raw_locations, list) or not raw_locations:
        raw_locations = config.get("locations") if isinstance(config.get("locations"), list) else []
    locations = []
    for index, location in enumerate(raw_locations, start=1):
        if isinstance(location, dict):
            locations.append(_location_from_dict(location, index))
        elif str(location or "").strip():
            name = str(location).strip()
            locations.append(MovieLocation(id=safe_id(name, f"location_{index}"), name=name, visual_description=name))
    return locations


def _actor_from_dict(actor: dict, index: int) -> MovieActor:
    actor_id = safe_id(actor.get("id") or actor.get("name"), f"actor_{index}")
    name = str(actor.get("name") or actor.get("id") or f"Actor {index}").strip()
    visual_description = build_movie_actor_visual_description(
        str(actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or name).strip(),
    )
    return MovieActor(
        id=actor_id,
        name=name,
        role=str(actor.get("role") or "").strip(),
        visual_description=visual_description,
    )


def _location_from_dict(location: dict, index: int) -> MovieLocation:
    location_id = safe_id(location.get("id") or location.get("name"), f"location_{index}")
    name = str(location.get("name") or location.get("id") or f"Location {index}").strip()
    return MovieLocation(
        id=location_id,
        name=name,
        visual_description=clean_visual_description(
            location.get("visual_description") or location.get("image_prompt") or location.get("prompt"),
            name,
        ),
    )


def _default_movie_actor(request: MovieInput) -> MovieActor:
    subject = request.config.get("subject") if isinstance(request.config, dict) else None
    name = str(subject or "Main Character").strip()
    return MovieActor(
        id=safe_id(name, "main_character"),
        name=name,
        role="lead",
        visual_description=name,
    )


def _default_movie_location() -> MovieLocation:
    location_name = "Primary Location"
    return MovieLocation(
        id=safe_id(location_name, "primary_location"),
        name=location_name,
        visual_description=location_name,
    )


def _style_constraints(config: dict) -> tuple[str, ...]:
    values = []
    for key in ("style", "prompt_guidance", "subject"):
        value = config.get(key)
        if value:
            values.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
    return tuple(values)


def _runtime_constraints(request: MovieInput, config: dict) -> dict:
    constraints = {
        "desired_length": float(request.desired_length),
        "width": int(request.width),
        "height": int(request.height),
        "max_scene_actors": min(4, max(1, int(config.get("max_scene_actors") or 4))),
    }
    dialogue_language = str(config.get("dialogue_language") or "").strip()
    if dialogue_language:
        constraints["dialogue_language"] = dialogue_language
    for key in ("fps", "width", "height"):
        if key in config:
            constraints[key] = int(config[key])
    return constraints


def _story_arch_from_dict(data: dict, *, title: str, premise: str):
    return StoryArch(
        title=str(data.get("title") or title),
        premise=str(data.get("premise") or premise),
        beats=tuple(str(beat).strip() for beat in data.get("beats") or [] if str(beat).strip()),
    )


def _default_actor_id(config: dict) -> str:
    actors = config.get("actors") if isinstance(config.get("actors"), list) else []
    if actors and isinstance(actors[0], dict):
        return safe_id(actors[0].get("id") or actors[0].get("name"), "actor_1")
    return "main_character"


def _default_location_id(config: dict) -> str:
    locations = config.get("locations") if isinstance(config.get("locations"), list) else []
    if locations:
        first = locations[0]
        if isinstance(first, dict):
            return safe_id(first.get("id") or first.get("name"), "location_1")
        return safe_id(str(first), "location_1")
    return "primary_location"
