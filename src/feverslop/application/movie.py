from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from feverslop.domain.movie import CinematicShot, MovieActor, MovieBible, MovieContinuityRule, MovieLocation, MovieProject
from feverslop.ports.movie import ReferenceGenerationPort, ScenePlanningPort, StoryGenerationPort, VisualGenerationPort


@dataclass(frozen=True)
class MovieInput:
    name: str
    source_type: str
    story_text: str
    desired_length: float
    width: int = 1280
    height: int = 704
    mode: str = "scaffold"
    min_scene_duration: float = 4.0
    max_scene_duration: float = 20.0
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MovieScaffoldResult:
    project_slug: str
    project_dir: Path
    bible_path: Path
    story_arch_path: Path
    render_plan_path: Path
    reference_manifest_path: Path


@dataclass(frozen=True)
class MovieProductionResult(MovieScaffoldResult):
    final_video_path: Path


class ScaffoldMovieUseCase:
    def __init__(self, *, planner: StoryGenerationPort & ScenePlanningPort, projects_root: Path):
        self.planner = planner
        self.projects_root = Path(projects_root)

    def execute(self, request: MovieInput) -> MovieScaffoldResult:
        validate_movie_input(request)
        slug = slugify_project_name(request.name)
        project_dir = self.projects_root / slug
        movie_dir = project_dir / "movie"
        movie_dir.mkdir(parents=True, exist_ok=False)

        config = dict(request.config or {})
        story_arch = self.planner.generate_story_arch(
            title=request.name,
            source_type=request.source_type,
            story_text=_planner_source_text(request, config),
            desired_length=float(request.desired_length),
        )
        bible = generate_movie_bible(
            planner=self.planner,
            request=request,
            story_arch=story_arch,
            config=config,
        )
        shots = plan_movie_shots_from_bible(
            planner=self.planner,
            bible=bible,
            desired_length=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
            min_duration=float(request.min_scene_duration),
            max_duration=float(request.max_scene_duration),
        )
        bible = augment_movie_bible_from_shot_references(bible, shots, config=config)
        shots = constrain_movie_shots_to_bible(shots, bible)
        movie = MovieProject(
            slug=slug,
            name=request.name,
            bible=bible,
            story_arch=story_arch,
            shots=shots,
            duration_seconds=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
            mode=request.mode,
            config=config,
        )

        metadata = {
            "project_type": "movie",
            "display_name": request.name,
            "slug": slug,
            "movie": {
                "source_type": request.source_type,
                "story_text": request.story_text,
                "desired_length": float(request.desired_length),
                "width": int(request.width),
                "height": int(request.height),
                "mode": request.mode,
            },
        }
        (project_dir / ".studio").mkdir(parents=True, exist_ok=True)
        (project_dir / ".studio" / "project.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        story_arch_path = movie_dir / "story_arch.json"
        bible_path = movie_dir / "bible.json"
        render_plan_path = movie_dir / "render_plan.json"
        reference_manifest_path = movie_dir / "references" / "manifest.json"
        story_arch_path.write_text(json.dumps(asdict(movie.story_arch), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        bible_path.write_text(json.dumps(_bible_dict(movie.bible), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if config:
            (project_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_plan_path.write_text(json.dumps(_render_plan(movie), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        reference_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        reference_manifest_path.write_text(json.dumps(_reference_manifest(movie), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return MovieScaffoldResult(slug, project_dir, bible_path, story_arch_path, render_plan_path, reference_manifest_path)


class AutoProduceMovieUseCase:
    def __init__(self, *, scaffold: ScaffoldMovieUseCase, visual_backend: VisualGenerationPort, reference_generator: ReferenceGenerationPort | None = None):
        self.scaffold = scaffold
        self.visual_backend = visual_backend
        self.reference_generator = reference_generator

    def execute(self, request: MovieInput) -> MovieProductionResult:
        scaffolded = self.scaffold.execute(request)
        if self.reference_generator is not None:
            self.reference_generator.generate(project_dir=scaffolded.project_dir)
        final_video = self.visual_backend.render_movie(
            project_dir=scaffolded.project_dir,
            render_plan_path=scaffolded.render_plan_path,
        )
        return MovieProductionResult(
            scaffolded.project_slug,
            scaffolded.project_dir,
            scaffolded.bible_path,
            scaffolded.story_arch_path,
            scaffolded.render_plan_path,
            scaffolded.reference_manifest_path,
            final_video,
        )


def validate_movie_input(request: MovieInput) -> None:
    if request.source_type not in {"short_story", "screenplay"}:
        raise ValueError("source_type must be short_story or screenplay")
    if not request.name.strip():
        raise ValueError("Movie project name is required")
    if not slugify_project_name(request.name):
        raise ValueError("Movie project slug is empty after slugifying the name")
    if len(request.story_text.strip()) < 20:
        raise ValueError("Movie story input is too short")
    if float(request.desired_length) <= 0:
        raise ValueError("desired_length must be positive")
    if int(request.width) <= 0 or int(request.height) <= 0:
        raise ValueError("resolution width and height must be positive")
    if request.mode not in {"scaffold", "full_auto"}:
        raise ValueError("movie mode must be scaffold or full_auto")
    if request.source_type == "screenplay" and not _looks_like_screenplay(request.story_text):
        raise ValueError("screenplay input must contain scene headings such as INT. or EXT.")


def _looks_like_screenplay(text: str) -> bool:
    upper = text.upper()
    return "INT." in upper or "EXT." in upper


def slugify_project_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return re.sub(r"-+", "-", slug)


def _render_plan(movie: MovieProject) -> dict:
    return {
        "project_type": "movie",
        "title": movie.name,
        "duration_seconds": movie.duration_seconds,
        "resolution": {"width": movie.width, "height": movie.height},
        "audio_policy": "ltx_native",
        "visual_backends": ["krea2", "ltx_msr"],
        "shots": [_render_plan_shot(shot, movie_config(movie)) for shot in movie.shots],
    }


def _render_plan_shot(shot, config: dict) -> dict:
    data = asdict(shot)
    data["acting"] = data.get("expression", "")
    data["continuity_notes"] = data.get("continuity_notes", "")
    data["reference_ids"] = {
        "actors": list(shot.actor_ids) or [_default_actor_id(config)],
        "location": shot.location_id or _default_location_id(config),
    }
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
                "image_prompt": location.visual_description,
                "prompt": location.visual_description,
                "status": "required",
                "msr_sheet_path": "",
            }
            for location in movie.bible.locations
        ],
    }


def generate_movie_bible(*, planner, request: MovieInput, story_arch, config: dict) -> MovieBible:
    generator = getattr(planner, "generate_movie_bible", None)
    if callable(generator):
        bible = generator(
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


def plan_movie_shots_from_bible(*, planner, bible: MovieBible, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> tuple[CinematicShot, ...]:
    planner_from_bible = getattr(planner, "plan_shots_from_bible", None)
    if callable(planner_from_bible):
        return tuple(
            planner_from_bible(
                bible=bible,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        )
    return tuple(
        planner.plan_shots(
            story_arch=bible.story_arch,
            desired_length=desired_length,
            width=width,
            height=height,
            min_duration=min_duration,
            max_duration=max_duration,
        )
    )


def constrain_movie_shots_to_bible(shots: tuple[CinematicShot, ...], bible: MovieBible) -> tuple[CinematicShot, ...]:
    actor_ids = [actor.id for actor in bible.actors]
    location_ids = [location.id for location in bible.locations]
    default_actor = actor_ids[0] if actor_ids else "main_character"
    default_location = location_ids[0] if location_ids else "primary_location"
    max_scene_actors = min(4, max(1, int(bible.runtime_constraints.get("max_scene_actors") or 4)))
    constrained = []
    for shot in shots:
        valid_actors = [actor_id for actor_id in shot.actor_ids if actor_id in actor_ids]
        if not valid_actors:
            valid_actors = [default_actor]
        location_id = shot.location_id if shot.location_id in location_ids else default_location
        constrained.append(
            replace(
                shot,
                actor_ids=tuple(dict.fromkeys(valid_actors[:max_scene_actors])),
                location_id=location_id,
                location=_location_name(bible, location_id),
            )
        )
    return tuple(constrained)


def augment_movie_bible_from_shot_references(bible: MovieBible, shots: tuple[CinematicShot, ...], *, config: dict) -> MovieBible:
    configured_actors = bool(_configured_movie_actors(config))
    configured_locations = bool(_configured_movie_locations(config))
    actors = list(bible.actors)
    locations = list(bible.locations)
    if not configured_actors:
        shot_actor_ids = []
        for shot in shots:
            for actor_id in shot.actor_ids:
                if actor_id and actor_id not in shot_actor_ids:
                    shot_actor_ids.append(actor_id)
        if shot_actor_ids and (len(actors) == 1 and actors[0].id == "main_character"):
            actors = []
        known_actor_ids = {actor.id for actor in actors}
        for index, actor_id in enumerate(shot_actor_ids, start=1):
            if actor_id not in known_actor_ids:
                actors.append(_generic_actor_from_id(actor_id, index))
                known_actor_ids.add(actor_id)
    if not configured_locations:
        shot_locations: dict[str, str] = {}
        for shot in shots:
            if shot.location_id and shot.location_id not in shot_locations:
                shot_locations[shot.location_id] = shot.location or shot.location_id.replace("_", " ").title()
        if shot_locations and (len(locations) == 1 and locations[0].id == "primary_location"):
            locations = []
        known_location_ids = {location.id for location in locations}
        for index, (location_id, name) in enumerate(shot_locations.items(), start=1):
            if location_id not in known_location_ids:
                locations.append(_generic_location_from_id(location_id, name, index))
                known_location_ids.add(location_id)
    return replace(bible, actors=tuple(actors), locations=tuple(locations))


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
                id=_safe_id(rule.get("id") or rule.get("description"), f"continuity_{index}"),
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
    actors = tuple(configured_actors) if configured_actors else tuple(bible.actors) or (_default_movie_actor(request, 1),)
    locations = tuple(configured_locations) if configured_locations else tuple(bible.locations) or (_default_movie_location(request, 1),)
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
    actors = tuple(_configured_movie_actors(config)) or (_default_movie_actor(request, 1),)
    locations = tuple(_configured_movie_locations(config)) or (_default_movie_location(request, 1),)
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


def _planner_source_text(request: MovieInput, config: dict) -> str:
    parts = [request.story_text]
    for label, value in [
        ("story_idea", config.get("story_idea")),
        ("style", config.get("style")),
        ("subject", config.get("subject")),
        ("steering", config.get("steering")),
        ("prompt_guidance", config.get("prompt_guidance")),
        ("actors", config.get("actors")),
        ("locations", config.get("locations")),
    ]:
        if value:
            parts.append(f"\n{label}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(parts).strip()


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
            locations.append(MovieLocation(id=_safe_id(name, f"location_{index}"), name=name, visual_description=name))
    return locations


def _actor_from_dict(actor: dict, index: int) -> MovieActor:
    actor_id = _safe_id(actor.get("id") or actor.get("name"), f"actor_{index}")
    name = str(actor.get("name") or actor.get("id") or f"Actor {index}").strip()
    visual_description = build_movie_actor_visual_description(
        str(actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or name).strip()
    )
    return MovieActor(
        id=actor_id,
        name=name,
        role=str(actor.get("role") or "").strip(),
        visual_description=visual_description,
    )


def _location_from_dict(location: dict, index: int) -> MovieLocation:
    location_id = _safe_id(location.get("id") or location.get("name"), f"location_{index}")
    name = str(location.get("name") or location.get("id") or f"Location {index}").strip()
    return MovieLocation(
        id=location_id,
        name=name,
        visual_description=str(location.get("visual_description") or location.get("image_prompt") or location.get("prompt") or name).strip(),
    )


def _default_movie_actor(request: MovieInput, index: int) -> MovieActor:
    subject = request.config.get("subject") if isinstance(request.config, dict) else None
    name = str(subject or "Main Character").strip()
    return MovieActor(
        id=_safe_id(name, "main_character"),
        name=name,
        role="lead",
        visual_description=f"{name}, story-defined cinematic character with consistent face, body shape, hair, wardrobe, and posture",
    )


def _default_movie_location(request: MovieInput, index: int) -> MovieLocation:
    location_name = "Primary Location"
    return MovieLocation(
        id=_safe_id(location_name, "primary_location"),
        name=location_name,
        visual_description="story-defined cinematic location with consistent production design, lighting, geography, and atmosphere",
    )


def _generic_actor_from_id(actor_id: str, index: int) -> MovieActor:
    name = actor_id.replace("_", " ").title()
    return MovieActor(
        id=_safe_id(actor_id, f"actor_{index}"),
        name=name,
        role="character",
        visual_description=f"{name}, story-defined cinematic character with consistent face, hair, body shape, wardrobe, and posture",
    )


def _generic_location_from_id(location_id: str, name: str, index: int) -> MovieLocation:
    display_name = str(name or location_id.replace("_", " ")).strip()
    return MovieLocation(
        id=_safe_id(location_id, f"location_{index}"),
        name=display_name,
        visual_description=f"{display_name}, story-defined cinematic location with consistent production design, geography, lighting, and atmosphere",
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
    for key in ("fps", "width", "height"):
        if key in config:
            constraints[key] = int(config[key])
    return constraints


def _story_arch_from_dict(data: dict, *, title: str, premise: str):
    from feverslop.domain.movie import StoryArch

    return StoryArch(
        title=str(data.get("title") or title),
        premise=str(data.get("premise") or premise),
        beats=tuple(str(beat).strip() for beat in data.get("beats") or [] if str(beat).strip()),
    )


def _location_name(bible: MovieBible, location_id: str) -> str:
    for location in bible.locations:
        if location.id == location_id:
            return location.name
    return location_id.replace("_", " ").title()


def _movie_actor_refs(movie: MovieProject, config: dict) -> list[dict[str, str]]:
    configured = config.get("actors") if isinstance(config.get("actors"), list) else []
    actors = [
        _configured_actor_ref(actor, index)
        for index, actor in enumerate(configured, start=1)
        if isinstance(actor, dict)
    ]
    if actors:
        return actors
    names = []
    for shot in movie.shots:
        for actor_id in shot.actor_ids:
            if actor_id and actor_id not in names:
                names.append(actor_id)
        dialogue = str(shot.dialogue or "")
        speaker = dialogue.split(":", 1)[0].strip() if ":" in dialogue else ""
        if speaker and speaker not in names:
            names.append(speaker)
    if not names:
        names = [_movie_actor_name(movie)]
    actors = []
    for index, name in enumerate(names, start=1):
        display_name = str(name).replace("_", " ").title()
        actor_id = _safe_id(name, f"actor_{index}")
        visual_description = _actor_visual_description(_shots_for_actor(movie, actor_id))
        actors.append({
            "id": actor_id,
            "name": display_name,
            "visual_description": visual_description,
            "prompt": build_movie_actor_reference_prompt(display_name, visual_description),
        })
    return actors


def _configured_actor_ref(actor: dict, index: int) -> dict[str, str]:
    actor_id = _safe_id(actor.get("id") or actor.get("name"), f"actor_{index}")
    name = str(actor.get("name") or actor.get("id") or f"Actor {index}").strip()
    visual_description = build_movie_actor_visual_description(
        str(actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or name).strip()
    )
    return {
        "id": actor_id,
        "name": name,
        "visual_description": visual_description,
        "prompt": build_movie_actor_reference_prompt(name, visual_description),
    }


def _movie_location_refs(movie: MovieProject, config: dict) -> list[dict[str, str]]:
    configured = config.get("locations") if isinstance(config.get("locations"), list) else []
    locations = []
    for index, location in enumerate(configured, start=1):
        if isinstance(location, dict):
            name = str(location.get("name") or location.get("id") or f"Location {index}").strip()
            locations.append({
                "id": _safe_id(location.get("id") or name, f"location_{index}"),
                "name": name,
                "prompt": str(location.get("image_prompt") or location.get("visual_description") or name).strip(),
            })
        elif str(location or "").strip():
            name = str(location).strip()
            locations.append({"id": _safe_id(name, f"location_{index}"), "name": name, "prompt": name})
    if locations:
        return locations
    shot_locations: dict[str, str] = {}
    for shot in movie.shots:
        location_id = str(getattr(shot, "location_id", "") or "").strip()
        location_name = _display_name(str(getattr(shot, "location", "") or "").strip())
        if location_id and location_id not in shot_locations:
            shot_locations[location_id] = location_name or location_id.replace("_", " ").title()
    if shot_locations:
        return [
            {
                "id": _safe_id(location_id, f"location_{index}"),
                "name": name,
                "prompt": f"{name}, story-consistent cinematic environment, production design, lighting, and atmosphere",
            }
            for index, (location_id, name) in enumerate(shot_locations.items(), start=1)
        ]
    name = _movie_location_name(movie)
    return [{
        "id": _safe_id(name, "primary_location"),
        "name": name,
        "prompt": f"{name}, story-consistent cinematic environment, production design, lighting, and atmosphere",
    }]


def _default_actor_id(config: dict) -> str:
    actors = config.get("actors") if isinstance(config.get("actors"), list) else []
    if actors and isinstance(actors[0], dict):
        return _safe_id(actors[0].get("id") or actors[0].get("name"), "actor_1")
    return "main_character"


def _default_location_id(config: dict) -> str:
    locations = config.get("locations") if isinstance(config.get("locations"), list) else []
    if locations:
        first = locations[0]
        if isinstance(first, dict):
            return _safe_id(first.get("id") or first.get("name"), "location_1")
        return _safe_id(str(first), "location_1")
    return "primary_location"


def _safe_id(value: object, fallback: str) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return raw or fallback


def _display_name(value: str) -> str:
    return value.title() if value.isupper() else value


def _movie_actor_name(movie: MovieProject) -> str:
    for shot in movie.shots:
        dialogue = str(getattr(shot, "dialogue", "") or "")
        speaker = dialogue.split(":", 1)[0].strip()
        if speaker:
            return speaker.title()
    return "Main Character"


def _movie_location_name(movie: MovieProject) -> str:
    for shot in movie.shots:
        location = str(getattr(shot, "location", "") or "").strip()
        if location and location != "story-consistent cinematic location":
            return location.title()
    return "Primary Location"


def _shots_for_actor(movie: MovieProject, actor_id: str) -> list:
    shots = [shot for shot in movie.shots if actor_id in getattr(shot, "actor_ids", ())]
    solo_shots = [shot for shot in shots if len(getattr(shot, "actor_ids", ())) == 1]
    return solo_shots or shots


def _actor_reference_prompt(name: str, shots: list) -> str:
    return build_movie_actor_reference_prompt(name, _actor_visual_description(shots))


def _actor_visual_description(shots: list) -> str:
    return build_movie_actor_visual_description(_actor_static_cues(shots))


def build_movie_actor_visual_description(cues: str) -> str:
    return _sanitize_actor_cues(cues)


def build_movie_actor_reference_prompt(name: str, cues: str = "") -> str:
    cue_text = _sanitize_actor_cues(cues)
    description = f" {cue_text}." if cue_text else ""
    return (
        f"Full-body cinematic character reference sheet for {name}.{description} "
        "Four vertical panels in one image: 1st panel head-and-shoulders closeup, "
        "2nd panel straight full-body front view, 3rd panel clean full-body left view, "
        "4th panel clean full-body back view. Consistent face, hair, body shape, wardrobe, "
        "posture, neutral expression, plain white seamless studio background, even reference-sheet lighting, "
        "no environment, no scenery, no props, no text, no extra characters."
    )


def _actor_static_cues(shots: list) -> str:
    parts: list[str] = []
    for shot in shots[:4]:
        for value in (getattr(shot, "description", ""), getattr(shot, "expression", "")):
            text = _sanitize_actor_cue_fragment(str(value or ""))
            if text and text not in parts:
                parts.append(text)
    return "; ".join(parts)[:700]


def _sanitize_actor_cues(cues: str) -> str:
    parts: list[str] = []
    cues = _strip_actor_prompt_boilerplate(str(cues or ""))
    for raw in re.split(r";|\.", cues):
        text = _sanitize_actor_cue_fragment(raw)
        if text and text not in parts:
            parts.append(text)
    return "; ".join(parts).strip(" ;.")


def _strip_actor_prompt_boilerplate(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"(?is)^.*?Full-body cinematic character reference sheet for [^.]+[.]\s*", "", text)
    text = re.sub(r"(?is)\bFour vertical panels in one image\b.*$", "", text)
    text = re.sub(r"(?is)\bConsistent face\b.*$", "", text)
    return text


def _sanitize_actor_cue_fragment(value: str) -> str:
    text = " ".join(str(value or "").split()).strip(" .;,")
    if not text:
        return ""
    lower = text.lower()
    if lower.endswith("'s") or lower in {"the man", "the woman", "the character"}:
        return ""
    if any(token in lower for token in ("jump cut", "shot", "close-up", "closeup", "camera", "tracking", "split-screen")):
        text = re.sub(r"(?i)^a\s+(?:sudden,\s+violent\s+)?jump cut to\s+", "", text)
        text = re.sub(r"(?i)^an?\s+[^.;,]*\bshot of\s+", "", text)
        text = re.sub(r"(?i)^extreme close-up of\s+", "", text)
        text = re.sub(r"(?i)^close-up of\s+", "", text)
        text = re.sub(r"(?i)^medium shot of\s+", "", text)
        text = re.sub(r"(?i)^wide shot of\s+", "", text)
    lower = text.lower()
    if any(
        token in lower
        for token in (
            "lunges",
            "bellows",
            "glides",
            "walks",
            "stumbles",
            "recoiling",
            "falls",
            "stands",
            "tearing through",
            "shaking",
            "appearing from",
            "eye fluttering",
            "eyes roll back",
            "screen fades",
            "enters a trance",
            "gaze",
            "mesmerized",
            "reaches",
            "leans",
            "breathing",
        )
    ):
        if any(token in lower for token in ("gaze", "mesmerized", "reaches")):
            return ""
        if "," not in text and " with " not in lower:
            return ""
        text = re.sub(r"(?i)\btearing through\b.*$", "", text)
        text = re.sub(r"(?i)\bappearing from\b.*$", "", text)
        text = re.sub(r"(?i)\bglides\b.*$", "", text)
        text = re.sub(r"(?i)\bbellows\b.*$", "", text)
        text = re.sub(r"(?i)\blunges\b.*$", "", text)
        text = re.sub(r"(?i)\beye fluttering\b.*$", "", text)
        text = re.sub(r"(?i)\beyes roll back\b.*$", "", text)
        text = re.sub(r"(?i)\bscreen fades\b.*$", "", text)
    text = text.strip(" .;,")
    if text.lower().endswith("'s") or text.lower() in {"the man", "the woman", "the character"}:
        return ""
    return text


def _shot_cues(shots: list, *, include_location: bool) -> str:
    parts: list[str] = []
    for shot in shots[:4]:
        for value in (
            getattr(shot, "description", ""),
            getattr(shot, "action", ""),
            getattr(shot, "expression", ""),
            getattr(shot, "location", "") if include_location else "",
        ):
            text = str(value or "").strip()
            if text and text not in parts:
                parts.append(text)
    return "; ".join(parts)[:700]
