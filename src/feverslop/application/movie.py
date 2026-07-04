from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from feverslop.domain.movie import MovieProject
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
        shots = self.planner.plan_shots(
            story_arch=story_arch,
            desired_length=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
            min_duration=float(request.min_scene_duration),
            max_duration=float(request.max_scene_duration),
        )
        movie = MovieProject(
            slug=slug,
            name=request.name,
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
        render_plan_path = movie_dir / "render_plan.json"
        reference_manifest_path = movie_dir / "references" / "manifest.json"
        story_arch_path.write_text(json.dumps(asdict(movie.story_arch), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if config:
            (project_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_plan_path.write_text(json.dumps(_render_plan(movie), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        reference_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        reference_manifest_path.write_text(json.dumps(_reference_manifest(movie), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return MovieScaffoldResult(slug, project_dir, story_arch_path, render_plan_path, reference_manifest_path)


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
    data.pop("actor_ids", None)
    data.pop("location_id", None)
    data["reference_ids"] = {
        "actors": list(shot.actor_ids) or [_default_actor_id(config)],
        "location": shot.location_id or _default_location_id(config),
    }
    return data


def _reference_manifest(movie: MovieProject) -> dict:
    config = movie_config(movie)
    actors = _movie_actor_refs(movie, config)
    locations = _movie_location_refs(movie, config)
    return {
        "project_type": "movie",
        "actors": [
            {
                "id": actor["id"],
                "name": actor["name"],
                "role": actor.get("role", ""),
                "visual_description": actor["prompt"],
                "image_prompt": actor["prompt"],
                "prompt": actor["prompt"],
                "status": "required",
                "msr_sheet_path": "",
            }
            for actor in actors
        ],
        "locations": [
            {
                "id": location["id"],
                "name": location["name"],
                "visual_description": location["prompt"],
                "image_prompt": location["prompt"],
                "prompt": location["prompt"],
                "status": "required",
                "msr_sheet_path": "",
            }
            for location in locations
        ],
    }


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


def _movie_actor_refs(movie: MovieProject, config: dict) -> list[dict[str, str]]:
    configured = config.get("actors") if isinstance(config.get("actors"), list) else []
    actors = [
        {
            "id": _safe_id(actor.get("id") or actor.get("name"), f"actor_{index}"),
            "name": str(actor.get("name") or actor.get("id") or f"Actor {index}").strip(),
            "prompt": str(actor.get("image_prompt") or actor.get("visual_description") or actor.get("name") or f"Actor {index}").strip(),
        }
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
        actors.append({
            "id": actor_id,
            "name": display_name,
            "prompt": _actor_reference_prompt(display_name, _shots_for_actor(movie, actor_id)),
        })
    return actors


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
    return [shot for shot in movie.shots if actor_id in getattr(shot, "actor_ids", ())]


def _actor_reference_prompt(name: str, shots: list) -> str:
    cues = _shot_cues(shots, include_location=False)
    if cues:
        return (
            f"Full-body cinematic character reference sheet for {name}. "
            f"Visual identity inferred from scenes: {cues}. "
            "Consistent face, hair, body shape, wardrobe, posture, neutral expression, clean studio background, no text."
        )
    return f"Full-body cinematic character reference sheet for {name}, consistent face, wardrobe, posture, clean studio background, no text."


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
