from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
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

        story_arch = self.planner.generate_story_arch(
            title=request.name,
            source_type=request.source_type,
            story_text=request.story_text,
            desired_length=float(request.desired_length),
        )
        shots = self.planner.plan_shots(
            story_arch=story_arch,
            desired_length=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
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
        "shots": [_render_plan_shot(shot) for shot in movie.shots],
    }


def _render_plan_shot(shot) -> dict:
    data = asdict(shot)
    data["reference_ids"] = {"actors": ["main_character"], "location": "primary_location"}
    return data


def _reference_manifest(movie: MovieProject) -> dict:
    actor_name = _movie_actor_name(movie)
    location_name = _movie_location_name(movie)
    return {
        "project_type": "movie",
        "actors": [
            {
                "id": "main_character",
                "name": actor_name,
                "prompt": f"consistent cinematic protagonist {actor_name} for {movie.name}, drawn from the story premise",
                "status": "required",
                "msr_sheet_path": "",
            }
        ],
        "locations": [
            {
                "id": "primary_location",
                "name": location_name,
                "prompt": f"{location_name}, story-consistent cinematic environment, production design, lighting, and atmosphere",
                "status": "required",
                "msr_sheet_path": "",
            }
        ],
    }


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
