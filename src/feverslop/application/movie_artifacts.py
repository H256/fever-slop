from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from feverslop.application.movie import (
    MovieInput,
    _bible_dict,
    build_movie_actor_reference_prompt,
    generate_movie_bible,
    movie_bible_from_dict,
    movie_continuity_plan_to_dict,
)
from feverslop.application.movie_memory import (
    build_movie_narrative_plan_fallback,
    build_movie_scene_cards,
    build_movie_screenplay_fallback,
    build_movie_shot_cards,
    build_movie_story_design_fallback,
    movie_narrative_plan_to_dict,
    movie_scene_cards_from_dict,
    movie_scene_cards_to_dict,
    movie_screenplay_from_dict,
    movie_screenplay_to_dict,
    movie_screenplay_to_markdown,
    movie_shot_cards_to_dict,
    movie_story_design_from_dict,
    movie_story_design_to_dict,
)
from feverslop.domain.movie import CinematicShot, MovieContinuityPlan
from feverslop.domain.movie_utils import transition_from_previous
from feverslop.utils.io import read_json_object


@dataclass(frozen=True)
class MoviePlanningArtifacts:
    bible_path: Path
    story_design_path: Path
    screenplay_path: Path
    narrative_plan_path: Path
    scene_cards_path: Path
    shot_cards_path: Path
    continuity_plan_path: Path
    render_plan_path: Path


def ensure_movie_planning_artifacts(project_dir: Path, *, force_screenplay: bool = False, force_story_design: bool = False) -> MoviePlanningArtifacts:
    project_dir = Path(project_dir)
    bible_path = ensure_movie_bible(project_dir)
    render_plan_path = ensure_movie_render_plan_matches_bible(project_dir)
    story_design_path = ensure_movie_story_design(project_dir, force=force_story_design)
    screenplay_path = ensure_movie_screenplay(project_dir, force=force_screenplay)
    narrative_plan_path = ensure_movie_narrative_plan(project_dir)
    scene_cards_path = ensure_movie_scene_cards(project_dir)
    shot_cards_path = ensure_movie_shot_cards(project_dir)
    continuity_plan_path = ensure_movie_continuity_plan(project_dir)
    return MoviePlanningArtifacts(
        bible_path=bible_path,
        story_design_path=story_design_path,
        screenplay_path=screenplay_path,
        narrative_plan_path=narrative_plan_path,
        scene_cards_path=scene_cards_path,
        shot_cards_path=shot_cards_path,
        continuity_plan_path=continuity_plan_path,
        render_plan_path=render_plan_path,
    )


def ensure_movie_bible(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    bible_path = project_dir / "movie" / "bible.json"
    if bible_path.exists():
        return bible_path
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    render_plan = _read_json(render_plan_path)
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except (FileNotFoundError, IsADirectoryError):
        manifest = {"actors": [], "locations": []}
    bible = _legacy_bible_from_render_plan(render_plan, manifest, project_dir=project_dir)
    bible_path.write_text(json.dumps(bible, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bible_path


def regenerate_movie_bible(project_dir: Path, *, planner) -> Path:
    project_dir = Path(project_dir)
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    render_plan = _read_json(render_plan_path)
    request = _movie_input_from_project(project_dir, render_plan)
    story_arch = planner.generate_story_arch(
        title=request.name,
        source_type=request.source_type,
        story_text=request.story_text,
        desired_length=float(request.desired_length),
    )
    bible = generate_movie_bible(planner=planner, request=request, story_arch=story_arch, config=request.config)
    movie_dir = project_dir / "movie"
    movie_dir.mkdir(parents=True, exist_ok=True)
    (movie_dir / "story_arch.json").write_text(json.dumps(asdict(story_arch), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bible_path = movie_dir / "bible.json"
    bible_path.write_text(json.dumps(_bible_dict(bible), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bible_path


def ensure_movie_render_plan_matches_bible(project_dir: Path) -> Path:
    render_plan_path = Path(project_dir) / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    return render_plan_path


def ensure_movie_screenplay(project_dir: Path, *, force: bool = False) -> Path:
    project_dir = Path(project_dir)
    screenplay_path = project_dir / "movie" / "screenplay.json"
    if screenplay_path.exists() and not force:
        return screenplay_path
    bible = movie_bible_from_dict(_read_json(ensure_movie_bible(project_dir)))
    render_plan = _read_json(project_dir / "movie" / "render_plan.json")
    request = _movie_input_from_project(project_dir, render_plan)
    story_design = movie_story_design_from_dict(_read_json(ensure_movie_story_design(project_dir)), fallback_title=request.name, bible=bible)
    screenplay = build_movie_screenplay_fallback(request=request, bible=bible, story_arch=bible.story_arch, story_design=story_design, config=request.config)
    screenplay_path.write_text(json.dumps(movie_screenplay_to_dict(screenplay), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (project_dir / "movie" / "screenplay.md").write_text(movie_screenplay_to_markdown(screenplay), encoding="utf-8")
    return screenplay_path


def ensure_movie_story_design(project_dir: Path, *, force: bool = False) -> Path:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "story_design.json"
    if path.exists() and not force:
        return path
    bible = movie_bible_from_dict(_read_json(ensure_movie_bible(project_dir)))
    render_plan = _read_json(project_dir / "movie" / "render_plan.json")
    request = _movie_input_from_project(project_dir, render_plan)
    design = build_movie_story_design_fallback(request=request, bible=bible, story_arch=bible.story_arch, config=request.config)
    path.write_text(json.dumps(movie_story_design_to_dict(design), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def ensure_movie_narrative_plan(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "narrative_plan.json"
    if path.exists():
        return path
    bible = movie_bible_from_dict(_read_json(ensure_movie_bible(project_dir)))
    screenplay = movie_screenplay_from_dict(
        _read_json(ensure_movie_screenplay(project_dir)),
        fallback_title=project_dir.name,
        source_type=_movie_source_metadata(project_dir, {})[0],
        bible=bible,
    )
    narrative = build_movie_narrative_plan_fallback(screenplay=screenplay)
    path.write_text(json.dumps(movie_narrative_plan_to_dict(narrative), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def ensure_movie_scene_cards(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "scene_cards.json"
    if path.exists():
        return path
    bible = movie_bible_from_dict(_read_json(ensure_movie_bible(project_dir)))
    screenplay = movie_screenplay_from_dict(
        _read_json(ensure_movie_screenplay(project_dir)),
        fallback_title=project_dir.name,
        source_type=_movie_source_metadata(project_dir, {})[0],
        bible=bible,
    )
    shots = _shots_from_project_render_plan(project_dir)
    cards = build_movie_scene_cards(screenplay=screenplay, shots=shots)
    path.write_text(json.dumps(movie_scene_cards_to_dict(cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def ensure_movie_shot_cards(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "shot_cards.json"
    if path.exists():
        return path
    scene_cards_path = ensure_movie_scene_cards(project_dir)
    scene_cards = movie_scene_cards_from_dict(_read_json(scene_cards_path))
    cards = build_movie_shot_cards(shots=_shots_from_project_render_plan(project_dir), scene_cards=scene_cards)
    path.write_text(json.dumps(movie_shot_cards_to_dict(cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def ensure_movie_continuity_plan(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    continuity_path = project_dir / "movie" / "continuity_plan.json"
    if continuity_path.exists():
        return continuity_path
    bible_path = ensure_movie_bible(project_dir)
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    bible = movie_bible_from_dict(_read_json(bible_path))
    render_plan = _read_json(render_plan_path)
    shots = tuple(_shot_from_render_plan(item, index) for index, item in enumerate(render_plan.get("shots") or [], start=1) if isinstance(item, dict))
    continuity = MovieContinuityPlan.fallback(bible=bible, shots=shots)
    continuity_path.write_text(json.dumps(movie_continuity_plan_to_dict(continuity), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return continuity_path


def _shots_from_project_render_plan(project_dir: Path) -> tuple[CinematicShot, ...]:
    render_plan = _read_json(Path(project_dir) / "movie" / "render_plan.json")
    return tuple(_shot_from_render_plan(item, index) for index, item in enumerate(render_plan.get("shots") or [], start=1) if isinstance(item, dict))


def _movie_source_metadata(project_dir: Path, render_plan: dict) -> tuple[str, str, float]:
    metadata_path = project_dir / ".studio" / "project.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        movie = metadata.get("movie") if isinstance(metadata.get("movie"), dict) else {}
        source_type = str(movie.get("source_type") or "short_story")
        story_text = str(movie.get("story_text") or "").strip()
        desired_length = float(movie.get("desired_length") or render_plan.get("duration_seconds") or 1)
        if story_text:
            return source_type, story_text, desired_length
    shots = render_plan.get("shots") or []
    story_text = "\n".join(str(shot.get("description") or shot.get("action") or "") for shot in shots if isinstance(shot, dict)).strip()
    return "short_story", story_text or str(render_plan.get("title") or project_dir.name), float(render_plan.get("duration_seconds") or len(shots) or 1)


def _movie_input_from_project(project_dir: Path, render_plan: dict) -> MovieInput:
    source_type, story_text, desired_length = _movie_source_metadata(project_dir, render_plan)
    config_path = project_dir / "config.json"
    try:
        config = _read_json(config_path)
    except (FileNotFoundError, IsADirectoryError):
        config = {}
    return MovieInput(
        name=str(render_plan.get("title") or project_dir.name),
        source_type=source_type,
        story_text=story_text,
        desired_length=desired_length,
        width=int((render_plan.get("resolution") or {}).get("width") or 1280),
        height=int((render_plan.get("resolution") or {}).get("height") or 704),
        config=config,
    )


def write_movie_reference_manifest_from_bible(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    bible_path = ensure_movie_bible(project_dir)
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    bible = _read_json(bible_path)
    try:
        existing = _read_json(manifest_path)
    except (FileNotFoundError, IsADirectoryError):
        existing = {}
    existing_actors = {str(actor.get("id")): actor for actor in existing.get("actors") or [] if isinstance(actor, dict)}
    existing_locations = {str(location.get("id")): location for location in existing.get("locations") or [] if isinstance(location, dict)}
    manifest = dict(existing)
    manifest["project_type"] = "movie"
    manifest["actors"] = [_manifest_actor(actor, existing_actors.get(str(actor.get("id"))) or {}) for actor in bible.get("actors") or [] if isinstance(actor, dict)]
    manifest["locations"] = [
        _manifest_location(location, existing_locations.get(str(location.get("id"))) or {}) for location in bible.get("locations") or [] if isinstance(location, dict)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _shot_from_render_plan(shot: dict, index: int) -> CinematicShot:
    references = shot.get("reference_ids") if isinstance(shot.get("reference_ids"), dict) else {}
    actor_ids = shot.get("actor_ids") or references.get("actors") or []
    return CinematicShot(
        shot_id=str(shot.get("shot_id") or f"shot_{index:04}"),
        description=str(shot.get("description") or shot.get("action") or f"Shot {index}").strip(),
        duration_seconds=float(shot.get("duration_seconds") or shot.get("duration") or 1),
        camera=str(shot.get("camera") or "").strip(),
        action=str(shot.get("action") or shot.get("description") or "").strip(),
        expression=str(shot.get("acting") or shot.get("expression") or "").strip(),
        location=str(shot.get("location") or "").strip(),
        dialogue=str(shot.get("dialogue") or "").strip(),
        actor_ids=tuple(str(item).strip() for item in actor_ids if str(item).strip()),
        location_id=str(shot.get("location_id") or references.get("location") or "").strip(),
        continuity_notes=str(shot.get("continuity_notes") or "").strip(),
        story_state_before=str(shot.get("story_state_before") or "").strip(),
        story_state_after=str(shot.get("story_state_after") or "").strip(),
        cause_from_previous=str(shot.get("cause_from_previous") or "").strip(),
        narrative_purpose=str(shot.get("narrative_purpose") or "").strip(),
        conflict_or_tension=str(shot.get("conflict_or_tension") or "").strip(),
        turning_point=str(shot.get("turning_point") or "").strip(),
        sets_up_next=str(shot.get("sets_up_next") or "").strip(),
        transition_from_previous=transition_from_previous(shot.get("transition_from_previous")),
    )


def _manifest_actor(actor: dict, current: dict) -> dict:
    visual_description = str(actor.get("visual_description") or actor.get("name") or actor.get("id") or "").strip()
    prompt = build_movie_actor_reference_prompt(str(actor.get("name") or actor.get("id")), visual_description)
    return {
        **current,
        "id": actor.get("id"),
        "name": actor.get("name") or actor.get("id"),
        "role": actor.get("role") or "",
        "visual_description": visual_description,
        "image_prompt": prompt,
        "prompt": prompt,
        "status": current.get("status") or "required",
        "msr_sheet_path": current.get("msr_sheet_path") or "",
    }


def _manifest_location(location: dict, current: dict) -> dict:
    visual_description = str(location.get("visual_description") or location.get("name") or location.get("id") or "").strip()
    image_prompt = str(location.get("image_prompt") or visual_description).strip()
    return {
        **current,
        "id": location.get("id"),
        "name": location.get("name") or location.get("id"),
        "visual_description": visual_description,
        "image_prompt": image_prompt,
        "prompt": image_prompt,
        "status": current.get("status") or "required",
        "msr_sheet_path": current.get("msr_sheet_path") or "",
    }


def _legacy_bible_from_render_plan(render_plan: dict, manifest: dict, *, project_dir: Path) -> dict:
    resolution = render_plan.get("resolution") or {}
    bible = {
        "title": render_plan.get("title") or project_dir.name,
        "premise": render_plan.get("premise") or "",
        "story_arch": {
            "title": render_plan.get("title") or project_dir.name,
            "premise": render_plan.get("premise") or "",
            "beats": [str(shot.get("description") or shot.get("action") or "") for shot in render_plan.get("shots") or [] if isinstance(shot, dict)],
        },
        "actors": [
            {
                "id": actor.get("id"),
                "name": actor.get("name") or actor.get("id"),
                "role": actor.get("role") or "",
                "visual_description": actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or actor.get("name") or actor.get("id"),
            }
            for actor in manifest.get("actors") or []
            if isinstance(actor, dict) and actor.get("id")
        ],
        "locations": [
            {
                "id": location.get("id"),
                "name": location.get("name") or location.get("id"),
                "visual_description": location.get("visual_description") or location.get("image_prompt") or location.get("prompt") or location.get("name") or location.get("id"),
            }
            for location in manifest.get("locations") or []
            if isinstance(location, dict) and location.get("id")
        ],
        "continuity": [{"id": "legacy_visual_continuity", "description": "Preserve existing actor and location references from the legacy movie manifest."}],
        "style_constraints": [],
        "runtime_constraints": {
            "width": int(resolution.get("width") or 1280),
            "height": int(resolution.get("height") or 704),
            "max_scene_actors": 4,
        },
    }
    if not bible["actors"]:
        bible["actors"] = [{"id": "main_character", "name": "Main Character", "role": "lead", "visual_description": "Main Character"}]
    if not bible["locations"]:
        bible["locations"] = [{"id": "primary_location", "name": "Primary Location", "visual_description": "Primary Location"}]
    return bible


_read_json = read_json_object
