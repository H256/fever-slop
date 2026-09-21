"""Movie artifact lifecycle for the canonical story plan (issue #1387).

The movie planning artifacts (bible, story design, screenplay, narrative
plan, scene/shot cards, continuity plan) are fingerprinted projections of
the authoritative ``StoryPlan(mode="narrative_film")`` persisted at
``movie/plan.json``. Each ``ensure_movie_*`` function checks the plan
fingerprint (via ``manifest_is_stale``) and reuses or regenerates the
artifact as a ``resume_cache`` or ``review_export`` projection derived
from the canonical plan.

Migration rules:
- An existing artifact without a projection manifest is a legacy file; it
  is never deleted or overwritten merely by the migration.
- An artifact with a fresh projection manifest is reused as-is.
- Only a stale projection (plan fingerprint or a recorded dependency
  changed) is regenerated from the canonical plan.
- ``movie_artifact_migration_report`` names the legacy/stale projections
  and the projections regenerated during the current ensure cycle.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from feverslop.application.movie import (
    MovieInput,
    bible_dict,
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
from feverslop.domain.artifact_hash import sha256_file
from feverslop.domain.movie import (
    CinematicShot,
    MovieAct,
    MovieCharacterArc,
    MovieContinuityPlan,
    MovieSceneBlueprint,
    MovieScreenplayArtifact,
    MovieSetupPayoff,
    MovieStoryDesign,
    MovieTurningPoint,
    story_plan_to_cinematic_shots,
    story_plan_to_scene_cards,
    story_plan_to_shot_cards,
    story_plan_to_screenplay_scenes,
)
from feverslop.domain.movie_utils import transition_from_previous
from feverslop.domain.story_plan import StoryBeat, StoryMode, StoryPlan
from feverslop.domain.story_plan_artifacts import (
    ArtifactClass,
    ArtifactDependency,
    RegenerationPolicy,
    StoryPlanArtifactManifest,
    manifest_is_stale,
    read_manifest,
    read_story_plan,
    write_manifest,
)
from feverslop.errors import FeverSlopDataError
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
    migration_report: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _MovieProjection:
    name: str
    filename: str
    artifact_class: ArtifactClass
    dependency_paths: tuple[str, ...]


_MOVIE_PROJECTIONS: dict[str, _MovieProjection] = {
    spec.name: spec
    for spec in (
        _MovieProjection("bible", "bible.json", ArtifactClass.resume_cache, ("movie/render_plan.json",)),
        _MovieProjection("story_design", "story_design.json", ArtifactClass.resume_cache, ("movie/render_plan.json",)),
        _MovieProjection("screenplay", "screenplay.json", ArtifactClass.resume_cache, ("movie/render_plan.json", ".studio/project.json", "config.json")),
        _MovieProjection("screenplay_review", "screenplay.md", ArtifactClass.review_export, ("movie/render_plan.json", ".studio/project.json", "config.json")),
        _MovieProjection("narrative_plan", "narrative_plan.json", ArtifactClass.resume_cache, ("movie/render_plan.json",)),
        _MovieProjection("scene_cards", "scene_cards.json", ArtifactClass.resume_cache, ()),
        _MovieProjection("shot_cards", "shot_cards.json", ArtifactClass.resume_cache, ()),
        _MovieProjection("continuity_plan", "continuity_plan.json", ArtifactClass.resume_cache, ("movie/render_plan.json",)),
    )
}

#: Projections overwritten from the canonical plan during the current
#: ensure cycle; consumed and cleared by ``movie_artifact_migration_report``.
_REGENERATED_PROJECTIONS: set[str] = set()


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
        migration_report=movie_artifact_migration_report(project_dir),
    )


def ensure_movie_bible(project_dir: Path) -> Path:
    return _ensure_projection(
        Path(project_dir),
        name="bible",
        build_plan=_write_bible_from_plan,
        build_legacy=_legacy_ensure_bible,
    )


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
    bible_path.write_text(json.dumps(bible_dict(bible), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bible_path


def ensure_movie_render_plan_matches_bible(project_dir: Path) -> Path:
    render_plan_path = Path(project_dir) / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    return render_plan_path


def ensure_movie_screenplay(project_dir: Path, *, force: bool = False) -> Path:
    return _ensure_projection(
        Path(project_dir),
        name="screenplay",
        build_plan=_write_screenplay_from_plan,
        build_legacy=lambda directory: _legacy_ensure_screenplay(directory, force=force),
        force=force,
        extra_names=("screenplay_review",),
    )


def ensure_movie_story_design(project_dir: Path, *, force: bool = False) -> Path:
    return _ensure_projection(
        Path(project_dir),
        name="story_design",
        build_plan=_write_story_design_from_plan,
        build_legacy=lambda directory: _legacy_ensure_story_design(directory, force=force),
        force=force,
    )


def ensure_movie_narrative_plan(project_dir: Path) -> Path:
    return _ensure_projection(
        Path(project_dir),
        name="narrative_plan",
        build_plan=_write_narrative_plan_from_plan,
        build_legacy=_legacy_ensure_narrative_plan,
    )


def ensure_movie_scene_cards(project_dir: Path) -> Path:
    return _ensure_projection(
        Path(project_dir),
        name="scene_cards",
        build_plan=_write_scene_cards_from_plan,
        build_legacy=_legacy_ensure_scene_cards,
    )


def ensure_movie_shot_cards(project_dir: Path) -> Path:
    return _ensure_projection(
        Path(project_dir),
        name="shot_cards",
        build_plan=_write_shot_cards_from_plan,
        build_legacy=_legacy_ensure_shot_cards,
    )


def ensure_movie_continuity_plan(project_dir: Path) -> Path:
    return _ensure_projection(
        Path(project_dir),
        name="continuity_plan",
        build_plan=_write_continuity_plan_from_plan,
        build_legacy=_legacy_ensure_continuity_plan,
    )


def movie_artifact_migration_report(project_dir: Path) -> dict[str, Any]:
    """Summarize the movie artifact projections for the plan migration.

    Names projections that are still legacy (on disk without a projection
    manifest), stale (manifest out of date with the plan fingerprint), or
    missing, plus the projections regenerated during the current ensure
    cycle. The regenerated list is consumed (and cleared) by this call.
    """
    project_dir = Path(project_dir)
    loaded = _load_authoritative_plan(project_dir)
    plan_present = loaded is not None
    plan_fingerprint: str | None = None
    if plan_present:
        plan_path, _ = _plan_paths(project_dir)
        plan_fingerprint = sha256_file(plan_path)
    projections: list[dict[str, Any]] = []
    stale_names: list[str] = []
    legacy_names: list[str] = []
    missing_names: list[str] = []
    for spec in _MOVIE_PROJECTIONS.values():
        artifact_path = project_dir / "movie" / spec.filename
        manifest = _read_projection_manifest(_projection_manifest_path(artifact_path))
        if not artifact_path.exists():
            status = "missing"
            missing_names.append(spec.name)
        elif manifest is None:
            status = "legacy"
            legacy_names.append(spec.name)
        elif not plan_present:
            status = "fresh"
        elif _projection_is_stale(manifest, project_dir=project_dir, spec=spec):
            status = "stale"
            stale_names.append(spec.name)
        else:
            status = "fresh"
        projections.append(
            {
                "name": spec.name,
                "path": f"movie/{spec.filename}",
                "artifact_class": spec.artifact_class.value,
                "status": status,
            }
        )
    report = {
        "plan_present": plan_present,
        "plan_fingerprint": plan_fingerprint,
        "projections": projections,
        "stale": stale_names,
        "legacy": legacy_names,
        "missing": missing_names,
        "regenerated": sorted(_REGENERATED_PROJECTIONS),
    }
    _REGENERATED_PROJECTIONS.clear()
    return report


def _ensure_projection(
    project_dir: Path,
    *,
    name: str,
    build_plan: Callable[[Path, StoryPlan], None],
    build_legacy: Callable[[Path], None],
    force: bool = False,
    extra_names: tuple[str, ...] = (),
) -> Path:
    """Ensure one movie artifact projection of the canonical plan.

    With the authoritative plan present: an existing file without a
    projection manifest is preserved as-is (legacy), a fresh manifest is
    reused, and a stale manifest triggers regeneration from the plan.
    Without the plan, the legacy build path runs unchanged.
    """
    project_dir = Path(project_dir)
    spec = _MOVIE_PROJECTIONS[name]
    artifact_path = project_dir / "movie" / spec.filename
    loaded = _load_authoritative_plan(project_dir)
    if loaded is None:
        build_legacy(project_dir)
        return artifact_path
    plan, plan_manifest = loaded
    existed = artifact_path.exists()
    if existed and not force:
        manifest = _read_projection_manifest(_projection_manifest_path(artifact_path))
        if manifest is None:
            return artifact_path
        if not _projection_is_stale(manifest, project_dir=project_dir, spec=spec):
            return artifact_path
    _write_plan_projection(
        project_dir,
        spec=spec,
        plan=plan,
        plan_manifest=plan_manifest,
        build_plan=build_plan,
        extra_names=extra_names,
        overwrite=existed,
    )
    return artifact_path


def _write_plan_projection(
    project_dir: Path,
    *,
    spec: _MovieProjection,
    plan: StoryPlan,
    plan_manifest: StoryPlanArtifactManifest | None,
    build_plan: Callable[[Path, StoryPlan], None],
    extra_names: tuple[str, ...],
    overwrite: bool,
) -> None:
    build_plan(project_dir, plan)
    _write_projection_manifest(project_dir, spec=spec, plan_manifest=plan_manifest)
    for extra_name in extra_names:
        extra_spec = _MOVIE_PROJECTIONS[extra_name]
        extra_path = project_dir / "movie" / extra_spec.filename
        if extra_path.exists():
            _write_projection_manifest(project_dir, spec=extra_spec, plan_manifest=plan_manifest)
    if overwrite:
        _REGENERATED_PROJECTIONS.add(spec.name)


def _write_projection_manifest(
    project_dir: Path,
    *,
    spec: _MovieProjection,
    plan_manifest: StoryPlanArtifactManifest | None,
) -> None:
    plan_path, _ = _plan_paths(project_dir)
    manifest = StoryPlanArtifactManifest(
        artifact_class=spec.artifact_class,
        regeneration_policy=RegenerationPolicy.on_input_change,
        input_fingerprint=sha256_file(plan_path),
        dependencies=[
            ArtifactDependency(path=relative, sha256=digest)
            for relative, digest in sorted(_dependency_digests(project_dir, spec.dependency_paths).items())
        ],
        plan_fingerprint=plan_manifest.input_fingerprint if plan_manifest is not None else None,
    )
    write_manifest(_projection_manifest_path(project_dir / "movie" / spec.filename), manifest)


def _projection_is_stale(
    manifest: StoryPlanArtifactManifest,
    *,
    project_dir: Path,
    spec: _MovieProjection,
) -> bool:
    plan_path, _ = _plan_paths(project_dir)
    return manifest_is_stale(
        manifest,
        input_fingerprint=sha256_file(plan_path),
        dependency_digests=_dependency_digests(project_dir, spec.dependency_paths),
    )


def _load_authoritative_plan(project_dir: Path) -> tuple[StoryPlan, StoryPlanArtifactManifest | None] | None:
    """Load the authoritative movie plan; None when absent or unusable.

    A missing, malformed, or non-narrative-film plan falls back to the
    legacy build path so the migration never breaks an existing project.
    """
    plan_path, manifest_path = _plan_paths(project_dir)
    if not plan_path.is_file():
        return None
    try:
        plan = read_story_plan(plan_path)
    except (FeverSlopDataError, OSError, ValueError):
        return None
    if plan.mode is not StoryMode.narrative_film:
        return None
    manifest: StoryPlanArtifactManifest | None = None
    if manifest_path.is_file():
        try:
            manifest = read_manifest(manifest_path)
        except (FeverSlopDataError, OSError, ValueError):
            manifest = None
    return plan, manifest


def _plan_paths(project_dir: Path) -> tuple[Path, Path]:
    movie_dir = project_dir / "movie"
    return movie_dir / "plan.json", movie_dir / "plan.manifest.json"


def _projection_manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + ".manifest.json")


def _read_projection_manifest(manifest_path: Path) -> StoryPlanArtifactManifest | None:
    if not manifest_path.is_file():
        return None
    try:
        return read_manifest(manifest_path)
    except (FeverSlopDataError, OSError, ValueError):
        return None


def _dependency_digests(project_dir: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative in relative_paths:
        path = project_dir / relative
        if path.is_file():
            digests[relative] = sha256_file(path)
    return digests


def _try_read_json(path: Path) -> dict:
    try:
        return _read_json(path)
    except (FileNotFoundError, IsADirectoryError):
        return {}


def _write_bible_from_plan(project_dir: Path, plan: StoryPlan) -> None:
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    render_plan = _read_json(render_plan_path)
    bible = _bible_projection_dict(plan, render_plan, project_dir)
    (project_dir / "movie" / "bible.json").write_text(json.dumps(bible, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bible_projection_dict(plan: StoryPlan, render_plan: dict, project_dir: Path) -> dict:
    resolution = render_plan.get("resolution") or {}
    title = str(render_plan.get("title") or project_dir.name)
    premise = str(render_plan.get("premise") or "")
    bible = {
        "title": title,
        "premise": premise,
        "story_arch": {
            "title": title,
            "premise": premise,
            "beats": [beat.description for beat in plan.beats],
        },
        "actors": [
            {
                "id": character.id,
                "name": character.name,
                "role": "singer" if character.is_singer else "character",
                "visual_description": character.description or character.name,
            }
            for character in plan.characters
        ],
        "locations": [
            {
                "id": location.id,
                "name": location.name,
                "visual_description": location.description or location.name,
            }
            for location in plan.locations
        ],
        "continuity": [{"id": "story_plan_continuity", "description": "Preserve the canonical story plan's character and location references across shots."}],
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


def _write_story_design_from_plan(project_dir: Path, plan: StoryPlan) -> None:
    render_plan = _try_read_json(project_dir / "movie" / "render_plan.json")
    title = str(render_plan.get("title") or project_dir.name)
    premise = str(render_plan.get("premise") or "")
    beat_ids = tuple(beat.id for beat in plan.beats)
    max_actors = int(render_plan.get("max_scene_actors") or 4)
    total_duration = float(render_plan.get("duration_seconds") or len(plan.beats) or 1)
    expected_duration = max(1.0, total_duration / max(1, len(plan.beats)))
    design = MovieStoryDesign(
        title=title,
        premise=premise,
        theme="",
        act_structure=(
            MovieAct(act_id="act_1", title="Setup", purpose="Establish the premise, world, and central dramatic pressure.", scene_ids=_plan_beat_slice(beat_ids, 0, 1 / 3)),
            MovieAct(act_id="act_2", title="Confrontation", purpose="Escalate conflict and force irreversible choices.", scene_ids=_plan_beat_slice(beat_ids, 1 / 3, 2 / 3)),
            MovieAct(act_id="act_3", title="Resolution", purpose="Resolve the central dramatic pressure and leave the final emotional state.", scene_ids=_plan_beat_slice(beat_ids, 2 / 3, 1)),
        ),
        turning_points=_plan_turning_points(beat_ids),
        setup_payoff_threads=_plan_setup_payoffs(beat_ids),
        character_arcs=tuple(
            MovieCharacterArc(
                actor_id=arc.character_id,
                starting_state=arc.from_state,
                ending_state=arc.to_state,
            )
            for arc in plan.arcs
        ),
        scene_blueprint=tuple(
            _plan_scene_blueprint(beat, index, len(plan.beats), expected_duration, max_actors)
            for index, beat in enumerate(plan.beats, start=1)
        ),
    )
    (project_dir / "movie" / "story_design.json").write_text(json.dumps(movie_story_design_to_dict(design), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _plan_beat_slice(beat_ids: tuple[str, ...], start_ratio: float, end_ratio: float) -> tuple[str, ...]:
    if not beat_ids:
        return ()
    start = min(len(beat_ids) - 1, int(len(beat_ids) * start_ratio))
    end = max(start + 1, int(len(beat_ids) * end_ratio))
    return beat_ids[start:min(len(beat_ids), end)]


def _plan_turning_points(beat_ids: tuple[str, ...]) -> tuple[MovieTurningPoint, ...]:
    if not beat_ids:
        return ()
    return (
        MovieTurningPoint(id="inciting_turn", scene_id=beat_ids[0], description="The first scene turns the premise into immediate dramatic pressure."),
        MovieTurningPoint(id="final_turn", scene_id=beat_ids[-1], description="The final scene resolves the main emotional and narrative pressure."),
    )


def _plan_setup_payoffs(beat_ids: tuple[str, ...]) -> tuple[MovieSetupPayoff, ...]:
    if not beat_ids:
        return ()
    return (
        MovieSetupPayoff(id="central_thread", setup_scene_id=beat_ids[0], payoff_scene_id=beat_ids[-1], description="The opening dramatic pressure receives its final consequence by the end."),
    )


def _plan_scene_blueprint(beat: StoryBeat, index: int, total: int, expected_duration: float, max_actors: int) -> MovieSceneBlueprint:
    if index == 1:
        emotional_turn = "The emotional state moves from orientation into tension."
    elif index == total:
        emotional_turn = "The emotional state lands on the final consequence."
    else:
        emotional_turn = "The emotional state changes under escalating pressure."
    return MovieSceneBlueprint(
        scene_id=beat.id,
        purpose=beat.description,
        conflict=f"The visible goal is pressured by opposition or uncertainty in: {beat.description}",
        emotional_turn=emotional_turn,
        subtext="The scene carries its meaning through behavior, environment, and withheld emotion.",
        dialogue_function="No spoken dialogue; silence and action carry the dramatic information.",
        required_actors=tuple(beat.character_ids[:max_actors]),
        location_id=beat.location_id or "",
        expected_duration=expected_duration,
    )


def _write_screenplay_from_plan(project_dir: Path, plan: StoryPlan) -> None:
    render_plan = _try_read_json(project_dir / "movie" / "render_plan.json")
    source_type, _, _ = _movie_source_metadata(project_dir, render_plan)
    title = str(render_plan.get("title") or project_dir.name)
    screenplay = MovieScreenplayArtifact(
        title=title,
        source_type=source_type,
        dialogue_language="",
        scenes=story_plan_to_screenplay_scenes(plan),
    )
    (project_dir / "movie" / "screenplay.json").write_text(json.dumps(movie_screenplay_to_dict(screenplay), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (project_dir / "movie" / "screenplay.md").write_text(movie_screenplay_to_markdown(screenplay), encoding="utf-8")


def _write_narrative_plan_from_plan(project_dir: Path, plan: StoryPlan) -> None:
    render_plan = _try_read_json(project_dir / "movie" / "render_plan.json")
    title = str(render_plan.get("title") or project_dir.name)
    screenplay = MovieScreenplayArtifact(
        title=title,
        source_type="short_story",
        dialogue_language="",
        scenes=story_plan_to_screenplay_scenes(plan),
    )
    narrative = build_movie_narrative_plan_fallback(screenplay=screenplay)
    (project_dir / "movie" / "narrative_plan.json").write_text(json.dumps(movie_narrative_plan_to_dict(narrative), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_scene_cards_from_plan(project_dir: Path, plan: StoryPlan) -> None:
    cards = story_plan_to_scene_cards(plan)
    (project_dir / "movie" / "scene_cards.json").write_text(json.dumps(movie_scene_cards_to_dict(cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_shot_cards_from_plan(project_dir: Path, plan: StoryPlan) -> None:
    cards = story_plan_to_shot_cards(plan)
    (project_dir / "movie" / "shot_cards.json").write_text(json.dumps(movie_shot_cards_to_dict(cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_continuity_plan_from_plan(project_dir: Path, plan: StoryPlan) -> None:
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    render_plan = _read_json(render_plan_path)
    shots = story_plan_to_cinematic_shots(plan, render_plan)
    bible = movie_bible_from_dict(_bible_projection_dict(plan, render_plan, project_dir))
    continuity = MovieContinuityPlan.fallback(bible=bible, shots=shots)
    (project_dir / "movie" / "continuity_plan.json").write_text(json.dumps(movie_continuity_plan_to_dict(continuity), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _legacy_ensure_bible(project_dir: Path) -> None:
    project_dir = Path(project_dir)
    bible_path = project_dir / "movie" / "bible.json"
    if bible_path.exists():
        return
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


def _legacy_ensure_screenplay(project_dir: Path, *, force: bool) -> None:
    project_dir = Path(project_dir)
    screenplay_path = project_dir / "movie" / "screenplay.json"
    if screenplay_path.exists() and not force:
        return
    bible = movie_bible_from_dict(_read_json(ensure_movie_bible(project_dir)))
    render_plan = _read_json(project_dir / "movie" / "render_plan.json")
    request = _movie_input_from_project(project_dir, render_plan)
    story_design = movie_story_design_from_dict(_read_json(ensure_movie_story_design(project_dir)), fallback_title=request.name, bible=bible)
    screenplay = build_movie_screenplay_fallback(request=request, bible=bible, story_arch=bible.story_arch, story_design=story_design, config=request.config)
    screenplay_path.write_text(json.dumps(movie_screenplay_to_dict(screenplay), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (project_dir / "movie" / "screenplay.md").write_text(movie_screenplay_to_markdown(screenplay), encoding="utf-8")


def _legacy_ensure_story_design(project_dir: Path, *, force: bool) -> None:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "story_design.json"
    if path.exists() and not force:
        return
    bible = movie_bible_from_dict(_read_json(ensure_movie_bible(project_dir)))
    render_plan = _read_json(project_dir / "movie" / "render_plan.json")
    request = _movie_input_from_project(project_dir, render_plan)
    design = build_movie_story_design_fallback(request=request, bible=bible, story_arch=bible.story_arch, config=request.config)
    path.write_text(json.dumps(movie_story_design_to_dict(design), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _legacy_ensure_narrative_plan(project_dir: Path) -> None:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "narrative_plan.json"
    if path.exists():
        return
    bible = movie_bible_from_dict(_read_json(ensure_movie_bible(project_dir)))
    screenplay = movie_screenplay_from_dict(
        _read_json(ensure_movie_screenplay(project_dir)),
        fallback_title=project_dir.name,
        source_type=_movie_source_metadata(project_dir, {})[0],
        bible=bible,
    )
    narrative = build_movie_narrative_plan_fallback(screenplay=screenplay)
    path.write_text(json.dumps(movie_narrative_plan_to_dict(narrative), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _legacy_ensure_scene_cards(project_dir: Path) -> None:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "scene_cards.json"
    if path.exists():
        return
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


def _legacy_ensure_shot_cards(project_dir: Path) -> None:
    project_dir = Path(project_dir)
    path = project_dir / "movie" / "shot_cards.json"
    if path.exists():
        return
    scene_cards_path = ensure_movie_scene_cards(project_dir)
    scene_cards = movie_scene_cards_from_dict(_read_json(scene_cards_path))
    cards = build_movie_shot_cards(shots=_shots_from_project_render_plan(project_dir), scene_cards=scene_cards)
    path.write_text(json.dumps(movie_shot_cards_to_dict(cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _legacy_ensure_continuity_plan(project_dir: Path) -> None:
    project_dir = Path(project_dir)
    continuity_path = project_dir / "movie" / "continuity_plan.json"
    if continuity_path.exists():
        return
    bible_path = ensure_movie_bible(project_dir)
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    bible = movie_bible_from_dict(_read_json(bible_path))
    render_plan = _read_json(render_plan_path)
    shots = tuple(_shot_from_render_plan(item, index) for index, item in enumerate(render_plan.get("shots") or [], start=1) if isinstance(item, dict))
    continuity = MovieContinuityPlan.fallback(bible=bible, shots=shots)
    continuity_path.write_text(json.dumps(movie_continuity_plan_to_dict(continuity), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
