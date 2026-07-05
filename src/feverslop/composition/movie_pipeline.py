from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.adapters.movie_references import LocalMovieImageBackend
from feverslop.adapters.movie_visual import LocalMovieVisualAdapter
from feverslop.application.movie_artifacts import (
    ensure_movie_bible as ensure_movie_bible_artifact,
    ensure_movie_continuity_plan as ensure_movie_continuity_plan_artifact,
    ensure_movie_narrative_plan as ensure_movie_narrative_plan_artifact,
    ensure_movie_planning_artifacts,
    ensure_movie_render_plan_matches_bible as ensure_movie_render_plan_matches_bible_artifact,
    ensure_movie_scene_cards as ensure_movie_scene_cards_artifact,
    ensure_movie_screenplay as ensure_movie_screenplay_artifact,
    ensure_movie_shot_cards as ensure_movie_shot_cards_artifact,
    write_movie_reference_manifest_from_bible as write_movie_reference_manifest_from_bible_artifact,
)
from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts
from feverslop.application.movie_references import MovieReferenceSheetGenerator
from feverslop.path_utils import coerce_local_path
from feverslop.studio.job_service import (
    build_movie_reference_generator,
    build_movie_visual_adapter,
    mark_movie_reference_backend,
    movie_references_ready,
    movie_runtime_config,
    patch_movie_msr_workflow,
)


@dataclass(frozen=True)
class MoviePipelineResult:
    project_dir: Path
    bible_path: Path | None = None
    screenplay_path: Path | None = None
    narrative_plan_path: Path | None = None
    scene_cards_path: Path | None = None
    shot_cards_path: Path | None = None
    render_plan_path: Path | None = None
    continuity_plan_path: Path | None = None
    render_plan_msr_path: Path | None = None
    reference_manifest_path: Path | None = None
    final_video_path: Path | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run movie pipeline stages for an existing FeverSlop movie project.")
    parser.add_argument("project_dir", help="Movie project directory, for example projects/tm3")
    parser.add_argument("--reference-backend", choices=["comfyui", "local"], default=None)
    parser.add_argument("--render-backend", choices=["comfyui", "local"], default=None)
    parser.add_argument("--hero-workflow", default=None)
    parser.add_argument("--edit-workflow", default=None)
    parser.add_argument("--msr-workflow", default=None)
    parser.add_argument("--skip-movie-bible", action="store_true", help="Reuse existing movie/bible.json.")
    parser.add_argument("--skip-movie-screenplay", action="store_true", help="Reuse existing movie/screenplay.json.")
    parser.add_argument("--force-movie-screenplay", action="store_true", help="Regenerate movie/screenplay.json from project source/render plan.")
    parser.add_argument("--skip-movie-narrative", action="store_true", help="Reuse existing movie/narrative_plan.json.")
    parser.add_argument("--skip-movie-scene-cards", action="store_true", help="Reuse existing movie/scene_cards.json.")
    parser.add_argument("--skip-movie-shot-cards", action="store_true", help="Reuse existing movie/shot_cards.json.")
    parser.add_argument("--skip-movie-continuity", action="store_true", help="Reuse existing movie/continuity_plan.json.")
    parser.add_argument("--skip-movie-plan", action="store_true", help="Reuse existing movie/render_plan.json.")
    parser.add_argument("--skip-movie-references", action="store_true", help="Reuse existing movie reference manifest paths.")
    parser.add_argument("--skip-movie-msr-enrich", action="store_true", help="Reuse existing movie/render_plan_msr.json or render the plain plan.")
    parser.add_argument("--skip-movie-render", action="store_true", help="Stop after syncing/rendering movie references.")
    parser.add_argument("--force-movie-references", action="store_true", help="Render movie references even when manifest paths already exist.")
    parser.add_argument("--keyframe-mode", choices=["none", "start", "start-end"], default="none")
    parser.add_argument("--movie-video-workflow", choices=["msr", "msr-i2v-startframe"], default="msr")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in ("reference_backend", "render_backend", "hero_workflow", "edit_workflow", "msr_workflow", "movie_video_workflow", "keyframe_mode"):
        value = getattr(args, key, None)
        if value:
            config[key] = value
    return movie_runtime_config(config)


def run(args: argparse.Namespace) -> MoviePipelineResult:
    project_dir = coerce_local_path(args.project_dir).resolve()
    config = config_from_args(args)
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    render_plan_path = project_dir / "movie" / "render_plan.json"
    bible_path = project_dir / "movie" / "bible.json"
    screenplay_path = project_dir / "movie" / "screenplay.json"
    narrative_plan_path = project_dir / "movie" / "narrative_plan.json"
    scene_cards_path = project_dir / "movie" / "scene_cards.json"
    shot_cards_path = project_dir / "movie" / "shot_cards.json"
    continuity_plan_path = project_dir / "movie" / "continuity_plan.json"
    render_plan_msr_path = project_dir / "movie" / "render_plan_msr.json"

    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    if any(
        (
            args.skip_movie_bible,
            args.skip_movie_screenplay,
            args.skip_movie_narrative,
            args.skip_movie_scene_cards,
            args.skip_movie_shot_cards,
            args.skip_movie_continuity,
            args.skip_movie_plan,
        )
    ):
        if not args.skip_movie_bible:
            bible_path = ensure_movie_bible_artifact(project_dir)
        elif not bible_path.exists():
            raise FileNotFoundError(f"Movie bible not found: {bible_path}")
        if not args.skip_movie_screenplay:
            screenplay_path = ensure_movie_screenplay_artifact(project_dir, force=args.force_movie_screenplay)
        elif not screenplay_path.exists():
            raise FileNotFoundError(f"Movie screenplay not found: {screenplay_path}")
        if not args.skip_movie_narrative:
            narrative_plan_path = ensure_movie_narrative_plan_artifact(project_dir)
        elif not narrative_plan_path.exists():
            raise FileNotFoundError(f"Movie narrative plan not found: {narrative_plan_path}")
        if not args.skip_movie_scene_cards:
            scene_cards_path = ensure_movie_scene_cards_artifact(project_dir)
        elif not scene_cards_path.exists():
            raise FileNotFoundError(f"Movie scene cards not found: {scene_cards_path}")
        if not args.skip_movie_shot_cards:
            shot_cards_path = ensure_movie_shot_cards_artifact(project_dir)
        elif not shot_cards_path.exists():
            raise FileNotFoundError(f"Movie shot cards not found: {shot_cards_path}")
        if not args.skip_movie_continuity:
            continuity_plan_path = ensure_movie_continuity_plan_artifact(project_dir)
        elif not continuity_plan_path.exists():
            raise FileNotFoundError(f"Movie continuity plan not found: {continuity_plan_path}")
        if not args.skip_movie_plan:
            ensure_movie_render_plan_matches_bible_artifact(project_dir)
    else:
        planning = ensure_movie_planning_artifacts(project_dir, force_screenplay=args.force_movie_screenplay)
        bible_path = planning.bible_path
        screenplay_path = planning.screenplay_path
        narrative_plan_path = planning.narrative_plan_path
        scene_cards_path = planning.scene_cards_path
        shot_cards_path = planning.shot_cards_path
        continuity_plan_path = planning.continuity_plan_path
        render_plan_path = planning.render_plan_path
    if not manifest_path.exists():
        if args.skip_movie_references:
            raise FileNotFoundError(f"Movie reference manifest not found: {manifest_path}")
        manifest_path = write_movie_reference_manifest_from_bible_artifact(project_dir)

    write_movie_reference_manifest_from_bible_artifact(project_dir)
    reference_manifest_path: Path | None = manifest_path

    if not args.skip_movie_references:
        if args.force_movie_references or not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            reference_manifest_path = mark_movie_reference_backend(
                _build_reference_generator(config).generate(project_dir=project_dir),
                config["reference_backend"],
            )

    if not args.skip_movie_msr_enrich:
        render_plan_msr_path = enrich_movie_render_plan_with_msr_prompts(project_dir=project_dir, keyframe_mode=args.keyframe_mode)
    elif not render_plan_msr_path.exists():
        render_plan_msr_path = None

    final_video_path: Path | None = None
    if not args.skip_movie_render:
        if not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            raise ValueError("Movie references are not ready; run without --skip-movie-references first")
        workflow = patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        final_video_path = _build_visual_adapter(project_dir, config, workflow).render_movie(
            project_dir=project_dir,
            render_plan_path=render_plan_msr_path or render_plan_path,
        )

    return MoviePipelineResult(
        project_dir=project_dir,
        bible_path=bible_path,
        screenplay_path=screenplay_path,
        narrative_plan_path=narrative_plan_path,
        scene_cards_path=scene_cards_path,
        shot_cards_path=shot_cards_path,
        render_plan_path=render_plan_path,
        continuity_plan_path=continuity_plan_path,
        render_plan_msr_path=render_plan_msr_path,
        reference_manifest_path=reference_manifest_path,
        final_video_path=final_video_path,
    )


def _build_reference_generator(config: dict[str, Any]):
    if config["reference_backend"] == "local":
        local = LocalMovieImageBackend()
        return MovieReferenceSheetGenerator(backend=local, edit_backend=local)
    return build_movie_reference_generator(movie_config=config)


def _build_visual_adapter(project_dir: Path, config: dict[str, Any], workflow: dict):
    if config["render_backend"] == "local":
        return LocalMovieVisualAdapter()
    return build_movie_visual_adapter(project_dir, Path(config["msr_workflow"]), movie_config=config, workflow=workflow)


def main() -> None:
    result = run(build_arg_parser().parse_args())
    payload = {
        "project_dir": result.project_dir.as_posix(),
        "reference_manifest_path": result.reference_manifest_path.as_posix() if result.reference_manifest_path else "",
        "final_video_path": result.final_video_path.as_posix() if result.final_video_path else "",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
