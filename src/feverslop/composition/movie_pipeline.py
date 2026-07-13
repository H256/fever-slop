from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from feverslop.adapters.movie_references import LocalMovieImageBackend
from feverslop.adapters.movie_visual import LocalMovieVisualAdapter
from feverslop.application.movie_artifacts import (
    ensure_movie_bible as ensure_movie_bible_artifact,
    ensure_movie_continuity_plan as ensure_movie_continuity_plan_artifact,
    ensure_movie_narrative_plan as ensure_movie_narrative_plan_artifact,
    ensure_movie_planning_artifacts,
    regenerate_movie_bible as regenerate_movie_bible_artifact,
    ensure_movie_render_plan_matches_bible as ensure_movie_render_plan_matches_bible_artifact,
    ensure_movie_scene_cards as ensure_movie_scene_cards_artifact,
    ensure_movie_screenplay as ensure_movie_screenplay_artifact,
    ensure_movie_shot_cards as ensure_movie_shot_cards_artifact,
    ensure_movie_story_design as ensure_movie_story_design_artifact,
    write_movie_reference_manifest_from_bible as write_movie_reference_manifest_from_bible_artifact,
)
from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts
from feverslop.application.movie_references import MovieReferenceSheetGenerator
from feverslop.composition.movie_debug_workflows import write_movie_debug_workflows
from feverslop.path_utils import coerce_local_path
from feverslop.studio.job_service import (
    build_movie_reference_generator,
    build_movie_visual_adapter,
    mark_movie_reference_backend,
    movie_references_ready,
    movie_runtime_config,
    patch_movie_msr_workflow,
)


console = Console()

MOVIE_BASE_STAGE_TITLES = {
    "Movie planning",
    "Movie reference manifest",
    "Movie references",
}

MOVIE_I2V_EDIT_STAGE_TITLES = {
    *MOVIE_BASE_STAGE_TITLES,
    "Movie visual plan",
    "Movie I2V render plan",
    "Movie I2V/edit render",
    "Storyboard review page",
    "Movie complete",
}

MOVIE_STARTFRAME_DIRECTOR_STAGE_TITLES = {
    *MOVIE_BASE_STAGE_TITLES,
    "Movie identity ledger",
    "Movie startframe plan",
    "Movie director prompts",
    "Movie I2V render plan",
    "Movie startframe-director render",
    "Movie complete",
}

MOVIE_INGREDIENTS_STAGE_TITLES = {
    *MOVIE_BASE_STAGE_TITLES,
    "Movie Ingredients scene sheets",
    "Movie Ingredients render",
    "Movie complete",
}


class MovieStageProgressReporter:
    def __init__(self, stage_titles: set[str], *, console: Console = console):
        self.stage_titles = set(stage_titles)
        self.total = len(self.stage_titles)
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self.task_id = None
        self.completed = 0
        self.seen: set[str] = set()

    def __enter__(self) -> MovieStageProgressReporter:
        self.progress.__enter__()
        self.task_id = self.progress.add_task("Movie pipeline stages", total=self.total)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.progress.__exit__(exc_type, exc_value, traceback)

    def advance(self, title: str) -> None:
        if title not in self.stage_titles or title in self.seen or self.task_id is None:
            return
        self.seen.add(title)
        self.completed = min(self.total, self.completed + 1)
        self.progress.update(self.task_id, completed=self.completed)


_stage_progress: MovieStageProgressReporter | None = None


@dataclass(frozen=True)
class MoviePipelineResult:
    project_dir: Path
    bible_path: Path | None = None
    story_design_path: Path | None = None
    screenplay_path: Path | None = None
    narrative_plan_path: Path | None = None
    scene_cards_path: Path | None = None
    shot_cards_path: Path | None = None
    render_plan_path: Path | None = None
    continuity_plan_path: Path | None = None
    render_plan_msr_path: Path | None = None
    render_plan_ingredients_path: Path | None = None
    visual_plan_path: Path | None = None
    render_plan_i2v_path: Path | None = None
    identity_ledger_path: Path | None = None
    startframe_plan_path: Path | None = None
    startframe_director_prompts_path: Path | None = None
    startframe_validation_path: Path | None = None
    reference_manifest_path: Path | None = None
    final_video_path: Path | None = None
    debug_workflows_dir: Path | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run movie pipeline stages for an existing FeverSlop movie project.")
    parser.add_argument("project_dir", help="Movie project directory, for example projects/tm3")
    parser.add_argument("--reference-backend", choices=["comfyui", "local"], default=None)
    parser.add_argument("--render-backend", choices=["comfyui", "local"], default=None)
    parser.add_argument("--hero-workflow", default=None)
    parser.add_argument("--edit-workflow", default=None)
    parser.add_argument("--director-workflow", default=None)
    parser.add_argument("--startframe-director-backend", choices=["krea2", "ideogram"], default=None)
    parser.add_argument("--mask-workflow", default=None)
    parser.add_argument("--identity-repair-workflow", default=None)
    parser.add_argument("--detail-workflow", default=None)
    parser.add_argument("--startframe-comfyui-base-url", default=None)
    parser.add_argument("--startframe-validator-base-url", default=None)
    parser.add_argument("--startframe-validator-model", default=None)
    parser.add_argument("--msr-workflow", default=None)
    parser.add_argument("--msr-i2v-workflow", default=None)
    parser.add_argument("--i2v-workflow", default=None)
    parser.add_argument("--ingredients-workflow", default=None)
    parser.add_argument("--skip-movie-bible", action="store_true", help="Reuse existing movie/bible.json.")
    parser.add_argument("--force-movie-bible", action="store_true", help="Regenerate movie/bible.json from the configured movie planner.")
    parser.add_argument("--movie-planner-backend", choices=["llm", "deterministic", "local"], default=None)
    parser.add_argument("--skip-movie-story-design", action="store_true", help="Reuse existing movie/story_design.json.")
    parser.add_argument("--force-movie-story-design", action="store_true", help="Regenerate movie/story_design.json from project source/render plan.")
    parser.add_argument("--skip-movie-screenplay", action="store_true", help="Reuse existing movie/screenplay.json.")
    parser.add_argument("--force-movie-screenplay", action="store_true", help="Regenerate movie/screenplay.json from project source/render plan.")
    parser.add_argument("--skip-movie-narrative", action="store_true", help="Reuse existing movie/narrative_plan.json.")
    parser.add_argument("--skip-movie-scene-cards", action="store_true", help="Reuse existing movie/scene_cards.json.")
    parser.add_argument("--skip-movie-shot-cards", action="store_true", help="Reuse existing movie/shot_cards.json.")
    parser.add_argument("--skip-movie-continuity", action="store_true", help="Reuse existing movie/continuity_plan.json.")
    parser.add_argument("--skip-movie-plan", action="store_true", help="Reuse existing movie/render_plan.json.")
    parser.add_argument("--skip-movie-references", action="store_true", help="Reuse existing movie reference manifest paths.")
    parser.add_argument("--skip-movie-msr-enrich", action="store_true", help="Reuse existing movie/render_plan_msr.json or render the plain plan.")
    parser.add_argument("--skip-movie-ingredients-sheets", action="store_true", help="Skip Ingredients scene sheet composition.")
    parser.add_argument("--skip-movie-render", action="store_true", help="Stop after syncing/rendering movie references.")
    parser.add_argument("--force-movie-references", action="store_true", help="Render movie references even when manifest paths already exist.")
    parser.add_argument("--keyframe-mode", choices=["none", "start", "start-end"], default="none")
    parser.add_argument("--movie-video-workflow", choices=["msr", "msr-i2v-startframe", "i2v-edit", "startframe-director", "ingredients"], default="msr")
    parser.add_argument("--continuity-keyframes", choices=["none", "last-to-start"], default="none")
    parser.add_argument("--write-debug-workflows", action="store_true", help="Write patched movie MSR workflow JSONs without queueing ComfyUI.")
    parser.add_argument("--debug-workflows-dir", default=None, help="Directory for --write-debug-workflows output.")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in (
        "reference_backend",
        "render_backend",
        "hero_workflow",
        "edit_workflow",
        "director_workflow",
        "startframe_director_backend",
        "mask_workflow",
        "identity_repair_workflow",
        "detail_workflow",
        "startframe_comfyui_base_url",
        "startframe_validator_base_url",
        "startframe_validator_model",
        "msr_workflow",
        "msr_i2v_workflow",
        "i2v_workflow",
        "ingredients_workflow",
        "movie_video_workflow",
        "keyframe_mode",
        "continuity_keyframes",
    ):
        value = getattr(args, key, None)
        if value:
            config[key] = value
    if (
        config.get("movie_video_workflow") == "msr-i2v-startframe"
        and config.get("continuity_keyframes") == "last-to-start"
        and _looks_like_i2v_workflow(config.get("msr_workflow"))
        and not config.get("msr_i2v_workflow")
    ):
        config["msr_i2v_workflow"] = config.pop("msr_workflow")
    return movie_runtime_config(config)


def _looks_like_i2v_workflow(value: object) -> bool:
    return "i2v" in Path(str(value or "")).name.lower()


def run(args: argparse.Namespace) -> MoviePipelineResult:
    config = config_from_args(args)
    with MovieStageProgressReporter(stage_titles=_movie_stage_titles(config), console=console) as progress:
        global _stage_progress
        previous_progress = _stage_progress
        _stage_progress = progress
        try:
            return _run(args, config)
        finally:
            _stage_progress = previous_progress


def _movie_stage_titles(config: dict[str, Any]) -> set[str]:
    if config["movie_video_workflow"] == "i2v-edit":
        return MOVIE_I2V_EDIT_STAGE_TITLES
    if config["movie_video_workflow"] == "startframe-director":
        return MOVIE_STARTFRAME_DIRECTOR_STAGE_TITLES
    if config["movie_video_workflow"] == "ingredients":
        return MOVIE_INGREDIENTS_STAGE_TITLES
    return MOVIE_BASE_STAGE_TITLES


def _run(args: argparse.Namespace, config: dict[str, Any]) -> MoviePipelineResult:
    project_dir = coerce_local_path(args.project_dir).resolve()
    _log_stage("Movie pipeline", f"{project_dir} ({config['movie_video_workflow']})")
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    render_plan_path = project_dir / "movie" / "render_plan.json"
    bible_path = project_dir / "movie" / "bible.json"
    story_design_path = project_dir / "movie" / "story_design.json"
    screenplay_path = project_dir / "movie" / "screenplay.json"
    narrative_plan_path = project_dir / "movie" / "narrative_plan.json"
    scene_cards_path = project_dir / "movie" / "scene_cards.json"
    shot_cards_path = project_dir / "movie" / "shot_cards.json"
    continuity_plan_path = project_dir / "movie" / "continuity_plan.json"
    render_plan_msr_path = project_dir / "movie" / "render_plan_msr.json"
    render_plan_ingredients_path = project_dir / "movie" / "render_plan_ingredients.json"

    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    if any(
        (
            args.skip_movie_bible,
            args.skip_movie_story_design,
            args.skip_movie_screenplay,
            args.skip_movie_narrative,
            args.skip_movie_scene_cards,
            args.skip_movie_shot_cards,
            args.skip_movie_continuity,
            args.skip_movie_plan,
        )
    ):
        _log_stage("Movie planning", "using requested skip/force flags")
        if args.force_movie_bible:
            from feverslop.studio.project_repository import build_movie_planner

            planner_backend = args.movie_planner_backend or "llm"
            _log_stage("Movie bible", f"regenerating via {planner_backend}")
            bible_path = regenerate_movie_bible_artifact(project_dir, planner=build_movie_planner({"planner_backend": planner_backend}))
        elif not args.skip_movie_bible:
            _log_stage("Movie bible", "ensuring artifact")
            bible_path = ensure_movie_bible_artifact(project_dir)
        elif not bible_path.exists():
            raise FileNotFoundError(f"Movie bible not found: {bible_path}")
        if not args.skip_movie_story_design:
            _log_stage("Movie story design", "ensuring artifact")
            story_design_path = ensure_movie_story_design_artifact(project_dir, force=args.force_movie_story_design)
        elif not story_design_path.exists():
            raise FileNotFoundError(f"Movie story design not found: {story_design_path}")
        if not args.skip_movie_screenplay:
            _log_stage("Movie screenplay", "ensuring canonical screenplay")
            screenplay_path = ensure_movie_screenplay_artifact(project_dir, force=args.force_movie_screenplay)
        elif not screenplay_path.exists():
            raise FileNotFoundError(f"Movie screenplay not found: {screenplay_path}")
        if not args.skip_movie_narrative:
            _log_stage("Movie narrative", "ensuring narrative memory")
            narrative_plan_path = ensure_movie_narrative_plan_artifact(project_dir)
        elif not narrative_plan_path.exists():
            raise FileNotFoundError(f"Movie narrative plan not found: {narrative_plan_path}")
        if not args.skip_movie_scene_cards:
            _log_stage("Movie scene cards", "ensuring scene cards")
            scene_cards_path = ensure_movie_scene_cards_artifact(project_dir)
        elif not scene_cards_path.exists():
            raise FileNotFoundError(f"Movie scene cards not found: {scene_cards_path}")
        if not args.skip_movie_shot_cards:
            _log_stage("Movie shot cards", "ensuring shot cards")
            shot_cards_path = ensure_movie_shot_cards_artifact(project_dir)
        elif not shot_cards_path.exists():
            raise FileNotFoundError(f"Movie shot cards not found: {shot_cards_path}")
        if not args.skip_movie_continuity:
            _log_stage("Movie continuity", "ensuring continuity plan")
            continuity_plan_path = ensure_movie_continuity_plan_artifact(project_dir)
        elif not continuity_plan_path.exists():
            raise FileNotFoundError(f"Movie continuity plan not found: {continuity_plan_path}")
        if not args.skip_movie_plan:
            _log_stage("Movie render plan", "syncing render plan with bible")
            ensure_movie_render_plan_matches_bible_artifact(project_dir)
    else:
        if args.force_movie_bible:
            from feverslop.studio.project_repository import build_movie_planner

            planner_backend = args.movie_planner_backend or "llm"
            _log_stage("Movie bible", f"regenerating via {planner_backend}")
            bible_path = regenerate_movie_bible_artifact(project_dir, planner=build_movie_planner({"planner_backend": planner_backend}))
        _log_stage("Movie planning", "ensuring bible, screenplay, cards, continuity, and render plan")
        planning = ensure_movie_planning_artifacts(project_dir, force_screenplay=args.force_movie_screenplay, force_story_design=args.force_movie_story_design)
        bible_path = planning.bible_path
        story_design_path = planning.story_design_path
        screenplay_path = planning.screenplay_path
        narrative_plan_path = planning.narrative_plan_path
        scene_cards_path = planning.scene_cards_path
        shot_cards_path = planning.shot_cards_path
        continuity_plan_path = planning.continuity_plan_path
        render_plan_path = planning.render_plan_path
    if not manifest_path.exists():
        if args.skip_movie_references:
            raise FileNotFoundError(f"Movie reference manifest not found: {manifest_path}")
        _log_stage("Movie reference manifest", "creating from bible")
        manifest_path = write_movie_reference_manifest_from_bible_artifact(project_dir)

    _log_stage("Movie reference manifest", "syncing actor/location ids")
    write_movie_reference_manifest_from_bible_artifact(project_dir)
    reference_manifest_path: Path | None = manifest_path

    if not args.skip_movie_references:
        if args.force_movie_references or not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            _log_stage("Movie references", f"rendering via {config['reference_backend']}")
            reference_manifest_path = mark_movie_reference_backend(
                _build_reference_generator(config).generate(project_dir=project_dir),
                config["reference_backend"],
            )
        else:
            _log_stage("Movie references", "ready; reusing existing sheets")
    else:
        _log_stage("Movie references", "skipped; reusing manifest paths")

    if config["movie_video_workflow"] == "startframe-director":
        from feverslop.adapters.startframe_director_visual import LocalStartframeDirectorVisualAdapter
        from feverslop.application.startframe_director_prompts import build_startframe_director_prompts
        from feverslop.application.startframe_i2v_render_plan import write_startframe_i2v_render_plan
        from feverslop.application.startframe_identity import build_startframe_identity_ledger
        from feverslop.application.startframe_plan import build_startframe_plan
        from feverslop.application.startframe_validation import write_local_startframe_validation

        _log_stage("Movie identity ledger", "deriving face, body, wardrobe, and reference contracts")
        identity_ledger_path = build_startframe_identity_ledger(project_dir=project_dir)
        _log_stage("Movie startframe plan", "deriving shot contracts, bboxes, and continuity requirements")
        startframe_plan_path = build_startframe_plan(project_dir=project_dir)
        _log_stage("Movie director prompts", f"writing {config['startframe_director_backend']} director prompts")
        startframe_director_prompts_path = build_startframe_director_prompts(
            project_dir=project_dir,
            director_backend=config["startframe_director_backend"],
        )
        _log_stage("Movie I2V render plan", "writing classic I2V handoff plan")
        render_plan_i2v_path = write_startframe_i2v_render_plan(project_dir=project_dir)
        final_video_path: Path | None = None
        startframe_validation_path = project_dir / "movie" / "startframe_validation.json"
        startframe_debug_workflows_dir = _startframe_debug_workflows_dir(project_dir, args)
        if startframe_debug_workflows_dir is not None:
            config = {
                **config,
                "startframe_write_debug_workflows": True,
                "startframe_debug_workflows_dir": startframe_debug_workflows_dir,
            }
        if not args.skip_movie_render:
            _log_stage("Movie startframe-director render", f"rendering via {config['render_backend']}")
            if config["render_backend"] != "local":
                adapter = _build_startframe_director_visual_adapter(project_dir, config)
            else:
                adapter = LocalStartframeDirectorVisualAdapter()
            final_video_path = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=render_plan_i2v_path,
                on_startframe_step=lambda event: _log_stage(
                    "Movie startframe-director render",
                    _format_startframe_step(event),
                ),
                on_clip_rendered=lambda completed, total, scene_number: _log_stage(
                    "Movie I2V clip",
                    f"rendered {completed}/{total}: scene {scene_number}",
                ),
            )
            startframe_validation_path = write_local_startframe_validation(project_dir=project_dir)
        else:
            _log_stage("Movie startframe-director render", "skipped")
        _log_stage("Movie complete", str(final_video_path or render_plan_i2v_path))
        return MoviePipelineResult(
            project_dir=project_dir,
            bible_path=bible_path,
            story_design_path=story_design_path,
            screenplay_path=screenplay_path,
            narrative_plan_path=narrative_plan_path,
            scene_cards_path=scene_cards_path,
            shot_cards_path=shot_cards_path,
            render_plan_path=render_plan_path,
            continuity_plan_path=continuity_plan_path,
            render_plan_i2v_path=render_plan_i2v_path,
            identity_ledger_path=identity_ledger_path,
            startframe_plan_path=startframe_plan_path,
            startframe_director_prompts_path=startframe_director_prompts_path,
            startframe_validation_path=startframe_validation_path if startframe_validation_path.exists() else None,
            reference_manifest_path=reference_manifest_path,
            final_video_path=final_video_path,
            debug_workflows_dir=startframe_debug_workflows_dir,
        )

    if config["movie_video_workflow"] == "i2v-edit":
        from feverslop.adapters.movie_i2v_visual import LocalMovieI2VEditVisualAdapter
        from feverslop.application.movie_i2v_render_plan import write_movie_i2v_render_plan
        from feverslop.application.movie_visual_plan import build_movie_visual_plan
        from feverslop.tools.movie_storyboard_page import generate_movie_storyboard_page

        _log_stage("Movie visual plan", "deriving scene views and character edit passes")
        visual_plan_path = build_movie_visual_plan(project_dir=project_dir)
        _log_stage("Movie I2V render plan", "writing classic I2V adapter plan")
        render_plan_i2v_path = write_movie_i2v_render_plan(project_dir=project_dir)
        final_video_path: Path | None = None
        if not args.skip_movie_render:
            _log_stage("Movie I2V/edit render", f"rendering via {config['render_backend']}")
            if config["render_backend"] != "local":
                adapter = _build_i2v_edit_visual_adapter(project_dir, config)
            else:
                adapter = LocalMovieI2VEditVisualAdapter()
            final_video_path = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=render_plan_i2v_path,
                on_startframe_step=lambda event: _log_stage(
                    "Movie startframe",
                    _format_startframe_step(event),
                ),
                on_clip_rendered=lambda completed, total, scene_number: _log_stage(
                    "Movie I2V clip",
                    f"rendered {completed}/{total}: scene {scene_number}",
                ),
            )
        else:
            _log_stage("Movie I2V/edit render", "skipped")
        _log_stage("Storyboard review page", "writing HTML review page")
        generate_movie_storyboard_page(project_dir=project_dir)
        _log_stage("Movie complete", str(final_video_path or render_plan_i2v_path))
        return MoviePipelineResult(
            project_dir=project_dir,
            bible_path=bible_path,
            story_design_path=story_design_path,
            screenplay_path=screenplay_path,
            narrative_plan_path=narrative_plan_path,
            scene_cards_path=scene_cards_path,
            shot_cards_path=shot_cards_path,
            render_plan_path=render_plan_path,
            continuity_plan_path=continuity_plan_path,
            visual_plan_path=visual_plan_path,
            render_plan_i2v_path=render_plan_i2v_path,
            reference_manifest_path=reference_manifest_path,
            final_video_path=final_video_path,
        )

    if config["movie_video_workflow"] == "ingredients":
        if not args.skip_movie_ingredients_sheets:
            from feverslop.application.movie_ingredients_sheets import enrich_movie_render_plan_with_ingredients_sheets
            _log_stage("Movie Ingredients scene sheets", "composing letterboxed scene reference sheets")
            render_plan_ingredients_path = enrich_movie_render_plan_with_ingredients_sheets(
                project_dir=project_dir,
                sheet_scale=config.get("ingredients_sheet_scale", 3.0),
            )
        elif not render_plan_ingredients_path.exists():
            render_plan_ingredients_path = None

        if render_plan_ingredients_path is None:
            raise FileNotFoundError(f"Movie ingredients render plan not found: {render_plan_ingredients_path}")

        ingredients_debug_workflows_dir = _ingredients_debug_workflows_dir(project_dir, args)
        final_video_path: Path | None = None
        if not args.skip_movie_render:
            _log_stage("Movie Ingredients render", f"rendering via {config['render_backend']}")
            if config["render_backend"] != "local":
                adapter = _build_ingredients_adapter(project_dir, config, debug_workflows_dir=ingredients_debug_workflows_dir)
            else:
                adapter = LocalMovieVisualAdapter()
            final_video_path = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=render_plan_ingredients_path,
                on_clip_rendered=lambda completed, total, scene_number: _log_stage(
                    "Movie Ingredients clip",
                    f"rendered {completed}/{total}: scene {scene_number}",
                ),
            )
        else:
            _log_stage("Movie Ingredients render", "skipped")
        _log_stage("Movie complete", str(final_video_path or render_plan_ingredients_path))
        return MoviePipelineResult(
            project_dir=project_dir,
            bible_path=bible_path,
            story_design_path=story_design_path,
            screenplay_path=screenplay_path,
            narrative_plan_path=narrative_plan_path,
            scene_cards_path=scene_cards_path,
            shot_cards_path=shot_cards_path,
            render_plan_path=render_plan_path,
            continuity_plan_path=continuity_plan_path,
            render_plan_ingredients_path=render_plan_ingredients_path,
            reference_manifest_path=reference_manifest_path,
            final_video_path=final_video_path,
            debug_workflows_dir=ingredients_debug_workflows_dir,
        )

    if not args.skip_movie_msr_enrich:
        render_plan_msr_path = enrich_movie_render_plan_with_msr_prompts(project_dir=project_dir, keyframe_mode=args.keyframe_mode)
    elif not render_plan_msr_path.exists():
        render_plan_msr_path = None

    if not args.skip_movie_ingredients_sheets:
        from feverslop.application.movie_ingredients_sheets import enrich_movie_render_plan_with_ingredients_sheets
        _log_stage("Movie Ingredients scene sheets", "composing letterboxed scene reference sheets")
        render_plan_ingredients_path = enrich_movie_render_plan_with_ingredients_sheets(
            project_dir=project_dir,
            sheet_scale=config.get("ingredients_sheet_scale", 3.0),
        )
    elif not render_plan_ingredients_path.exists():
        render_plan_ingredients_path = None

    debug_workflows_dir: Path | None = None
    if args.write_debug_workflows:
        if render_plan_msr_path is None:
            raise FileNotFoundError("Movie debug workflow export requires movie/render_plan_msr.json; run without --skip-movie-msr-enrich first")
        if not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            raise ValueError("Movie debug workflow export requires ready movie references; run without --skip-movie-references first")
        workflow = patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        debug_workflows_dir = write_movie_debug_workflows(
            project_dir=project_dir,
            render_plan_path=render_plan_msr_path,
            workflow_path=Path(config["msr_workflow"]),
            workflow=workflow,
            output_dir=coerce_local_path(args.debug_workflows_dir).resolve()
            if args.debug_workflows_dir
            else project_dir / "output" / "movie" / "ltx_msr_debug",
        )

    final_video_path: Path | None = None
    if not args.skip_movie_render:
        if not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            raise ValueError("Movie references are not ready; run without --skip-movie-references first")
        workflow = patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        final_video_path = _build_visual_adapter(project_dir, config, workflow).render_movie(
            project_dir=project_dir,
            render_plan_path=render_plan_msr_path or render_plan_path,
            continuity_keyframes=config["continuity_keyframes"],
            on_clip_rendered=lambda completed, total, scene_number: print(
                f"Rendered movie clip {completed}/{total}: scene {scene_number}"
            ),
        )

    return MoviePipelineResult(
        project_dir=project_dir,
        bible_path=bible_path,
        story_design_path=story_design_path,
        screenplay_path=screenplay_path,
        narrative_plan_path=narrative_plan_path,
        scene_cards_path=scene_cards_path,
        shot_cards_path=shot_cards_path,
        render_plan_path=render_plan_path,
        continuity_plan_path=continuity_plan_path,
        render_plan_msr_path=render_plan_msr_path,
        reference_manifest_path=reference_manifest_path,
        final_video_path=final_video_path,
        debug_workflows_dir=debug_workflows_dir,
    )


def _log_stage(title: str, detail: str = "") -> None:
    if _stage_progress is not None:
        _stage_progress.advance(title)
    message = f"[bold cyan]{title}[/bold cyan]"
    if detail:
        message = f"{message}: {detail}"
    console.print(message)


def _format_startframe_step(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or "step")
    completed = int(event.get("completed") or 0)
    total = int(event.get("total") or 0)
    scene = int(event.get("scene") or 0)
    actor_id = str(event.get("actor_id") or "").strip()
    actor_suffix = f" actor {actor_id}" if actor_id else ""
    return f"rendered {kind} {completed}/{total}: scene {scene}{actor_suffix}"


def _build_reference_generator(config: dict[str, Any]):
    if config["reference_backend"] == "local":
        local = LocalMovieImageBackend()
        return MovieReferenceSheetGenerator(backend=local, edit_backend=local)
    return build_movie_reference_generator(movie_config=config)


def _build_visual_adapter(project_dir: Path, config: dict[str, Any], workflow: dict):
    if config["render_backend"] == "local":
        return LocalMovieVisualAdapter()
    i2v_workflow = patch_movie_msr_workflow(template_path=Path(config["msr_i2v_workflow"])) if config.get("msr_i2v_workflow") else None
    return build_movie_visual_adapter(
        project_dir,
        Path(config["msr_workflow"]),
        movie_config=config,
        workflow=workflow,
        i2v_workflow=i2v_workflow,
    )


def _build_ingredients_adapter(project_dir: Path, config: dict[str, Any], *, debug_workflows_dir: Path | None = None):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_ingredients_video_backend import ComfyUIIngredientsVideoRenderBackend
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.movie_ingredients_visual import ComfyUIMovieIngredientsVisualAdapter
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load("app_config.json")
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides)
    ltx_dir = project_dir / "output" / "movie" / "ltx_ingredients"
    backend = ComfyUIIngredientsVideoRenderBackend(
        client=client,
        workflow_path=config.get("ingredients_workflow", "workflows/video_ltxv_ingredients_v1.json"),
        output_dir=ltx_dir,
        project_dir=project_dir,
        model_resolver=model_resolver,
        debug_workflows_dir=debug_workflows_dir,
    )
    return ComfyUIMovieIngredientsVisualAdapter(backend=backend)


def _build_i2v_edit_visual_adapter(project_dir: Path, config: dict[str, Any]):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
    from feverslop.adapters.local_artifacts import JsonArtifactStore
    from feverslop.adapters.movie_edit_image_backend import MovieTwoRefEditImageBackend
    from feverslop.adapters.movie_i2v_visual import ComfyUIMovieI2VEditVisualAdapter
    from feverslop.adapters.video_postprocessor import VideoPostProcessor
    from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load("app_config.json")
    client = ComfyUIClient(
        base_url=str(config.get("startframe_comfyui_base_url") or app_config.comfyui.base_url),
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides)
    ltx_dir = project_dir / "output" / "movie" / "ltx_i2v"
    video_use_case = build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            workflow_path=config["i2v_workflow"],
            single_prompt_workflow_path=config["i2v_workflow"],
            output_dir=ltx_dir,
            video_pipeline="ltx_i2v",
        )
    )
    return ComfyUIMovieI2VEditVisualAdapter(
        base_image_backend=ComfyUIImageBackend(
            client=client,
            workflow_path=config["hero_workflow"],
            output_dir=project_dir / "output" / "movie" / "storyboard" / "base",
            model_resolver=model_resolver,
        ),
        edit_backend=MovieTwoRefEditImageBackend(
            client=client,
            workflow_path=config["edit_workflow"],
            model_resolver=model_resolver,
        ),
        artifact_store=JsonArtifactStore(),
        video_use_case=video_use_case,
        workflow_path=Path(config["hero_workflow"]),
        edit_workflow_path=Path(config["edit_workflow"]),
        i2v_workflow_path=Path(config["i2v_workflow"]),
        postprocessor=VideoPostProcessor(),
    )


def _build_startframe_director_visual_adapter(project_dir: Path, config: dict[str, Any]):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.gemma4_startframe_validator import Gemma4StartframeValidator
    from feverslop.adapters.movie_workflow import MovieWorkflowPatcher
    from feverslop.adapters.startframe_director_comfyui import ComfyUIStartframeDirectorVisualAdapter
    from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load("app_config.json")
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    ltx_dir = project_dir / "output" / "movie" / "ltx_startframe_director"
    i2v_workflow_path = _write_startframe_i2v_empty_audio_workflow(
        project_dir=project_dir,
        workflow_path=Path(config["i2v_workflow"]),
        patcher=MovieWorkflowPatcher(),
    )
    video_use_case = build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            workflow_path=i2v_workflow_path,
            single_prompt_workflow_path=i2v_workflow_path,
            output_dir=ltx_dir,
            video_pipeline="ltx_i2v",
            debug_workflows_dir=config.get("startframe_debug_workflows_dir")
            if config.get("startframe_write_debug_workflows")
            else None,
        )
    )
    return ComfyUIStartframeDirectorVisualAdapter(
        client=client,
        director_workflow_path=config["director_workflow"],
        mask_workflow_path=config["mask_workflow"],
        identity_repair_workflow_path=config["identity_repair_workflow"],
        detail_workflow_path=config["detail_workflow"],
        i2v_workflow_path=i2v_workflow_path,
        video_use_case=video_use_case,
        validator=Gemma4StartframeValidator(
            base_url=config["startframe_validator_base_url"],
            model=config["startframe_validator_model"],
        ),
        debug_workflows_dir=config.get("startframe_debug_workflows_dir")
        if config.get("startframe_write_debug_workflows")
        else None,
    )


def _startframe_debug_workflows_dir(project_dir: Path, args: argparse.Namespace) -> Path | None:
    if not bool(getattr(args, "write_debug_workflows", False)):
        return None
    raw = getattr(args, "debug_workflows_dir", None)
    if raw:
        return coerce_local_path(raw).resolve()
    return project_dir / "output" / "movie" / "startframes" / "debug_workflows"


def _ingredients_debug_workflows_dir(project_dir: Path, args: argparse.Namespace) -> Path | None:
    if not bool(getattr(args, "write_debug_workflows", False)):
        return None
    raw = getattr(args, "debug_workflows_dir", None)
    if raw:
        return coerce_local_path(raw).resolve()
    return project_dir / "output" / "movie" / "ltx_ingredients" / "debug_workflows"


def _write_startframe_i2v_empty_audio_workflow(*, project_dir: Path, workflow_path: Path, patcher) -> Path:
    workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8-sig"))
    stripped = patcher.strip_audio_inputs(workflow)
    output = project_dir / "output" / "movie" / "startframes" / "workflows" / "ltx_i2v_empty_audio.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stripped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


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
