from __future__ import annotations

import argparse
import contextvars
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rich.console import Console

from feverslop.adapters.movie_references import LocalMovieImageBackend
from feverslop.adapters.movie_visual import LocalMovieVisualAdapter
from feverslop.application.openshot_exporter import export_render_plan_to_openshot
from feverslop.cli.movie_cli import build_movie_arg_parser, config_from_args
from feverslop.composition.movie_pipeline_jobs import (
    build_movie_reference_generator,
    build_movie_visual_adapter,
    mark_movie_reference_backend,
    movie_references_ready,
)
from feverslop.config.app_config import AppConfig
from feverslop.path_utils import coerce_local_path
from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.utils.rich_progress import build_progress

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

MOVIE_MINIMAX_STAGE_TITLES = {
    *MOVIE_BASE_STAGE_TITLES,
    "Movie MiniMax H3 prompts",
    "Movie MiniMax H3 render",
    "Movie complete",
}

MOVIE_MSR_STAGE_TITLES = {
    *MOVIE_BASE_STAGE_TITLES,
    "Movie MSR render",
    "Movie complete",
}


def _movie_uses_msr_reference_enrichment(movie_video_workflow: str) -> bool:
    """Return whether the selected movie mode needs LTX MSR enrichment."""
    return movie_video_workflow in {"msr", "msr-i2v-startframe"}


def _is_minimax_movie_workflow(movie_video_workflow: str) -> bool:
    return movie_video_workflow in {"minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"}


class MovieStageProgressReporter:
    def __init__(self, stage_titles: set[str], *, console: Console = console):
        self.stage_titles = set(stage_titles)
        self.total = len(self.stage_titles)
        self.progress = build_progress(console=console)
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


_stage_progress: contextvars.ContextVar[MovieStageProgressReporter | None] = contextvars.ContextVar(
    "movie_stage_progress",
    default=None,
)


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
    openshot_project_path: Path | None = None
    debug_workflows_dir: Path | None = None


def run(args: argparse.Namespace) -> MoviePipelineResult:
    config = config_from_args(args)
    if getattr(args, "stage", None) == "openshot_export":
        stage_titles = {"Movie OpenShot export"}
    else:
        stage_titles = _movie_stage_titles(config)
        if not getattr(args, "skip_openshot_export", False) and not getattr(args, "skip_movie_render", False):
            stage_titles.add("Movie OpenShot export")
    with MovieStageProgressReporter(stage_titles=stage_titles, console=console) as progress:
        progress_token = _stage_progress.set(progress)
        try:
            result = _run(args, config)
            if (
                getattr(args, "stage", None) != "openshot_export"
                and not getattr(args, "skip_openshot_export", False)
                and not getattr(args, "skip_movie_render", False)
            ):
                export_result = _run_movie_openshot_export_stage(args, result.project_dir)
                result = replace(result, openshot_project_path=export_result.openshot_project_path)
            return result
        finally:
            _stage_progress.reset(progress_token)


def _movie_stage_titles(config: dict[str, Any]) -> set[str]:
    if config["movie_video_workflow"] == "i2v-edit":
        return set(MOVIE_I2V_EDIT_STAGE_TITLES)
    if config["movie_video_workflow"] == "startframe-director":
        return set(MOVIE_STARTFRAME_DIRECTOR_STAGE_TITLES)
    if config["movie_video_workflow"] == "ingredients":
        return set(MOVIE_INGREDIENTS_STAGE_TITLES)
    if _is_minimax_movie_workflow(config["movie_video_workflow"]):
        return set(MOVIE_MINIMAX_STAGE_TITLES)
    return set(MOVIE_MSR_STAGE_TITLES)


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

    if getattr(args, "stage", None) == "openshot_export":
        return _run_movie_openshot_export_stage(args, project_dir)

    def ingredients_llm():
        try:
            app_config = AppConfig.load(args.app_config, required_keys=["llm"])
            from feverslop.adapters.openai_compatible_llm import (
                OpenAICompatibleLLMClient,
            )

            return OpenAICompatibleLLMClient(
                base_url=app_config.llm.base_url,
                api_key=app_config.llm.api_key,
                model=app_config.llm.model_for("creative"),
                temperature=app_config.llm.temperature,
                dspy_temperature=app_config.llm.dspy_temperature,
                max_tokens=app_config.llm.max_tokens,
                request_timeout_seconds=app_config.llm.request_timeout_seconds,
                max_concurrent_requests=app_config.llm.max_concurrent_requests,
            )
        except (OSError, ValueError):
            return None

    def report_ingredients_analysis(shot_id: str, references: list[dict[str, str]]) -> None:
        summary = ", ".join(f"{item['type']}:{item['id']}" for item in references)
        console.print(f"Ingredients image analysis: shot {shot_id}; {len(references)} references [{summary}]")

    def report_msr_analysis(shot_id: str, references: list[dict[str, str]]) -> None:
        summary = ", ".join(f"{item['type']}:{item['id']}" for item in references)
        console.print(f"MSR image analysis: shot {shot_id}; {len(references)} references [{summary}]")

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
        ),
    ):
        _log_stage("Movie planning", "using requested skip/force flags")
        if args.force_movie_bible:
            planner_backend = args.movie_planner_backend or "llm"
            _log_stage("Movie bible", f"regenerating via {planner_backend}")
            bible_path = _regenerate_movie_bible_artifact(project_dir, planner_backend, config["app_config_path"])
        elif not args.skip_movie_bible:
            _log_stage("Movie bible", "ensuring artifact")
            bible_path = _ensure_movie_bible_artifact(project_dir)
        elif not bible_path.exists():
            raise FileNotFoundError(f"Movie bible not found: {bible_path}")
        if not args.skip_movie_story_design:
            _log_stage("Movie story design", "ensuring artifact")
            story_design_path = _ensure_movie_story_design_artifact(project_dir, force=args.force_movie_story_design)
        elif not story_design_path.exists():
            raise FileNotFoundError(f"Movie story design not found: {story_design_path}")
        if not args.skip_movie_screenplay:
            _log_stage("Movie screenplay", "ensuring canonical screenplay")
            screenplay_path = _ensure_movie_screenplay_artifact(project_dir, force=args.force_movie_screenplay)
        elif not screenplay_path.exists():
            raise FileNotFoundError(f"Movie screenplay not found: {screenplay_path}")
        if not args.skip_movie_narrative:
            _log_stage("Movie narrative", "ensuring narrative memory")
            narrative_plan_path = _ensure_movie_narrative_plan_artifact(project_dir)
        elif not narrative_plan_path.exists():
            raise FileNotFoundError(f"Movie narrative plan not found: {narrative_plan_path}")
        if not args.skip_movie_scene_cards:
            _log_stage("Movie scene cards", "ensuring scene cards")
            scene_cards_path = _ensure_movie_scene_cards_artifact(project_dir)
        elif not scene_cards_path.exists():
            raise FileNotFoundError(f"Movie scene cards not found: {scene_cards_path}")
        if not args.skip_movie_shot_cards:
            _log_stage("Movie shot cards", "ensuring shot cards")
            shot_cards_path = _ensure_movie_shot_cards_artifact(project_dir)
        elif not shot_cards_path.exists():
            raise FileNotFoundError(f"Movie shot cards not found: {shot_cards_path}")
        if not args.skip_movie_continuity:
            _log_stage("Movie continuity", "ensuring continuity plan")
            continuity_plan_path = _ensure_movie_continuity_plan_artifact(project_dir)
        elif not continuity_plan_path.exists():
            raise FileNotFoundError(f"Movie continuity plan not found: {continuity_plan_path}")
        if not args.skip_movie_plan:
            _log_stage("Movie render plan", "syncing render plan with bible")
            _ensure_movie_render_plan_matches_bible_artifact(project_dir)
    else:
        if args.force_movie_bible:
            planner_backend = args.movie_planner_backend or "llm"
            _log_stage("Movie bible", f"regenerating via {planner_backend}")
            bible_path = _regenerate_movie_bible_artifact(project_dir, planner_backend, config["app_config_path"])
        _log_stage("Movie planning", "ensuring bible, screenplay, cards, continuity, and render plan")
        planning = _ensure_movie_planning_artifacts(project_dir, force_screenplay=args.force_movie_screenplay, force_story_design=args.force_movie_story_design)
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
        manifest_path = _write_movie_reference_manifest_from_bible_artifact(project_dir)

    _log_stage("Movie reference manifest", "syncing actor/location ids")
    _write_movie_reference_manifest_from_bible_artifact(project_dir)
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

    # --- startframe-director workflow ---
    if config["movie_video_workflow"] == "startframe-director":
        return _run_startframe_director_workflow(args, config, project_dir, bible_path, story_design_path,
            screenplay_path, narrative_plan_path, scene_cards_path, shot_cards_path,
            render_plan_path, continuity_plan_path, reference_manifest_path, render_plan_ingredients_path, manifest_path)

    # --- i2v-edit workflow ---
    if config["movie_video_workflow"] == "i2v-edit":
        return _run_i2v_edit_workflow(args, config, project_dir, bible_path, story_design_path,
            screenplay_path, narrative_plan_path, scene_cards_path, shot_cards_path,
            render_plan_path, continuity_plan_path, reference_manifest_path)

    # --- ingredients workflow ---
    if config["movie_video_workflow"] == "ingredients":
        return _run_ingredients_workflow(args, config, project_dir, bible_path, story_design_path,
            screenplay_path, narrative_plan_path, scene_cards_path, shot_cards_path,
            render_plan_path, continuity_plan_path, reference_manifest_path,
            render_plan_ingredients_path, ingredients_llm, report_ingredients_analysis, manifest_path)

    # --- default MSR workflow ---
    return _run_msr_workflow(args, config, project_dir, bible_path, story_design_path,
        screenplay_path, narrative_plan_path, scene_cards_path, shot_cards_path,
        render_plan_path, continuity_plan_path, render_plan_msr_path,
        reference_manifest_path,
        ingredients_llm, report_msr_analysis, report_ingredients_analysis, manifest_path)


def _run_movie_openshot_export_stage(args: argparse.Namespace, project_dir: Path) -> MoviePipelineResult:
    """Regenerate an OpenShot project from existing movie plans and rendered clips."""
    config_path = project_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    plan_path = _find_existing_movie_export_plan(project_dir)
    entries = _movie_plan_entries(plan_path)
    clips = _find_existing_movie_render_clips(
        project_dir,
        entries,
        preferred_workflow=getattr(args, "movie_video_workflow", None),
    )
    video = raw_config.get("video") or {}
    audio_value = str(raw_config.get("input_audio") or "").strip()
    audio_path = None
    if audio_value:
        audio_path = coerce_local_path(audio_value, base_dir=project_dir).resolve()
    output_path = project_dir / "output" / "movie" / "openshot" / f"{project_dir.name}.osp"

    console.print(f"Movie OpenShot export: writing {len(clips)} rendered clips")

    def report(completed: int, total: int, label: str) -> None:
        console.print(f"[dim]Movie OpenShot export: {completed}/{total} ({label})[/dim]")

    export_render_plan_to_openshot(
        render_plan_path=plan_path,
        clip_paths=clips,
        audio_path=audio_path,
        output_path=output_path,
        width=int(video.get("width", 1280)),
        height=int(video.get("height", 704)),
        fps=int(video.get("fps", 24)),
        on_progress=report,
    )
    _log_stage("Movie OpenShot export", str(output_path))
    return MoviePipelineResult(
        project_dir=project_dir,
        render_plan_path=plan_path,
        openshot_project_path=output_path,
    )


def _find_existing_movie_export_plan(project_dir: Path) -> Path:
    candidates = (
        project_dir / "movie" / "render_plan.json",
        project_dir / "movie" / "render_plan_msr.json",
        project_dir / "movie" / "render_plan_ingredients.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No existing movie render plan found under movie/")


def _movie_plan_entries(plan_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    entries = payload if isinstance(payload, list) else payload.get("shots") or payload.get("scenes") or []
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Movie render plan contains no shots: {plan_path}")
    normalized = []
    for index, entry in enumerate(entries, start=1):
        item = dict(entry)
        item["scene"] = int(item.get("scene") or item.get("scene_number") or index)
        normalized.append(item)
    return normalized


def _find_existing_movie_render_clips(
    project_dir: Path,
    entries: list[dict[str, Any]],
    *,
    preferred_workflow: str | None,
) -> list[Path]:
    movie_output = project_dir / "output" / "movie"
    workflow_dirs = []
    if preferred_workflow:
        workflow_dirs.append(movie_output / preferred_workflow)
    if movie_output.is_dir():
        workflow_dirs.extend(path for path in sorted(movie_output.iterdir()) if path.is_dir())
    workflow_dirs.append(movie_output)
    clips: list[Path] = []
    for entry in entries:
        scene_number = int(entry.get("scene") or entry.get("scene_number"))
        candidates = []
        for directory in workflow_dirs:
            candidates.extend((
                directory / "final" / f"scene_{scene_number:04}.mp4",
                directory / f"scene_{scene_number:04}.mp4",
                directory / f"scene_{scene_number:04}" / "final.mp4",
            ))
        clip = next((candidate for candidate in candidates if candidate.is_file()), None)
        if clip is None:
            raise FileNotFoundError(f"No rendered movie clip found for scene {scene_number}")
        clips.append(clip)
    return clips


# ====================================================================
# Workflow-specific execution helpers
# ====================================================================

def _run_startframe_director_workflow(
    args, config, project_dir, bible_path, story_design_path, screenplay_path,
    narrative_plan_path, scene_cards_path, shot_cards_path, render_plan_path,
    continuity_plan_path, reference_manifest_path, render_plan_ingredients_path, manifest_path,
) -> MoviePipelineResult:
    from feverslop.adapters.startframe_director_visual import (
        LocalStartframeDirectorVisualAdapter,
    )
    from feverslop.application.startframe_director_prompts import (
        build_startframe_director_prompts,
    )
    from feverslop.application.startframe_i2v_render_plan import (
        write_startframe_i2v_render_plan,
    )
    from feverslop.application.startframe_identity import (
        build_startframe_identity_ledger,
    )
    from feverslop.application.startframe_plan import build_startframe_plan
    from feverslop.application.startframe_validation import (
        write_local_startframe_validation,
    )
    project_config_path = project_dir / "config.json"
    reference_image_size = None
    if project_config_path.is_file():
        from feverslop.config.project_config import ProjectConfig

        project_config = ProjectConfig.load(project_config_path)
        reference_image_size = project_config.reference_images.resolve(project_config.video)

    _log_stage("Movie identity ledger", "deriving face, body, wardrobe, and reference contracts")
    identity_ledger_path = build_startframe_identity_ledger(project_dir=project_dir)
    _log_stage("Movie startframe plan", "deriving shot contracts, bboxes, and continuity requirements")
    startframe_plan_path = build_startframe_plan(project_dir=project_dir)
    _log_stage("Movie director prompts", f"writing {config['startframe_director_backend']} director prompts")
    startframe_director_prompts_path = build_startframe_director_prompts(
        project_dir=project_dir,
        director_backend=config["startframe_director_backend"],
        reference_image_size=reference_image_size,
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


def _run_i2v_edit_workflow(
    args, config, project_dir, bible_path, story_design_path, screenplay_path,
    narrative_plan_path, scene_cards_path, shot_cards_path, render_plan_path,
    continuity_plan_path, reference_manifest_path,
) -> MoviePipelineResult:
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


def _run_ingredients_workflow(
    args, config, project_dir, bible_path, story_design_path, screenplay_path,
    narrative_plan_path, scene_cards_path, shot_cards_path, render_plan_path,
    continuity_plan_path, reference_manifest_path, render_plan_ingredients_path,
    ingredients_llm, report_ingredients_analysis, manifest_path,
) -> MoviePipelineResult:
    if not args.skip_movie_ingredients_sheets:
        from feverslop.application.movie_ingredients_sheets import (
            enrich_movie_render_plan_with_ingredients_sheets,
        )
        _log_stage("Movie Ingredients scene sheets", "composing letterboxed scene reference sheets")
        render_plan_ingredients_path = enrich_movie_render_plan_with_ingredients_sheets(
            project_dir=project_dir,
            sheet_scale=config.get("ingredients_sheet_scale", 2.0),
            llm=ingredients_llm(),
            on_analysis_status=report_ingredients_analysis,
            workflow_profile=Path(
                config.get(
                    "ingredients_workflow",
                    "workflows/video/ltx_25/ingredients/ingredients_draft.json",
                )
                or "workflows/video/ltx_25/ingredients/ingredients_draft.json",
            ).stem,
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
            prepare, render = _movie_workflow_actions(args.write_debug_workflows)
            final_video_path, ingredients_debug_workflows_dir = _prepare_and_render_ingredients_movie(
                adapter=adapter,
                project_dir=project_dir,
                render_plan_path=render_plan_ingredients_path,
                selected_scenes=args.scenes,
                prepare=prepare,
                render=render,
            )
        else:
            adapter = LocalMovieVisualAdapter()
            final_video_path = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=render_plan_ingredients_path,
                selected_scenes=args.scenes,
                on_clip_rendered=lambda completed, total, scene_number: _log_stage(
                    "Movie Ingredients clip", f"rendered {completed}/{total}: scene {scene_number}",
                ),
            )
    elif args.write_debug_workflows:
        if config["render_backend"] == "local":
            raise ValueError("Movie workflow preparation requires the ComfyUI render backend")
        adapter = _build_ingredients_adapter(project_dir, config, debug_workflows_dir=None)
        _, ingredients_debug_workflows_dir = _prepare_and_render_ingredients_movie(
            adapter=adapter, project_dir=project_dir, render_plan_path=render_plan_ingredients_path,
            selected_scenes=args.scenes, prepare=True, render=False,
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


def _run_msr_workflow(
    args, config, project_dir, bible_path, story_design_path, screenplay_path,
    narrative_plan_path, scene_cards_path, shot_cards_path, render_plan_path,
    continuity_plan_path, render_plan_msr_path, reference_manifest_path,
    ingredients_llm, report_msr_analysis, report_ingredients_analysis, manifest_path,
) -> MoviePipelineResult:
    if _movie_uses_msr_reference_enrichment(config["movie_video_workflow"]) and not args.skip_movie_msr_enrich:
        from feverslop.application.movie_msr_enrichment import (
            enrich_movie_render_plan_with_msr_prompts,
        )
        render_plan_msr_path = enrich_movie_render_plan_with_msr_prompts(
            project_dir=project_dir,
            keyframe_mode=args.keyframe_mode,
            llm=ingredients_llm(),
            on_analysis_status=report_msr_analysis,
            workflow_profile=Path(config["msr_workflow"]).stem,
        )
    elif not render_plan_msr_path.exists():
        render_plan_msr_path = None

    if _movie_uses_msr_reference_enrichment(config["movie_video_workflow"]) and not args.skip_movie_ingredients_sheets:
        from feverslop.application.movie_ingredients_sheets import (
            enrich_movie_render_plan_with_ingredients_sheets,
        )
        _log_stage("Movie Ingredients scene sheets", "composing letterboxed scene reference sheets")
        enrich_movie_render_plan_with_ingredients_sheets(
            project_dir=project_dir,
            sheet_scale=config.get("ingredients_sheet_scale", 2.0),
            llm=ingredients_llm(),
            on_analysis_status=report_ingredients_analysis,
            workflow_profile=Path(
                config.get(
                    "ingredients_workflow",
                    "workflows/video/ltx_25/ingredients/ingredients_draft.json",
                )
                or "workflows/video/ltx_25/ingredients/ingredients_draft.json",
            ).stem,
        )

    debug_workflows_dir: Path | None = None
    if args.write_debug_workflows and config["render_backend"] == "local":
        raise ValueError("Movie workflow preparation requires the ComfyUI render backend")
    if args.write_debug_workflows:
        if render_plan_msr_path is None:
            raise FileNotFoundError("Movie debug workflow export requires movie/render_plan_msr.json; run without --skip-movie-msr-enrich first")
        if not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            raise ValueError("Movie debug workflow export requires ready movie references; run without --skip-movie-references first")

    final_video_path: Path | None = None
    prepared_minimax_plan_path: Path | None = None
    if _is_minimax_movie_workflow(config["movie_video_workflow"]):
        _log_stage("Movie MiniMax H3 prompts", "preparing DSPy prompts")
        prompt_adapter = _build_visual_adapter(project_dir, config, workflow=None)
        prepared_minimax_plan_path = prompt_adapter.prepare_render_plan(
            render_plan_path,
            project_dir,
            on_scene_started=lambda index, total, scene: console.print(
                f"[cyan]H3 prompts: processing scene {index}/{total} - scene {scene}[/cyan]",
            ),
            on_scene_prepared=lambda completed, total, scene: console.print(
                f"[cyan]H3 prompts: {completed}/{total} scenes - scene {scene}[/cyan]",
            ),
        )
        _log_stage("Movie MiniMax H3 prompts", str(prepared_minimax_plan_path))
    render_stage_title = (
        "Movie MiniMax H3 render"
        if config["movie_video_workflow"].startswith("minimax-h3-")
        else "Movie MSR render"
    )
    if not args.skip_movie_render:
        _log_stage(render_stage_title, f"rendering via {config['render_backend']}")
        if not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            raise ValueError("Movie references are not ready; run without --skip-movie-references first")
        workflow = _patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        adapter = _build_visual_adapter(project_dir, config, workflow)
        if config["render_backend"] == "local" or config["movie_video_workflow"] != "msr":
            final_video_path = adapter.render_movie(
                project_dir=project_dir, render_plan_path=prepared_minimax_plan_path or render_plan_msr_path or render_plan_path,
                selected_scenes=args.scenes, continuity_keyframes=config["continuity_keyframes"],
                on_clip_rendered=lambda completed, total, scene_number: print(
                    f"Rendered movie clip {completed}/{total}: scene {scene_number}",
                ),
            )
        else:
            prepare, render = _movie_workflow_actions(args.write_debug_workflows)
            final_video_path, debug_workflows_dir = _prepare_and_render_msr_movie(
                adapter=adapter, project_dir=project_dir,
                render_plan_path=render_plan_msr_path or render_plan_path,
                selected_scenes=args.scenes, prepare=prepare,
                render=render,
            )
    elif args.write_debug_workflows:
        _log_stage(render_stage_title, "preparing workflows; render skipped")
        workflow = _patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        adapter = _build_visual_adapter(project_dir, config, workflow)
        _, debug_workflows_dir = _prepare_and_render_msr_movie(
            adapter=adapter, project_dir=project_dir,
            render_plan_path=render_plan_msr_path or render_plan_path,
            selected_scenes=args.scenes, prepare=True, render=False,
        )
    else:
        _log_stage(render_stage_title, "skipped")

    _log_stage("Movie complete", str(final_video_path or prepared_minimax_plan_path or render_plan_path))

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


# ====================================================================
# Helper functions (wiring, adapters, utilities)
# ====================================================================

def _movie_workflow_actions(write_debug_workflows: bool) -> tuple[bool, bool]:
    return True, not write_debug_workflows


def _log_stage(title: str, detail: str = "") -> None:
    progress = _stage_progress.get()
    if progress is not None:
        progress.advance(title)
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


# --- Adapter builders ---

def _build_reference_generator(config: dict[str, Any]):
    from feverslop.application.movie_references import MovieReferenceSheetGenerator

    if config["reference_backend"] == "local":
        local = LocalMovieImageBackend()
        return MovieReferenceSheetGenerator(backend=local, edit_backend=local)
    return build_movie_reference_generator(movie_config=config)


def _build_visual_adapter(project_dir: Path, config: dict[str, Any], workflow: dict):
    if config["render_backend"] == "local":
        return LocalMovieVisualAdapter()
    i2v_workflow = _patch_movie_msr_workflow(template_path=Path(config["msr_i2v_workflow"])) if config.get("msr_i2v_workflow") else None
    return build_movie_visual_adapter(
        project_dir,
        Path(config["msr_workflow"]),
        movie_config=config,
        workflow=workflow,
        i2v_workflow=i2v_workflow,
    )


def _patch_movie_msr_workflow(template_path: Path):
    from feverslop.composition.movie_workflow import patch_movie_msr_workflow

    return patch_movie_msr_workflow(template_path=template_path)


def _movie_app_config_path(config: dict[str, Any]) -> str:
    return str(config.get("app_config_path") or "app_config.json")


def _build_ingredients_adapter(project_dir: Path, config: dict[str, Any], *, debug_workflows_dir: Path | None = None):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_ingredients_video_backend import (
        ComfyUIIngredientsVideoRenderBackend,
    )
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.movie_ingredients_visual import (
        ComfyUIMovieIngredientsVisualAdapter,
    )
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load(_movie_app_config_path(config))
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides)
    ltx_dir = project_dir / "output" / "movie" / "ltx_ingredients"
    backend = ComfyUIIngredientsVideoRenderBackend(
        client=client,
        workflow_path=config.get("ingredients_workflow", "workflows/video/ltx_25/ingredients/ingredients_draft.json"),
        output_dir=ltx_dir,
        project_dir=project_dir,
        model_resolver=model_resolver,
        debug_workflows_dir=debug_workflows_dir,
        workflow_profile=Path(
            config.get(
                "ingredients_workflow",
                "workflows/video/ltx_25/ingredients/ingredients_draft.json",
            )
            or "workflows/video/ltx_25/ingredients/ingredients_draft.json",
        ).stem,
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
    from feverslop.composition.render_video import (
        RenderVideoCompositionOptions,
        build_render_video_scenes_use_case,
    )
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load(_movie_app_config_path(config))
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
        ),
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
    from feverslop.adapters.startframe_director_comfyui import (
        ComfyUIStartframeDirectorVisualAdapter,
    )
    from feverslop.composition.render_video import (
        RenderVideoCompositionOptions,
        build_render_video_scenes_use_case,
    )
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load(_movie_app_config_path(config))
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
        ),
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


# --- Artifact delegation (thin wrappers around application layer) ---

def _ensure_movie_planning_artifacts(project_dir, force_screenplay=False, force_story_design=False):
    from feverslop.application.movie_artifacts import ensure_movie_planning_artifacts
    return ensure_movie_planning_artifacts(project_dir, force_screenplay=force_screenplay, force_story_design=force_story_design)


def _ensure_movie_bible_artifact(project_dir):
    from feverslop.application.movie_artifacts import ensure_movie_bible
    return ensure_movie_bible(project_dir)


def _regenerate_movie_bible_artifact(project_dir, planner_backend, app_config_path="app_config.json"):
    from feverslop.application.movie_artifacts import regenerate_movie_bible
    from feverslop.composition.movie_planner import build_movie_planner

    return regenerate_movie_bible(
        project_dir,
        planner=build_movie_planner({
            "planner_backend": planner_backend,
            "app_config_path": app_config_path,
        }),
    )


def _ensure_movie_story_design_artifact(project_dir, force=False):
    from feverslop.application.movie_artifacts import ensure_movie_story_design
    return ensure_movie_story_design(project_dir, force=force)


def _ensure_movie_screenplay_artifact(project_dir, force=False):
    from feverslop.application.movie_artifacts import ensure_movie_screenplay
    return ensure_movie_screenplay(project_dir, force=force)


def _ensure_movie_narrative_plan_artifact(project_dir):
    from feverslop.application.movie_artifacts import ensure_movie_narrative_plan
    return ensure_movie_narrative_plan(project_dir)


def _ensure_movie_scene_cards_artifact(project_dir):
    from feverslop.application.movie_artifacts import ensure_movie_scene_cards
    return ensure_movie_scene_cards(project_dir)


def _ensure_movie_shot_cards_artifact(project_dir):
    from feverslop.application.movie_artifacts import ensure_movie_shot_cards
    return ensure_movie_shot_cards(project_dir)


def _ensure_movie_continuity_plan_artifact(project_dir):
    from feverslop.application.movie_artifacts import ensure_movie_continuity_plan
    return ensure_movie_continuity_plan(project_dir)


def _ensure_movie_render_plan_matches_bible_artifact(project_dir):
    from feverslop.application.movie_artifacts import (
        ensure_movie_render_plan_matches_bible,
    )
    ensure_movie_render_plan_matches_bible(project_dir)


def _write_movie_reference_manifest_from_bible_artifact(project_dir):
    from feverslop.application.movie_artifacts import (
        write_movie_reference_manifest_from_bible,
    )
    return write_movie_reference_manifest_from_bible(project_dir)


# --- Workflow helpers ---

def _canonical_movie_plan(project_dir: Path, source: Path, *, pipeline: str) -> Path:
    layout = SceneArtifactLayout(project_dir)
    destination = layout.ingredients_plan if pipeline == "ltx_ingredients" else layout.references_plan
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _prepare_and_render_msr_movie(
    *, adapter, project_dir: Path, render_plan_path: Path, selected_scenes: list[int],
    prepare: bool, render: bool,
) -> tuple[Path | None, Path]:
    from feverslop.adapters.prepared_workflow import (
        PreparedWorkflowRenderer,
        WorkflowMaterializer,
    )
    from feverslop.application.movie_prepared_workflows import (
        prepare_movie_workflows,
        render_prepared_movie_workflows,
    )

    canonical_plan = _canonical_movie_plan(project_dir, render_plan_path, pipeline="ltx_msr")
    plan = json.loads(canonical_plan.read_text(encoding="utf-8"))
    scenes = adapter._movie_scenes(plan, project_dir=project_dir)
    layout = SceneArtifactLayout(project_dir)
    backend = adapter._build_backend(
        workflow_path=adapter.workflow_path,
        workflow=adapter.workflow,
        output_dir=layout.render_dir,
        project_dir=project_dir,
    )
    if prepare:
        prepare_movie_workflows(
            project_dir=project_dir, render_plan_path=canonical_plan, pipeline="ltx_msr",
            scenes=scenes, selected_scenes=selected_scenes,
            materializer=WorkflowMaterializer(backend, layout),
            prompt_for_scene=lambda scene: str(
                (scene.get("ltx") or {}).get("original_style_i2v_prompt")
                or scene.get("description") or "",
            ),
        )
    if not render:
        return None, layout.scenes_dir
    final = render_prepared_movie_workflows(
        project_dir=project_dir, scenes=scenes, selected_scenes=selected_scenes,
        renderer=PreparedWorkflowRenderer(
            project_dir=project_dir, render_queue=backend.render_queue,
            postprocessor=adapter.postprocessor, expected_pipeline="ltx_msr",
            expected_workflow_profile=Path(adapter.workflow_path).stem,
        ),
        postprocessor=adapter.postprocessor,
        legacy_dirs=[project_dir / "output" / "movie" / "ltx_msr"],
        on_clip_rendered=lambda completed, total, scene: print(
            f"Rendered movie clip {completed}/{total}: scene {scene}",
        ),
    )
    return final, layout.scenes_dir


def _prepare_and_render_ingredients_movie(
    *, adapter, project_dir: Path, render_plan_path: Path, selected_scenes: list[int],
    prepare: bool, render: bool,
) -> tuple[Path | None, Path]:
    from feverslop.adapters.prepared_workflow import (
        PreparedWorkflowRenderer,
        WorkflowMaterializer,
    )
    from feverslop.application.movie_prepared_workflows import (
        prepare_movie_workflows,
        render_prepared_movie_workflows,
    )

    canonical_plan = _canonical_movie_plan(project_dir, render_plan_path, pipeline="ltx_ingredients")
    plan = json.loads(canonical_plan.read_text(encoding="utf-8"))
    scenes = adapter._movie_scenes(plan, project_dir=project_dir)
    layout = SceneArtifactLayout(project_dir)
    backend = adapter.backend
    if prepare:
        prepare_movie_workflows(
            project_dir=project_dir, render_plan_path=canonical_plan, pipeline="ltx_ingredients",
            scenes=scenes, selected_scenes=selected_scenes,
            materializer=WorkflowMaterializer(backend, layout),
            prompt_for_scene=lambda scene: str(
                (scene.get("ltx") or {}).get("static_prompt")
                or (scene.get("ingredients") or {}).get("global_prompt")
                or scene.get("ingredients_global_prompt")
                or (scene.get("ltx") or {}).get("ingredients_scene_sheet_description")
                or (scene.get("ltx") or {}).get("ingredients_target_prompt")
                or scene.get("description") or "",
            ),
        )
    if not render:
        return None, layout.scenes_dir
    final = render_prepared_movie_workflows(
        project_dir=project_dir, scenes=scenes, selected_scenes=selected_scenes,
        renderer=PreparedWorkflowRenderer(
            project_dir=project_dir, render_queue=backend.render_queue,
            postprocessor=backend.postprocessor, expected_pipeline="ltx_ingredients",
            expected_workflow_profile=Path(backend.workflow_label).stem,
        ),
        postprocessor=backend.postprocessor,
        legacy_dirs=[project_dir / "output" / "movie" / "ltx_ingredients"],
        on_clip_rendered=lambda completed, total, scene: _log_stage(
            "Movie Ingredients clip", f"rendered {completed}/{total}: scene {scene}",
        ),
    )
    return final, layout.scenes_dir


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
    """CLI entry point for the movie pipeline."""
    args = build_movie_arg_parser().parse_args()
    run(args)
