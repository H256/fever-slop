"""MSR/Ingredients stage runners for the M-20 split of ``stage_runners`` (issue #1194)."""
from __future__ import annotations

from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.application.msr_prompt_enrichment import (
    enrich_render_plan_with_msr_prompts,
)
from feverslop.application.reference_bible import (
    enrich_render_plan_with_reference_sheets,
)
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.tools.reference_bible import run as render_reference_bible

from ..config_loader import PipelineRunState, count_render_plan_items
from ..resume_plan import reference_manifests_reusable
from .plan_stages import (
    _get_reference_bible_parser,
    _get_resolution,
    _report_reference_fallbacks,
    _run_render_plan_stage,
    _seed_reference_bindings,
)
from .progress import (
    RenderProgressReporter,
    _canonical_plan_path,
    _report,
    _scene_progress_callback,
    get_reporter,
)


def _run_msr_references_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients", "minimax-h3-r2v", "minimax-h3-i2v"):
        raise ValueError(
            "msr_references requires --video-pipeline ltx_msr, ltx_ingredients, minimax-h3-r2v, or minimax-h3-i2v",
        )
    project_config_path = getattr(state.context, "project_config_path", None)
    if project_config_path is not None:
        config = ProjectConfig.load(project_config_path)
        _report_reference_fallbacks(_seed_reference_bindings(
            state.plan_for_next_step,
            config,
        ))
        if reference_manifests_reusable(
            state.context.references_dir,
            actor_ids=(actor.id for actor in config.actors),
            location_id=(location.id for location in config.structured_locations),
        ):
            _report("[yellow]Skipping MSR reference rendering; existing reference manifests are reusable.[/yellow]")
            return
    reference_args = _get_reference_bible_parser().parse_args([
        "--project-config",
        str(state.context.project_config_path),
        "--app-config",
        str(state.app_config_path),
        "--hero-workflow",
        str(state.reference_hero_workflow),
        "--edit-workflow",
        str(state.reference_edit_workflow),
        "--output-dir",
        str(state.context.references_dir),
        "--view-set",
        "msr",
        "--reference-generation",
        str(getattr(state.args, "reference_generation", "image_views")),
        "--sequence-workflow",
        str(getattr(state.args, "sequence_to_sheet_workflow", "workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json")),
    ])
    render_reference_bible(reference_args, reporter=get_reporter())


def _run_msr_reference_sheets_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients", "minimax-h3-r2v", "minimax-h3-i2v"):
        raise ValueError(
            "msr_reference_sheets requires --video-pipeline ltx_msr, ltx_ingredients, minimax-h3-r2v, or minimax-h3-i2v",
        )
    if not state.plan_for_next_step.is_file():
        _report(
            "[dim]Render plan missing; creating the intermediate plan before enriching MSR references...[/dim]",
        )
        _run_render_plan_stage(state)
    project_config_path = getattr(state.context, "project_config_path", None)
    project_config = ProjectConfig.load(project_config_path) if project_config_path is not None else None
    if project_config is not None:
        _report_reference_fallbacks(_seed_reference_bindings(
            state.plan_for_next_step,
            project_config,
        ))
    state.context.artifact_layout.plans_dir.mkdir(parents=True, exist_ok=True)
    msr_reference_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Enriching MSR references", msr_reference_total) as reference_progress:
        enrichment_options = (
            {"max_scene_actors": project_config.max_scene_actors}
            if project_config is not None and project_config.max_scene_actors != 4
            else {}
        )
        state.plan_for_next_step = enrich_render_plan_with_reference_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.reference_plan,
            on_scene_complete=_scene_progress_callback(reference_progress),
            canonical_plan_path=_canonical_plan_path(state),
            **enrichment_options,
        )


def _run_msr_prompt_enrich_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients"):
        raise ValueError("msr_prompt_enrich requires --video-pipeline ltx_msr or ltx_ingredients")
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        api_key=app_config.llm.api_key,
        model=app_config.llm.model_for("structured"),
        temperature=app_config.llm.temperature,
        dspy_temperature=app_config.llm.dspy_temperature,
        task_temperatures=app_config.llm.task_temperatures,
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        max_concurrent_requests=app_config.llm.max_concurrent_requests,
        chat_template_kwargs=app_config.llm.chat_template_kwargs,
    )
    msr_prompt_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Enriching MSR prompts", msr_prompt_total) as msr_prompt_progress:
        state.plan_for_next_step = enrich_render_plan_with_msr_prompts(
            state.plan_for_next_step,
            state.context.reference_plan,
            canonical_plan_path=_canonical_plan_path(state),
            llm=llm,
            on_analysis_status=msr_prompt_progress.analysis_attempt,
            on_scene_complete=_scene_progress_callback(msr_prompt_progress),
        )


def _run_ingredients_sheets_stage(state: PipelineRunState) -> None:
    from feverslop.application.render_plan_ingredients_sheets import (
        enrich_render_plan_with_ingredients_sheets,
    )
    if state.args.video_pipeline != "ltx_ingredients":
        raise ValueError("ingredients_sheets requires --video-pipeline ltx_ingredients")
    from feverslop.config.project_config import ProjectConfig
    project_config = ProjectConfig.load(state.context.project_config_path)
    resolution = _get_resolution(state.args)
    if resolution is not None:
        project_config = project_config.apply_resolution_override(
            width=resolution[0], height=resolution[1],
        )
    video_settings = project_config.to_video_settings()
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        api_key=app_config.llm.api_key,
        model=app_config.llm.model_for("structured"),
        temperature=app_config.llm.temperature,
        dspy_temperature=app_config.llm.dspy_temperature,
        task_temperatures=app_config.llm.task_temperatures,
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        max_concurrent_requests=app_config.llm.max_concurrent_requests,
        chat_template_kwargs=app_config.llm.chat_template_kwargs,
    )
    state.context.artifact_layout.plans_dir.mkdir(parents=True, exist_ok=True)
    ingredients_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Composing Ingredients scene sheets", ingredients_total) as progress:
        state.plan_for_next_step = enrich_render_plan_with_ingredients_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.ingredients_plan,
            canonical_plan_path=_canonical_plan_path(state),
            video_settings=video_settings,
            llm=llm,
            on_analysis_status=progress.analysis_attempt,
            on_scene_complete=_scene_progress_callback(progress),
            workflow_profile=str(
                getattr(state.args, "video_workflow_profile", None)
                or state.ingredients_workflow.stem,
            ),
        )

