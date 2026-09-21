"""Plan-stage runners for the M-20 split of ``stage_runners`` (issue #1194)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.h3_prompt_checkpoints import H3PromptCheckpointStore
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.adapters.reporting import ConsoleReporter
from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
from feverslop.application.h3_prompt_pipeline import H3PromptPipeline
from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.application.reference_bible import (
    enrich_render_plan_with_reference_sheets,
)
from feverslop.composition.generate_render_plan import (
    build_generate_render_plan_use_case,  # noqa: F401
    execute_generate_render_plan,
)
from feverslop.composition.canonical_plan_regenerator import CanonicalPlanRegenerator
from feverslop.composition.continuation_capability import resolve_continuation_capability
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.h3_audio_delivery import load_h3_audio_delivery
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.prompting.dspy_h3_prompt_builder import (
    DspyH3PromptBuilder,
    _normalize_relay_segments,
    build_dspy_generator,
)
from feverslop.prompting.h3_prompt_builder import H3PromptBuilder
from feverslop.prompting.ltx_prompt_anchor_fixer import (
    LTXPromptAnchorFixer,
    validate_anchor_file,
)
from feverslop.prompting.model_types import resolve_model_type
from feverslop.prompting.relay_direction_builder import RelayDirectionBuilder
from feverslop.tools.reference_bible import (
    build_arg_parser as build_reference_bible_arg_parser,
)
from feverslop.tools.storyboard_page import parse_scene_list
from feverslop.utils.stems import discover_stem_files

from ..config_loader import PipelineRunState
# Progress-reporting state lives in .progress; re-exported so existing
# callers and patch targets keep one source of truth.
from .progress import (  # noqa: F401
    _report,
    console,
)

_REFERENCE_BIBLE_PARSER = None
def _get_reference_bible_parser():
    global _REFERENCE_BIBLE_PARSER
    if _REFERENCE_BIBLE_PARSER is None:
        _REFERENCE_BIBLE_PARSER = build_reference_bible_arg_parser()
    return _REFERENCE_BIBLE_PARSER


def _get_resolution(args: argparse.Namespace) -> tuple[int, int] | None:
    """Extract resolution tuple from CLI args if --resolution was provided."""
    res = getattr(args, "resolution", None)
    if res is None:
        return None
    return (res.width, res.height)


def _run_tests_stage(_state: PipelineRunState) -> None:
    from feverslop.composition.stage_runners import run_unittest_suite
    run_unittest_suite()


def _selected_video_workflows(state: PipelineRunState) -> tuple[Path, ...]:
    if state.args.video_pipeline == "ltx_msr":
        candidates = (state.msr_workflow,)
    elif state.args.video_pipeline == "ltx_ingredients":
        candidates = (state.ingredients_workflow,)
    elif state.args.render_mode == "single_prompt":
        workflow = state.single_prompt_workflow
        candidates = (workflow,)
    elif state.args.render_mode == "relay":
        candidates = (state.relay_workflow,)
    else:
        candidates = (state.relay_workflow, state.single_prompt_workflow)
    result = tuple(path for path in candidates if str(path).strip() not in {"", "."})
    if not result:
        checked: list[str] = []
        vp = getattr(state.args, "video_pipeline", None)
        rm = getattr(state.args, "render_mode", None)
        if vp == "ltx_msr":
            checked.append(f"msr_workflow={state.msr_workflow!r}")
        elif vp == "ltx_ingredients":
            checked.append(f"ingredients_workflow={state.ingredients_workflow!r}")
        elif rm == "single_prompt":
            checked.append(f"single_prompt_workflow={state.single_prompt_workflow!r}")
        elif rm == "relay":
            checked.append(f"relay_workflow={state.relay_workflow!r}")
        else:
            checked.append(f"relay_workflow={state.relay_workflow!r}")
            checked.append(f"single_prompt_workflow={state.single_prompt_workflow!r}")
        raise ValueError(
            f"No valid video workflows found for pipeline {vp!r} "
            f"(render_mode={rm!r}); checked: {', '.join(checked)}",
        )
    return result


def _run_main_pipeline_stage(state: PipelineRunState) -> None:
    resolution = _get_resolution(state.args)
    execute_generate_render_plan(
        GenerateRenderPlanRequest(
            project_config_path=state.context.project_config_path,
            app_config_path=state.app_config_path,
            concept_batch_size=int(state.args.concept_batch_size),
            video_workflow_paths=_selected_video_workflows(state),
            rolling_frame_profile=state.args.rolling_frame_profile,
            defer_h3_until_references=state.args.video_pipeline == "minimax-h3-r2v",
            skip_stem_separation=getattr(state.args, "skip_stem_separation", False),
            skip_whisper=getattr(state.args, "skip_whisper", False),
            skip_beat_analysis=getattr(state.args, "skip_beat_analysis", False),
            resume=bool(getattr(state.args, "resume", False)),
            story_direction=str(getattr(state.args, "story_direction", "") or ""),
            story_plan_approve=bool(getattr(state.args, "story_plan_approve", False)),
        ),
        console=console,
        resolution=resolution,
    )
    state.plan_for_next_step = state.context.render_plan


def _read_h3_input(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot run h3_prompts: missing {label} artifact {path}. "
            "Run the main_pipeline once to create the upstream prompt artifacts.",
        )
    return JsonArtifactStore().read_json(path)


def _select_pipeline_scenes(scenes: list[dict], scene_spec: str | None) -> list[dict]:
    selected = parse_scene_list(scene_spec)
    if selected is None:
        return scenes
    return [scene for scene in scenes if int(scene.get("scene") or scene.get("scene_number")) in selected]


def _report_reference_fallbacks(warnings: list[str]) -> None:
    for warning in warnings:
        _report(f"[yellow]Reference fallback:[/yellow] {warning}")


def _seed_reference_bindings(plan_path: Path, config: ProjectConfig) -> list[str]:
    """Ensure every scene has actor/location IDs resolvable from the bible."""
    if not plan_path.is_file():
        return []
    store = JsonArtifactStore()
    plan = store.read_json(plan_path)
    actors = list(config.actors)
    locations = list(config.structured_locations)
    reference_root = config.project_dir / "output" / "references"
    actor_ids = [actor.id for actor in actors]
    if not actor_ids:
        actor_ids = sorted(
            path.name
            for path in (reference_root / "actors").iterdir()
            if path.is_dir()
        ) if (reference_root / "actors").is_dir() else []
    location_candidates: list[tuple[str, str]] = [
        (location.id, " ".join((location.id, location.name, location.visual_description, location.image_prompt)))
        for location in locations
    ]
    locations_root = reference_root / "locations"
    if locations_root.is_dir():
        for location_dir in sorted(path for path in locations_root.iterdir() if path.is_dir()):
            manifest_path = location_dir / "manifest.json"
            manifest: dict = {}
            if manifest_path.is_file():
                try:
                    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest = loaded if isinstance(loaded, dict) else {}
                except (OSError, json.JSONDecodeError):
                    manifest = {}
            location_candidates.append((
                location_dir.name,
                " ".join(
                    str(manifest.get(key, ""))
                    for key in ("id", "name", "visual_description", "image_prompt")
                ) + " " + location_dir.name,
            ))
    existing_actor_ids = [
        str(actor_id)
        for scene in plan
        for actor_id in (scene.get("references") or {}).get("actor_ids") or []
        if str(actor_id).strip()
    ]
    actor_ids = list(dict.fromkeys([*actor_ids, *existing_actor_ids]))
    if not actor_ids or not location_candidates:
        return []
    changed = False
    warnings: list[str] = []
    for scene in plan:
        references = scene.setdefault("references", {})
        fallback_fields: list[str] = []
        if "actor_ids" not in references:
            references["actor_ids"] = actor_ids[: config.max_scene_actors]
            changed = True
            fallback_fields.append(f"actor_ids={references['actor_ids']!r}")
        if not references.get("location_id"):
            text = " ".join(
                str(value)
                for value in (
                    scene.get("z_image", {}).get("prompt"),
                    scene.get("ltx", {}).get("base_prompt"),
                    scene.get("metadata", {}).get("base_concept"),
                )
                if value
            ).lower()
            matching = next(
                (
                    location
                    for location in location_candidates
                    if any(
                        part.strip() and part.strip().lower() in text
                        for part in location[1].split()
                    )
                ),
                location_candidates[0],
            )
            references["location_id"] = matching[0]
            changed = True
            fallback_fields.append(f"location_id={matching[0]!r}")
        if fallback_fields:
            warnings.append(
                f"scene {scene.get('scene')}: structured reference serialization missing; "
                f"using {' and '.join(fallback_fields)} from the existing reference bible.",
            )
    if changed:
        store.write_json(plan_path, plan)
    return warnings


def _discover_stem_files(
    stems_dir: Path,
    input_audio: Path | None = None,
) -> dict[str, Path] | None:
    return discover_stem_files(stems_dir, input_audio)


def _merge_reference_paths_into_h3_segments(
    stage1_segments: list[dict],
    reference_plan_path: Path,
) -> list[dict]:
    if not reference_plan_path.is_file():
        raise FileNotFoundError(
            f"Cannot run h3_prompts: missing reference-enriched render plan {reference_plan_path}. "
            "Run --stage msr_reference_sheets first.",
        )
    reference_plan = JsonArtifactStore().read_json(reference_plan_path)
    by_scene = {
        int(scene.get("scene")): scene.get("references", {})
        for scene in reference_plan
        if scene.get("scene") is not None
    }
    enriched = []
    for index, segment in enumerate(stage1_segments, start=1):
        scene_number = segment.get("scene") or segment.get("scene_number") or index
        references = by_scene.get(int(scene_number))
        if references is None:
            raise ValueError(f"No reference-enriched scene found for H3 segment {segment.get('segment_id', index)}")
        enriched.append({**segment, "references": {**(segment.get("references") or {}), **references}})
    return enriched


def _select_h3_segments(
    stage1_segments: list[dict],
    selected_scene_spec: str | None,
) -> tuple[set[int], list[dict]]:
    if not selected_scene_spec:
        return set(), stage1_segments
    selected = parse_scene_list(selected_scene_spec) or set()
    return selected, [
        segment
        for segment in stage1_segments
        if int(segment.get("scene") or 0) in selected
    ]


def _run_h3_prompts_stage(state: PipelineRunState) -> None:
    """Regenerate only stage 8.5 from the existing stage 7/8 artifacts."""
    config = ProjectConfig.load(state.context.project_config_path)
    if state.args.video_pipeline != config.video_pipeline:
        from dataclasses import replace
        try:
            resolve_model_type(state.args.video_pipeline)
        except ValueError:
            pass
        else:
            config = replace(config, video_pipeline=state.args.video_pipeline)
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    artifact_store = JsonArtifactStore()
    paths = config.paths
    stage1_segments = _read_h3_input(state.context.stage1_segments, "stage 1 segments")
    selected_scene_spec = getattr(state.args, "scenes", None)
    all_scene_numbers = {
        int(segment.get("scene") or segment.get("scene_number") or 0)
        for segment in stage1_segments
    }
    selected, stage1_segments = _select_h3_segments(stage1_segments, selected_scene_spec)
    selected_scene_selection_complete = bool(selected_scene_spec) and selected == all_scene_numbers
    if state.args.video_pipeline == "minimax-h3-r2v":
        _report_reference_fallbacks(_seed_reference_bindings(state.plan_for_next_step, config))
        state.plan_for_next_step = enrich_render_plan_with_reference_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.reference_plan,
            **({"max_scene_actors": config.max_scene_actors} if config.max_scene_actors != 4 else {}),
        )
        stage1_segments = _merge_reference_paths_into_h3_segments(
            stage1_segments,
            state.plan_for_next_step,
        )
    concept_prompts = _read_h3_input(state.context.concept_prompts, "concept prompts")
    scene_details = _read_h3_input(state.context.scene_details, "scene details")
    if selected_scene_spec:
        selected_ids = {str(segment.get("segment_id")) for segment in stage1_segments}
        concept_prompts = {
            key: value for key, value in concept_prompts.items() if str(key) in selected_ids
        }
        scene_details = {
            key: value for key, value in scene_details.items() if str(key) in selected_ids
        }
        _report(
            f"[cyan]Scene selection active: H3 generation limited to "
            f"{', '.join(str(number) for number in sorted(selected))}[/cyan]",
        )
    global_context = _read_h3_input(state.context.resolved_context, "resolved context")
    if state.args.video_pipeline.startswith("minimax-h3"):
        # Prompt semantics must reflect the workflow that will consume them.
        # In particular, audio-latent workflows do not turn a supplied mix into
        # audience-only score merely because its filename resembles a music stem.
        global_context = dict(global_context)
        global_context["h3_audio_delivery"] = load_h3_audio_delivery(
            state.single_prompt_workflow,
        ).to_context()
    h3_prompts_json = paths.prompts_dir / f"h3_prompts_{config.song_id}.json"
    paths.prompts_dir.mkdir(parents=True, exist_ok=True)
    reporter = ConsoleReporter(console)
    context = GenerateRenderPlanContext(
        request=state.args,
        config=config,
        paths=paths,
        app_config=app_config,
        song_id=config.song_id,
        artifact_store=artifact_store,
        reporter=reporter,
        console=console,
        log_step=reporter.step,
        log_file=reporter.file,
        stage1_segments=stage1_segments,
        global_context=global_context,
        concept_prompts=concept_prompts,
        scene_details=scene_details,
        scene_prompts_json=state.context.scene_prompts,
        h3_prompts_json=h3_prompts_json,
        render_plan_json=state.context.render_plan,
        stem_files=_discover_stem_files(paths.stems_dir, config.input_audio),
        ltx_prompt_relay_json=paths.prompts_dir / f"ltx_prompt_relay_{config.song_id}.json",
        beat_json=paths.timeline_dir / f"beat_data_{config.song_id}.json",
        selected_scene_numbers=selected if selected_scene_spec else None,
        selected_scene_selection_complete=selected_scene_selection_complete,
    )
    # Compute the worst-case shot count across all selected scenes so the
    # H3 planner token budget auto-scales to the largest plan instead of
    # truncating on the first try.  The biggest plan sets the ceiling.
    _capacity = max(
        (len(_normalize_relay_segments(seg)) for seg in stage1_segments), default=1
    )
    pipeline = H3PromptPipeline(
        llm_factory=lambda current_config: OpenAICompatibleLLMClient(
            base_url=current_config.llm.base_url,
            api_key=current_config.llm.api_key,
            model=current_config.llm.model_for("structured"),
            temperature=current_config.llm.temperature,
            dspy_temperature=current_config.llm.dspy_temperature,
            task_temperatures=current_config.llm.task_temperatures,
            max_tokens=current_config.llm.max_tokens,
            request_timeout_seconds=current_config.llm.request_timeout_seconds,
            dspy_cache=getattr(current_config.llm, "dspy_cache", False),
            max_concurrent_requests=current_config.llm.max_concurrent_requests,
            prompt_judge_attempts=current_config.llm.prompt_judge_attempts,
            prompt_judge_max_tokens=current_config.llm.prompt_judge_max_tokens,
            prompt_judge_blocking=current_config.llm.prompt_judge_blocking,
            prompt_judge_enabled=current_config.llm.prompt_judge_enabled,
            prompt_planner_max_tokens=current_config.llm.prompt_planner_max_tokens,
            chat_template_kwargs=current_config.llm.chat_template_kwargs,
        ),
        h3_prompt_builder_factory=H3PromptBuilder,
        dspy_prompt_builder_factory=lambda llm: DspyH3PromptBuilder(
            build_dspy_generator(llm, plan_shot_capacity=_capacity),
            reference_root=paths.project_dir,
            allow_fallback=True,
            reporter=reporter,
        ),
        checkpoint_store_factory=lambda _context: H3PromptCheckpointStore(
            paths.project_dir,
            reporter=reporter,
        ),
    )
    pipeline.execute(context)
    state.plan_for_next_step = state.context.render_plan
    _report(f"[green]OK H3 Prompts JSON: {h3_prompts_json}[/green]")


def _run_render_plan_stage(state: PipelineRunState) -> None:
    """Build step 9 from prompt artifacts produced by the main pipeline."""
    config = ProjectConfig.load(state.context.project_config_path)
    paths = config.paths
    song_id = config.song_id
    h3_prompts = paths.prompts_dir / f"h3_prompts_{song_id}.json"
    # -- stem audio (MiniMax H3 R2V) --
    stem_list: list[str] | None = list(config.minimax_h3_audio_refs.stems)
    input_audio: Path | None = config.input_audio
    stem_files = _discover_stem_files(paths.stems_dir, input_audio)
    selected_scene_spec = getattr(state.args, "scenes", None)
    selected_scene_numbers = parse_scene_list(selected_scene_spec) if selected_scene_spec else None
    reference_plan_path = (
        state.context.reference_plan
        if config.video_pipeline == "minimax-h3-r2v"
        else None
    )
    app_config = AppConfig.load(state.app_config_path)
    app_config.attach_import_store(state.context.project_config_dir)
    profile = app_config.resolve_video_workflow_profile(
        pipeline=config.video_pipeline,
        purpose="final",
        name=getattr(state.args, "video_workflow_profile", None),
    )
    regenerator = CanonicalPlanRegenerator(
        config.project_dir,
        selected_scene_numbers=selected_scene_numbers,
        reference_plan_path=reference_plan_path,
        reporter=ConsoleReporter(console),
    )
    build_render_plan(
        scene_prompts_json=paths.prompts_dir / f"scene_prompts_{song_id}.json",
        ltx_prompt_relay_json=paths.prompts_dir / f"ltx_prompt_relay_{song_id}.json",
        output_json_file=state.context.render_plan,
        video_settings=config.to_video_settings(),
        artifact_store=JsonArtifactStore(),
        h3_prompts_json=h3_prompts if h3_prompts.is_file() else None,
        stem_list=stem_list,
        input_audio=input_audio,
        stem_files=stem_files,
        project_dir=config.project_dir,
        max_scene_actors=config.max_scene_actors,
        duration_capability=resolve_continuation_capability(config.video_pipeline, profile),
        plan_writer=regenerator.write,
    )
    if selected_scene_spec:
        _report(
            "[cyan]Scene selection active: regenerated selected scenes while "
            "preserving unselected canonical scenes[/cyan]",
        )
    state.plan_for_next_step = state.context.render_plan
    _report(f"[green]OK Render Plan JSON: {state.plan_for_next_step}[/green]")


def _preserve_enriched_reference_paths(
    *,
    output_path: Path,
    reference_plan_path: Path,
) -> None:
    """Carry MSR paths into a rebuilt MiniMax plan without replacing new fields."""
    if not reference_plan_path.is_file() or reference_plan_path.resolve() == output_path.resolve():
        return

    artifact_store = JsonArtifactStore()
    rebuilt_plan = artifact_store.read_json(output_path)
    enriched_plan = artifact_store.read_json(reference_plan_path)
    enriched_by_scene = {
        str(scene.get("scene")): scene.get("references") or {}
        for scene in enriched_plan
    }
    reference_keys = {
        "actor_sheet_paths",
        "actor_msr_paths",
        "actor_reference_descriptions",
        "location_sheet_path",
        "location_msr_path",
        "location_reference_description",
        "visual_consistency_sources",
    }
    changed = False
    for scene in rebuilt_plan:
        source_references = enriched_by_scene.get(str(scene.get("scene")))
        if not source_references:
            continue
        references = scene.setdefault("references", {})
        for key in reference_keys:
            if key in source_references and references.get(key) != source_references[key]:
                references[key] = source_references[key]
                changed = True
    if changed:
        artifact_store.write_json(output_path, rebuilt_plan)


def _run_relay_compact_stage(state: PipelineRunState) -> None:
    if state.args.render_mode == "single_prompt":
        raise ValueError("relay_compact requires render_mode relay or auto")
    resolved_context = json.loads(state.context.resolved_context.read_text(encoding="utf-8-sig"))
    subject_anchor = str(resolved_context.get("subject", "")).strip()
    if not subject_anchor:
        raise ValueError(f"No subject anchor found in {state.context.resolved_context}")
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
    state.plan_for_next_step = RelayDirectionBuilder(
        llm=llm,
        subject_anchor=subject_anchor,
    ).compact_render_plan_file(
        input_render_plan=state.plan_for_next_step,
        output_render_plan=state.context.compact_plan,
    )


def _run_anchor_fix_stage(state: PipelineRunState) -> None:
    state.context.artifact_layout.plans_dir.mkdir(parents=True, exist_ok=True)
    resolved_context = json.loads(state.context.resolved_context.read_text(encoding="utf-8-sig"))
    subject_anchor = str(resolved_context.get("subject", "")).strip()
    if not subject_anchor:
        raise ValueError(f"No subject anchor found in {state.context.resolved_context}")

    state.plan_for_next_step = LTXPromptAnchorFixer(subject_anchor=subject_anchor).fix_file(
        input_render_plan=state.plan_for_next_step,
        output_render_plan=state.context.anchored_plan,
    )
    warnings = validate_anchor_file(state.plan_for_next_step, subject_hint=subject_anchor)
    for warning in warnings[:30]:
        _report(f"! {warning}")

