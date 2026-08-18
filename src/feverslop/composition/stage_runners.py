from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import subprocess
from tempfile import NamedTemporaryFile

from rich.console import Console

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.adapters.project_visual_consistency import (
    ProjectReferenceManifestAdapter,
    validate_project_scene_artifacts,
)
from feverslop.adapters.prepared_workflow import (
    PreparedWorkflowRenderer,
    WorkflowMaterializationRequest,
    WorkflowMaterializer,
)
from feverslop.adapters.postprocessor_frame_extractor import (
    PostprocessorFrameExtractor,
)
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.application.continuity_handoff import ContinuityHandoffUseCase
from feverslop.application.h3_prompt_pipeline import H3PromptPipeline
from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts
from feverslop.application.openshot_exporter import export_render_plan_to_openshot
from feverslop.application.mlt_exporter import export_render_plan_to_mlt
from feverslop.application.reference_bible import enrich_render_plan_with_reference_sheets
from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.application.render_video import RenderVideoScenesRequest
from feverslop.application.visual_consistency_preflight import (
    VisualConsistencyPreflightResult,
    preflight_visual_consistency,
    resolve_preflight_workflow_profile,
)
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.domain.render_plan import RenderPlan, RenderScene
from feverslop.domain.visual_consistency import (
    PreflightMode,
    SceneConsistencyContract,
    can_handoff,
    expand_handoff_selection,
    validate_scene_sequence,
)
from feverslop.composition.generate_render_plan import build_generate_render_plan_use_case  # noqa: F401
from feverslop.composition.generate_render_plan import execute_generate_render_plan
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
from feverslop.config.app_config import AppConfig
from feverslop.ports.rendering import WorkflowAnchorConfig
from feverslop.adapters.reporting import ConsoleReporter
from feverslop.prompting.ltx_prompt_anchor_fixer import LTXPromptAnchorFixer, validate_anchor_file
from feverslop.utils.io import file_is_valid
from feverslop.utils.rich_progress import build_progress
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder, build_dspy_generator
from feverslop.prompting.h3_prompt_builder import H3PromptBuilder
from feverslop.prompting.model_types import resolve_model_type
from feverslop.prompting.relay_direction_builder import RelayDirectionBuilder
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.tools.reference_bible import build_arg_parser as build_reference_bible_arg_parser
from feverslop.tools.reference_bible import run as render_reference_bible
from feverslop.tools.storyboard_page import parse_scene_list
from feverslop.tools.storyboard_page import generate_storyboard_page

from .arg_parser import PipelineStage
from .config_loader import PipelineRunContext, PipelineRunState, count_render_plan_items, runner_root

VIDEO_SCENE_PROGRESS_LABEL = "Rendering video scenes"

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


console = Console()


class RenderProgressReporter:
    def __init__(
        self,
        description: str,
        total: int,
        *,
        console: Console = console,
        emit_scene_progress: bool = False,
    ):
        self.description = description
        self.total = total
        self.emit_scene_progress = emit_scene_progress
        self.progress = build_progress(console=console)
        self.task_id = None

    def __enter__(self) -> RenderProgressReporter:
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.progress.__exit__(exc_type, exc_value, traceback)

    def update(self, _output_path: Path, completed: int, total: int) -> None:
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=completed)
        if self.emit_scene_progress:
            console.print(f"Rendered scene {completed}/{total}")

    def analysis_attempt(self, scene_id: int, references: list[dict[str, str]]) -> None:
        summary = ", ".join(f"{item['type']}:{item['id']}" for item in references)
        console.print(f"Ingredients image analysis: scene {scene_id}; {len(references)} references [{summary}]")
        if self.task_id is not None:
            self.progress.update(self.task_id, description=f"Analyzing scene {scene_id}: {summary}")


def _run_tests_stage(_state: PipelineRunState) -> None:
    run_unittest_suite()


def _selected_video_workflows(state: PipelineRunState) -> tuple[Path, ...]:
    if state.args.video_pipeline == "ltx_msr":
        candidates = (state.msr_workflow,)
    elif state.args.video_pipeline == "ltx_ingredients":
        candidates = (state.ingredients_workflow,)
    elif state.args.render_mode == "single_prompt":
        workflow = state.single_prompt_workflow
        if (
            state.args.video_pipeline == "minimax-h3-r2v"
            and Path(workflow).name == "video_ltxv_i2v_v2.json"
        ):
            workflow = Path("workflows/video_minimax_h3_r2v_v1.json")
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
            f"(render_mode={rm!r}); checked: {', '.join(checked)}"
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
        ),
        console=console,
        resolution=resolution,
    )
    state.plan_for_next_step = state.context.render_plan


def _read_h3_input(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot run h3_prompts: missing {label} artifact {path}. "
            "Run the main_pipeline once to create the upstream prompt artifacts."
        )
    return JsonArtifactStore().read_json(path)


def _select_pipeline_scenes(scenes: list[dict], scene_spec: str | None) -> list[dict]:
    selected = parse_scene_list(scene_spec)
    if selected is None:
        return scenes
    return [scene for scene in scenes if int(scene.get("scene") or scene.get("scene_number")) in selected]


def _report_reference_fallbacks(warnings: list[str]) -> None:
    for warning in warnings:
        console.print(f"[yellow]Reference fallback:[/yellow] {warning}")


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
        if not references.get("actor_ids"):
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
                f"using {' and '.join(fallback_fields)} from the existing reference bible."
            )
    if changed:
        store.write_json(plan_path, plan)
    return warnings


def _discover_stem_files(
    stems_dir: Path,
    input_audio: Path | None = None,
) -> dict[str, Path] | None:
    if not stems_dir.is_dir():
        return None
    stem_files = {}
    supported_suffixes = (".wav", ".mp3", ".flac")
    if input_audio is not None:
        input_stem = Path(input_audio).stem
        for stem_name in ("vocals", "drums", "bass", "other"):
            expected_stem = f"{stem_name}_{input_stem}"
            matches = sorted(
                path
                for path in stems_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in supported_suffixes
                and path.stem == expected_stem
            )
            if matches:
                stem_files[stem_name] = next(
                    (path for path in matches if path.suffix.lower() == ".wav"),
                    matches[0],
                )
    else:
        for stem_file in sorted(stems_dir.iterdir()):
            if stem_file.is_file() and stem_file.suffix.lower() in supported_suffixes:
                stem_files[stem_file.stem.split("_", 1)[0]] = stem_file
    return stem_files or None


def _merge_reference_paths_into_h3_segments(
    stage1_segments: list[dict],
    reference_plan_path: Path,
) -> list[dict]:
    if not reference_plan_path.is_file():
        raise FileNotFoundError(
            f"Cannot run h3_prompts: missing reference-enriched render plan {reference_plan_path}. "
            "Run --stage msr_reference_sheets first."
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
    selected, stage1_segments = _select_h3_segments(stage1_segments, selected_scene_spec)
    if state.args.video_pipeline == "minimax-h3-r2v":
        _report_reference_fallbacks(_seed_reference_bindings(state.plan_for_next_step, config))
        state.plan_for_next_step = enrich_render_plan_with_reference_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.reference_plan,
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
        console.print(
            f"[cyan]Scene selection active: H3 generation limited to "
            f"{', '.join(str(number) for number in sorted(selected))}[/cyan]"
        )
    global_context = _read_h3_input(state.context.resolved_context, "resolved context")
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
        stem_files=_discover_stem_files(paths.stems_dir, config.input_audio),
        ltx_prompt_relay_json=paths.prompts_dir / f"ltx_prompt_relay_{config.song_id}.json",
        beat_json=paths.timeline_dir / f"beat_data_{config.song_id}.json",
    )
    pipeline = H3PromptPipeline(
        llm_factory=lambda current_config: OpenAICompatibleLLMClient(
            base_url=current_config.llm.base_url,
            api_key=current_config.llm.api_key,
            model=current_config.llm.model_for("structured"),
            temperature=current_config.llm.temperature,
            dspy_temperature=current_config.llm.dspy_temperature,
            max_tokens=current_config.llm.max_tokens,
            request_timeout_seconds=current_config.llm.request_timeout_seconds,
            dspy_cache=getattr(current_config.llm, "dspy_cache", False),
            max_concurrent_requests=current_config.llm.max_concurrent_requests,
        ),
        h3_prompt_builder_factory=H3PromptBuilder,
        dspy_prompt_builder_factory=lambda llm: DspyH3PromptBuilder(
            build_dspy_generator(llm),
            reference_root=paths.project_dir,
            allow_fallback=False,
        ),
    )
    pipeline.execute(context)
    state.plan_for_next_step = state.context.render_plan
    console.print(f"[green]OK H3 Prompts JSON: {h3_prompts_json}[/green]")


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
    )
    if config.video_pipeline == "minimax-h3-r2v":
        _preserve_enriched_reference_paths(
            output_path=state.context.render_plan,
            reference_plan_path=state.context.reference_plan,
        )
    selected_scene_spec = getattr(state.args, "scenes", None)
    if selected_scene_spec:
        selected_plan = _select_pipeline_scenes(
            JsonArtifactStore().read_json(state.context.render_plan),
            selected_scene_spec,
        )
        JsonArtifactStore().write_json(state.context.render_plan, selected_plan)
        console.print(
            f"[cyan]Scene selection active: render plan limited to "
            f"{len(selected_plan)} selected scene(s)[/cyan]"
        )
    state.plan_for_next_step = state.context.render_plan
    console.print(f"[green]OK Render Plan JSON: {state.plan_for_next_step}[/green]")


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
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        api_key=app_config.llm.api_key,
        model=app_config.llm.model_for("structured"),
        temperature=app_config.llm.temperature,
        dspy_temperature=app_config.llm.dspy_temperature,
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        max_concurrent_requests=app_config.llm.max_concurrent_requests,
    )
    state.plan_for_next_step = RelayDirectionBuilder(llm=llm).compact_render_plan_file(
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
        console.print(f"! {warning}")


def _run_set_resolution_stage(state: PipelineRunState) -> None:
    """Persist new resolution to config.json and render plan, then re-prepare workflows.

    Does NOT render anything; purely updates prep files so a later render
    picks up the new resolution automatically.
    """
    set_res = getattr(state.args, "set_resolution", None)
    if set_res is None:
        raise ValueError("--set-resolution WxH is required for the set_resolution stage")

    width = set_res.width
    height = set_res.height
    console.print(f"Setting resolution to {width}x{height}...")

    # 1. Patch config.json
    ProjectConfig.set_resolution_on_disk(
        state.context.project_config_path,
        width=width,
        height=height,
    )
    console.print(f"[green]Updated config.json resolution to {width}x{height}[/green]")

    # 2. Patch render plan resolution
    #    - Music projects: list of scenes, each with width/height fields
    #    - Movie projects: dict with top-level resolution field
    # Update all render plans in the plans directory, not just the active one.
    render_plan_path = state.plan_for_next_step
    plans_dir = render_plan_path.parent if render_plan_path.parent.name == "plans" else None
    target_paths = sorted(plans_dir.glob("*.json")) if plans_dir else [render_plan_path]
    updated_count = 0
    for plan_path in target_paths:
        if not plan_path.is_file():
            continue
        raw = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            for item in raw:
                item["width"] = width
                item["height"] = height
        elif isinstance(raw, dict):
            if "resolution" in raw:
                raw["resolution"] = {"width": width, "height": height}
        plan_path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        updated_count += 1
    console.print(f"[green]Updated {updated_count} render plan(s) to {width}x{height}[/green]")

    # 3. For MSR/ingredients: re-prepare workflows with new resolution
    if state.args.video_pipeline in ("ltx_msr", "ltx_ingredients"):
        console.print("Re-preparing workflows with new resolution...")
        _run_ltx_prepare_workflows_stage(state)
        console.print("[green]Workflows re-prepared.[/green]")
    else:
        console.print(
            f"[green]Resolution updated. Run '--stage ltx_render_scenes' to render at "
            f"{width}x{height}.[/green]"
        )


def _run_storyboard_frames_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "ltx_msr":
        raise ValueError("storyboard_frames is not used by ltx_msr")
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    storyboard_use_case = build_render_storyboard_use_case(
        app_config=app_config,
        workflow_path=state.storyboard_workflow,
        output_dir=state.context.storyboard_dir,
    )
    storyboard_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Rendering storyboard frames", storyboard_total) as storyboard_progress:
        storyboard_use_case.execute(
            RenderStoryboardRequest(
                render_plan_path=state.plan_for_next_step,
                workflow_path=state.storyboard_workflow,
                output_dir=state.context.storyboard_dir,
                character_lora_strength=state.args.storyboard_lora_strength,
                on_frame_complete=storyboard_progress.update,
            )
        )


def _run_storyboard_page_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "ltx_msr":
        raise ValueError("storyboard_page is not used by ltx_msr")
    generate_storyboard_page(
        render_plan_path=state.plan_for_next_step,
        storyboard_dir=state.context.storyboard_dir,
        output_html=state.context.storyboard_page,
    )


def _run_msr_references_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients", "minimax-h3-r2v", "minimax-h3-i2v"):
        raise ValueError(
            "msr_references requires --video-pipeline ltx_msr, ltx_ingredients, minimax-h3-r2v, or minimax-h3-i2v"
        )
    project_config_path = getattr(state.context, "project_config_path", None)
    if project_config_path is not None:
        _report_reference_fallbacks(_seed_reference_bindings(
            state.plan_for_next_step,
            ProjectConfig.load(project_config_path),
        ))
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
        str(getattr(state.args, "sequence_to_sheet_workflow", "workflows/sequence_to_sheet_minimax_h3_i2va_v1.json")),
    ])
    render_reference_bible(reference_args)


def _run_msr_reference_sheets_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients", "minimax-h3-r2v", "minimax-h3-i2v"):
        raise ValueError(
            "msr_reference_sheets requires --video-pipeline ltx_msr, ltx_ingredients, minimax-h3-r2v, or minimax-h3-i2v"
        )
    if not state.plan_for_next_step.is_file():
        console.print(
            "[dim]Render plan missing; creating the intermediate plan before enriching MSR references...[/dim]"
        )
        _run_render_plan_stage(state)
    project_config_path = getattr(state.context, "project_config_path", None)
    if project_config_path is not None:
        _report_reference_fallbacks(_seed_reference_bindings(
            state.plan_for_next_step,
            ProjectConfig.load(project_config_path),
        ))
    state.context.artifact_layout.plans_dir.mkdir(parents=True, exist_ok=True)
    msr_reference_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Enriching MSR references", msr_reference_total) as reference_progress:
        state.plan_for_next_step = enrich_render_plan_with_reference_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.reference_plan,
            on_scene_complete=_scene_progress_callback(reference_progress),
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
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        max_concurrent_requests=app_config.llm.max_concurrent_requests,
    )
    msr_prompt_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Enriching MSR prompts", msr_prompt_total) as msr_prompt_progress:
        state.plan_for_next_step = enrich_render_plan_with_msr_prompts(
            state.plan_for_next_step,
            state.context.reference_plan,
            llm=llm,
            on_analysis_status=msr_prompt_progress.analysis_attempt,
            on_scene_complete=_scene_progress_callback(msr_prompt_progress),
        )


def _run_ingredients_sheets_stage(state: PipelineRunState) -> None:
    from feverslop.application.render_plan_ingredients_sheets import enrich_render_plan_with_ingredients_sheets
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
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        max_concurrent_requests=app_config.llm.max_concurrent_requests,
    )
    state.context.artifact_layout.plans_dir.mkdir(parents=True, exist_ok=True)
    ingredients_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Composing Ingredients scene sheets", ingredients_total) as progress:
        state.plan_for_next_step = enrich_render_plan_with_ingredients_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.ingredients_plan,
            video_settings=video_settings,
            llm=llm,
            on_analysis_status=progress.analysis_attempt,
            on_scene_complete=_scene_progress_callback(progress),
            workflow_profile=str(
                getattr(state.args, "video_workflow_profile", None)
                or state.ingredients_workflow.stem
            ),
        )


def _specialized_video_use_case(state: PipelineRunState):
    workflow = _selected_video_workflows(state)[0]
    return build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            app_config_path=state.app_config_path,
            project_config_path=state.context.project_config_path,
            render_plan_path=state.plan_for_next_step,
            workflow_path=workflow,
            output_dir=state.context.ltx_dir,
            video_pipeline=state.args.video_pipeline,
            video_workflow_profile=getattr(
                state.args,
                "video_workflow_profile",
                None,
            ),
            randomize_seed=state.args.randomize_seed,
            rolling_frame_profile=state.args.rolling_frame_profile,
            resolution=_get_resolution(state.args),
        ),
        console=console,
    )


def _selected_render_scenes(state: PipelineRunState) -> tuple[RenderScene, ...]:
    return _select_render_scenes(state, _all_render_scenes(state))


def _all_render_scenes(state: PipelineRunState) -> tuple[RenderScene, ...]:
    payload = json.loads(state.plan_for_next_step.read_text(encoding="utf-8-sig"))
    validate_scene_sequence(payload)
    return RenderPlan.from_dicts(payload).scenes


def _select_render_scenes(state: PipelineRunState, scenes: tuple[RenderScene, ...]) -> tuple[RenderScene, ...]:
    selected = {state.args.smoke_scene} if state.args.smoke_only else parse_scene_list(state.args.scenes)
    return RenderPlan(scenes).select(scene_numbers=selected).scenes


def _missing_prepare_inputs(state: PipelineRunState, scenes: tuple[RenderScene, ...]) -> list[str]:
    missing: list[str] = []
    for path, label in ((state.plan_for_next_step, "render plan"), (state.context.input_audio, "audio")):
        if not path.is_file():
            missing.append(f"{label}: {path}")
    workflow = state.msr_workflow if state.args.video_pipeline == "ltx_msr" else state.ingredients_workflow
    if not workflow.is_file():
        missing.append(f"workflow template: {workflow}")
    for render_scene in scenes:
        scene = render_scene.to_dict()
        number = render_scene.scene_number
        for relay in (scene.get("ltx") or {}).get("prompt_relay") or []:
            if str(relay.get("state") or "").strip().lower() != "singing":
                continue
            prompt = str(relay.get("prompt") or "").lower()
            if "sing" not in prompt or ("lip sync" not in prompt and "lip-sync" not in prompt):
                missing.append(f"scene {number}: singing relay requires singing and lip sync")
        candidates: list[tuple[str, str]] = []
        if state.args.video_pipeline == "ltx_ingredients":
            ingredients = scene.get("ingredients") or {}
            sheet = ingredients.get("sheet_path") or scene.get("ingredients_scene_sheet")
            if sheet:
                candidates.append(("ingredients sheet", sheet))
            else:
                missing.append(f"scene {number}: ingredients_scene_sheet")
            anchors = ingredients.get("anchors") or scene.get("ingredients_scene_sheet_anchors") or []
            target = str(
                ingredients.get("global_prompt")
                or scene.get("ingredients_global_prompt")
                or scene.get("ingredients_target_prompt")
                or (scene.get("ltx") or {}).get("ingredients_target_prompt")
                or ""
            )
            references = scene.get("references") or {}
            expected_ids = {
                str(value) for value in references.get("actor_ids") or [] if str(value)
            }
            location_id = str(references.get("location_id") or "")
            if location_id:
                expected_ids.add(location_id)
            anchor_ids = {str(anchor.get("id") or "") for anchor in anchors}
            if expected_ids and expected_ids != anchor_ids:
                missing.append(f"scene {number}: ingredients anchors do not match actor/location bindings")
            unbound = sorted(item_id for item_id in anchor_ids if f"`{item_id}`" not in target)
            if unbound:
                missing.append(
                    f"scene {number}: global prompt does not bind anchors {', '.join(unbound)}"
                )
        else:
            references = scene.get("references") or {}
            actors = references.get("actor_msr_paths") or references.get("actor_sheet_paths") or []
            location = references.get("location_msr_path") or references.get("location_sheet_path")
            if not actors:
                missing.append(f"scene {number}: actor reference sheet")
            if not location:
                missing.append(f"scene {number}: location reference sheet")
            candidates.extend(("actor reference sheet", path) for path in actors)
            if location:
                candidates.append(("location reference sheet", location))
        for label, value in candidates:
            path = Path(value)
            path = path if path.is_absolute() else state.context.project_config_dir / path
            if not path.is_file():
                missing.append(f"scene {number} {label}: {path}")
    return missing


def _run_ltx_prepare_workflows_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients"):
        raise ValueError("ltx_prepare_workflows requires --video-pipeline ltx_msr or ltx_ingredients")
    all_scenes = _all_render_scenes(state) if state.plan_for_next_step.is_file() else []
    preflight = _run_visual_consistency_preflight(state, all_scenes)
    all_scenes = _project_visual_consistency_contracts(
        all_scenes,
        preflight,
    )
    scenes = _select_render_scenes(state, all_scenes)
    profile = _resolved_startframe_profile(state)
    handoff_predecessors = _music_handoff_predecessors(
        all_scenes,
        profile=profile,
    )
    if handoff_predecessors:
        selected_numbers = expand_handoff_selection(
            [
                contract
                for scene in all_scenes
                if (
                    contract := _stored_consistency_contract(
                        scene.to_dict()
                    )
                )
                is not None
            ],
            {scene.scene_number for scene in scenes},
        )
        scenes = RenderPlan(all_scenes).select(
            scene_numbers=selected_numbers,
        ).scenes
    missing = _missing_prepare_inputs(state, scenes)
    if missing:
        raise FileNotFoundError("Cannot prepare scene workflows; missing inputs:\n- " + "\n- ".join(missing))
    backend = _specialized_video_use_case(state).backend
    materializer = WorkflowMaterializer(backend, state.context.artifact_layout)
    total = len(scenes)
    paths = [
        path
        for scene in scenes
        for path in (
            state.context.artifact_layout.scene_workflow(scene.scene_number),
            state.context.artifact_layout.scene_manifest(scene.scene_number),
        )
    ]
    previous = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        for completed, render_scene in enumerate(scenes, start=1):
            if render_scene.scene_number in handoff_predecessors:
                console.print(
                    f"Deferred scene {completed}/{total}: "
                    f"{render_scene.scene_number} (awaiting predecessor handoff)"
                )
                continue
            materializer.prepare(WorkflowMaterializationRequest(
                scene=render_scene.to_dict(),
                prompt=render_scene.video_prompt,
                audio_file=state.context.input_audio,
                render_plan_path=state.plan_for_next_step,
                pipeline=state.args.video_pipeline,
            ))
            console.print(f"Prepared scene {completed}/{total}: {render_scene.scene_number}")
    except Exception:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise


def _attach_music_continuity_handoffs(
    state: PipelineRunState,
    *,
    all_scenes: list,
    selected_scenes: list,
    backend,
) -> list:
    if state.args.video_pipeline != "ltx_msr":
        return selected_scenes
    profile = _resolved_startframe_profile(state)
    if profile is None or not profile.supports_start_frame:
        return selected_scenes

    all_payloads = [scene.to_dict() for scene in all_scenes]
    by_number = {int(scene["scene"]): scene for scene in all_payloads}
    selected_numbers = {scene.scene_number for scene in selected_scenes}
    explicitly_selected = bool(
        {state.args.smoke_scene}
        if state.args.smoke_only
        else parse_scene_list(state.args.scenes)
    )
    attached = []
    for render_scene in selected_scenes:
        scene = render_scene.to_dict()
        number = render_scene.scene_number
        previous = by_number.get(number - 1)
        previous_contract = _stored_consistency_contract(previous)
        current_contract = _stored_consistency_contract(scene)
        if (
            previous_contract is None
            or current_contract is None
            or previous_contract.scene + 1 != current_contract.scene
            or not can_handoff(previous_contract, current_contract)
        ):
            attached.append(render_scene)
            continue
        previous_clip = state.context.artifact_layout.scene_final_video(number - 1)
        if not previous_clip.is_file() and not explicitly_selected:
            attached.append(render_scene)
            continue
        output_frame = (
            state.context.render_dir
            / "keyframes"
            / f"scene_{number - 1:04}_to_{number:04}_start.png"
        )
        scene = ContinuityHandoffUseCase(
            PostprocessorFrameExtractor(
                backend.postprocessor,
                project_dir=state.context.project_config_dir,
                selected_rerender=explicitly_selected
                and number in selected_numbers,
            )
        ).execute(
            previous_contract,
            current_contract,
            previous_clip,
            output_frame,
            scene,
            handoff_prompt=_music_handoff_prompt(previous),
        )
        attached.append(type(render_scene).from_dict(scene))
    return attached


def _stored_consistency_contract(scene: dict | None):
    payload = scene.get("visual_consistency") if scene else None
    if not isinstance(payload, dict):
        return None
    try:
        return SceneConsistencyContract.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def _music_handoff_prompt(previous_scene: dict | None) -> str:
    ltx = (previous_scene or {}).get("ltx") or {}
    relays = ltx.get("msr_prompt_relay") or ltx.get("prompt_relay") or []
    if relays and isinstance(relays[-1], dict):
        prompt = str(relays[-1].get("prompt") or "").strip()
        if prompt:
            return prompt
    return str(
        ltx.get("original_style_i2v_prompt")
        or ltx.get("base_prompt")
        or ""
    ).strip()


_PROFILE_UNSET = object()


def _resolved_startframe_profile(
    state: PipelineRunState,
    profile_name: str | None | object = _PROFILE_UNSET,
):
    if state.args.video_pipeline != "ltx_msr":
        return None
    app_config = AppConfig.load(state.app_config_path)
    selected_name = (
        profile_name
        if profile_name is not _PROFILE_UNSET
        else getattr(state.args, "video_workflow_profile", None)
    )
    if profile_name is not _PROFILE_UNSET and selected_name is not None and not any(
        profile.name == selected_name
        for profile in app_config.video_workflow_profiles
    ):
        return None
    profile = app_config.resolve_video_workflow_profile(
        pipeline="ltx_msr",
        purpose="final",
        name=selected_name,
    )
    if profile is not None and profile.supports_start_frame:
        configured = Path(profile.workflow_path).resolve()
        materialized = Path(state.msr_workflow).resolve()
        if configured != materialized:
            raise ValueError(
                "Configured start-frame profile workflow does not match "
                f"the materialized MSR workflow: {configured} != {materialized}"
            )
    return profile


def _continuity_downstream(
    scene_number: int,
    predecessors: dict[int, int],
) -> set[int]:
    downstream: set[int] = set()
    current = scene_number
    while True:
        dependent = next(
            (
                candidate
                for candidate, predecessor in predecessors.items()
                if predecessor == current
            ),
            None,
        )
        if dependent is None:
            return downstream
        if dependent in downstream:  # cycle detection
            return downstream
        downstream.add(dependent)
        current = dependent


def _load_continuity_dirty(
    path: Path,
    scene_numbers: set[int],
) -> set[int]:
    if not path.exists() and not path.is_symlink():
        return set()
    if path.is_symlink() or not path.is_file():
        return set(scene_numbers)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dirty = payload["dirty_scenes"]
        if (
            payload.get("schema") != "feverslop.continuity-dirty/v1"
            or not isinstance(dirty, list)
            or any(type(number) is not int for number in dirty)
            or not set(dirty).issubset(scene_numbers)
        ):
            raise ValueError("invalid continuity dirty marker")
        return set(dirty)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return set(scene_numbers)


def _write_continuity_dirty(
    path: Path,
    *,
    dirty_scenes: set[int],
    predecessor_scene: int,
    predecessor_contract: SceneConsistencyContract | None,
    predecessor_output: Path,
    project_dir: Path,
) -> None:
    project_root = Path(project_dir).resolve()
    marker = path.resolve()
    output = Path(predecessor_output).resolve()
    if (
        not marker.is_relative_to(project_root)
        or not output.is_relative_to(project_root)
    ):
        raise ValueError("Continuity dirty state must remain inside project")
    relative_output = output.relative_to(project_root).as_posix()
    payload = {
        "schema": "feverslop.continuity-dirty/v1",
        "dirty_scenes": sorted(dirty_scenes),
        "predecessor": {
            "scene": predecessor_scene,
            "fingerprint": (
                predecessor_contract.fingerprint
                if predecessor_contract is not None
                else None
            ),
            "output": {
                "path": relative_output,
                "sha256": (
                    hashlib.sha256(output.read_bytes()).hexdigest()
                    if output.is_file()
                    else None
                ),
            },
        },
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=marker.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def _run_visual_consistency_preflight(
    state: PipelineRunState,
    scenes: list,
) -> VisualConsistencyPreflightResult:
    preflight_mode = getattr(
        state.args,
        "visual_consistency_preflight",
        PreflightMode.WARN,
    )
    preflight_mode = PreflightMode.parse(preflight_mode)
    if preflight_mode is PreflightMode.OFF:
        return VisualConsistencyPreflightResult((), ())
    project_config = ProjectConfig.load(state.context.project_config_path)
    snapshot = ProjectReferenceManifestAdapter(
        lambda _project_id: state.context.project_config_dir
    ).load(state.context.project_config_dir.name)
    mode = "msr" if state.args.video_pipeline == "ltx_msr" else "ingredients"
    workflow = (
        state.msr_workflow
        if state.args.video_pipeline == "ltx_msr"
        else state.ingredients_workflow
    )
    scene_payloads = [scene.to_dict() for scene in scenes]
    workflow_profile = resolve_preflight_workflow_profile(
        scene_payloads,
        explicit_profile=getattr(state.args, "video_workflow_profile", None),
        legacy_fallback=workflow.stem,
    )
    selected_profile = (
        _resolved_startframe_profile(state, workflow_profile)
        if mode == "msr"
        else None
    )
    result = preflight_visual_consistency(
        scene_payloads,
        snapshot,
        mode=mode,
        workflow_profile=workflow_profile,
        preflight_mode=preflight_mode,
        subject_mode=project_config.subject_mode,
        max_scene_actors=project_config.max_scene_actors,
        supports_continuous_transitions=(
            mode == "msr"
            and selected_profile is not None
            and selected_profile.supports_start_frame
        ),
    )
    artifact_issues = validate_project_scene_artifacts(
        state.context.project_config_dir,
        scene_payloads,
        mode=mode,
        preflight_mode=preflight_mode,
    )
    result = VisualConsistencyPreflightResult(
        result.contracts,
        (*result.issues, *artifact_issues),
    )
    for issue in result.issues:
        console.print(
            f"Visual consistency {issue.severity.upper()} "
            f"scene {issue.scene} {issue.code}: {issue.message}"
        )
    if not result.renderable:
        details = "\n- ".join(
            f"{issue.code}: {issue.message}"
            for issue in result.issues
            if issue.severity == "error"
        )
        raise ValueError(
            "Visual consistency preflight blocked workflow preparation:\n- "
            + details
        )
    return result


def _project_visual_consistency_contracts(
    scenes: list,
    result: VisualConsistencyPreflightResult,
) -> list:
    """Attach canonical preflight contracts to in-memory render scenes only."""
    if not isinstance(result, VisualConsistencyPreflightResult):
        return list(scenes)
    contracts = {contract.scene: contract for contract in result.contracts}
    projected = []
    for scene in scenes:
        payload = scene.to_dict()
        contract = contracts.get(scene.scene_number)
        if contract is not None:
            payload["visual_consistency"] = contract.to_dict()
        projected.append(type(scene).from_dict(payload))
    return projected


def _music_handoff_predecessors(
    scenes: list,
    *,
    profile,
) -> dict[int, int]:
    if profile is None or not profile.supports_start_frame:
        return {}
    predecessors: dict[int, int] = {}
    payloads = [scene.to_dict() for scene in scenes]
    for previous_scene, current_scene in zip(payloads, payloads[1:]):
        previous_contract = _stored_consistency_contract(previous_scene)
        current_contract = _stored_consistency_contract(current_scene)
        if (
            previous_contract is not None
            and current_contract is not None
            and previous_contract.scene + 1 == current_contract.scene
            and can_handoff(previous_contract, current_contract)
        ):
            predecessors[current_contract.scene] = previous_contract.scene
    return predecessors


def _run_ltx_render_scenes_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "minimax-h3-r2v":
        # An explicit --stage list may place rendering before
        # msr_reference_sheets. Refresh the plan from the existing enriched
        # plan immediately before scene validation so actor paths cannot be
        # lost merely because the stages were listed in that order.
        _preserve_enriched_reference_paths(
            output_path=state.plan_for_next_step,
            reference_plan_path=state.context.reference_plan,
        )
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients") and state.args.render_mode != "single_prompt" and not str(state.args.relay_workflow).strip():
        raise ValueError(f"RenderMode '{state.args.render_mode}' requires --relay-workflow pointing to a workflow with #PROMPT_RELAY.")

    if state.args.video_pipeline in ("ltx_msr", "ltx_ingredients"):
        all_scenes = _all_render_scenes(state)
        preflight = _run_visual_consistency_preflight(state, all_scenes)
        all_scenes = _project_visual_consistency_contracts(
            all_scenes,
            preflight,
        )
        scenes = _select_render_scenes(state, all_scenes)
        profile = _resolved_startframe_profile(state)
        handoff_predecessors = _music_handoff_predecessors(
            all_scenes,
            profile=profile,
        )
        if profile is not None and profile.supports_start_frame:
            contracts = [
                contract
                for scene in all_scenes
                if (
                    contract := _stored_consistency_contract(
                        scene.to_dict()
                    )
                )
                is not None
            ]
            selected_numbers = expand_handoff_selection(
                contracts,
                {scene.scene_number for scene in scenes},
            )
            scenes = RenderPlan(all_scenes).select(
                scene_numbers=selected_numbers
            ).scenes
        missing: list[Path] = []
        for scene in scenes:
            workflow_path = state.context.artifact_layout.scene_workflow(scene.scene_number)
            manifest_path = state.context.artifact_layout.scene_manifest(scene.scene_number)
            deferred = scene.scene_number in handoff_predecessors
            if not workflow_path.is_file() and (
                not deferred or manifest_path.is_file()
            ):
                missing.append(workflow_path)
            if not manifest_path.is_file() and (
                not deferred or workflow_path.is_file()
            ):
                missing.append(manifest_path)
        if missing:
            raise FileNotFoundError(
                "Missing prepared scene workflows: " + ", ".join(str(path) for path in missing)
                + ". Run --stage ltx_prepare_workflows first."
            )
        backend = _specialized_video_use_case(state).backend
        active_workflow_profile = (
            profile.name
            if profile is not None
            else resolve_preflight_workflow_profile(
                [scene.to_dict() for scene in all_scenes],
                explicit_profile=getattr(
                    state.args,
                    "video_workflow_profile",
                    None,
                ),
                legacy_fallback=(
                    state.msr_workflow
                    if state.args.video_pipeline == "ltx_msr"
                    else state.ingredients_workflow
                ).stem,
            )
        )
        renderer = PreparedWorkflowRenderer(
            project_dir=state.context.project_config_dir,
            render_queue=backend.render_queue,
            postprocessor=backend.postprocessor,
            expected_pipeline=state.args.video_pipeline,
            expected_workflow_profile=active_workflow_profile,
            max_render_frames=backend.max_render_frames,
            max_render_duration_seconds=backend.max_render_duration_seconds,
            render_budget_workflow_path=backend.render_budget_workflow_path,
            round_render_frames_to_8n1=backend.round_render_frames_to_8n1,
            asset_uploader=backend.asset_uploader,
            model_resolver=backend.model_resolver,
            model_workflow_path=backend.workflow_label,
        )
        total = len(scenes)
        materializer = WorkflowMaterializer(
            backend,
            state.context.artifact_layout,
        )
        explicit_selection = bool(
            {state.args.smoke_scene}
            if state.args.smoke_only
            else parse_scene_list(state.args.scenes)
        )
        rendered_this_run: set[int] = set()
        dirty_marker = (
            state.context.render_dir / "continuity_dirty.json"
        )
        persisted_dirty = _load_continuity_dirty(
            dirty_marker,
            {scene.scene_number for scene in all_scenes},
        )
        with RenderProgressReporter(
            "Rendering prepared LTX scenes", total, emit_scene_progress=True
        ) as progress:
            for completed, scene in enumerate(scenes, start=1):
                workflow = state.context.artifact_layout.scene_workflow(scene.scene_number)
                final_path = state.context.artifact_layout.scene_final_video(scene.scene_number)
                predecessor_rendered = (
                    handoff_predecessors.get(scene.scene_number)
                    in rendered_this_run
                )
                skip_existing = (
                    False
                    if (
                        state.args.smoke_only
                        or explicit_selection
                        or predecessor_rendered
                        or scene.scene_number in persisted_dirty
                        or not workflow.is_file()
                    )
                    else not state.args.no_skip_existing
                )
                if not (skip_existing and file_is_valid(final_path)):
                    randomize_seed = bool(getattr(backend, "randomize_seed", False))
                    if randomize_seed:
                        scene_payload = scene.to_dict()
                        scene_payload["seed"] = random.SystemRandom().randint(0, 2**63 - 1)
                        plan_data = JsonArtifactStore().read_render_plan(state.plan_for_next_step)
                        plan_data = [
                            (
                                scene_payload
                                if int(item["scene"]) == scene.scene_number
                                else item
                            )
                            for item in plan_data
                        ]
                        JsonArtifactStore().write_render_plan(
                            state.plan_for_next_step,
                            plan_data,
                        )
                        scene = RenderPlan.from_dicts([scene_payload]).scenes[0]
                    downstream = _continuity_downstream(
                        scene.scene_number,
                        handoff_predecessors,
                    )
                    if downstream:
                        persisted_dirty.update(downstream)
                        _write_continuity_dirty(
                            dirty_marker,
                            dirty_scenes=persisted_dirty,
                            predecessor_scene=scene.scene_number,
                            predecessor_contract=(
                                _stored_consistency_contract(
                                    scene.to_dict()
                                )
                            ),
                            predecessor_output=final_path,
                            project_dir=state.context.project_config_dir,
                        )
                    [render_scene] = _attach_music_continuity_handoffs(
                        state,
                        all_scenes=all_scenes,
                        selected_scenes=[scene],
                        backend=backend,
                    )
                    original_randomize_seed = backend.randomize_seed
                    if randomize_seed:
                        backend.randomize_seed = False
                    try:
                        if randomize_seed or render_scene.to_dict() != scene.to_dict():
                            materializer.prepare(
                                WorkflowMaterializationRequest(
                                    scene=render_scene.to_dict(),
                                    prompt=render_scene.video_prompt,
                                    audio_file=state.context.input_audio,
                                    render_plan_path=state.plan_for_next_step,
                                    pipeline=state.args.video_pipeline,
                                )
                            )
                        final_path = renderer.render(workflow)
                    finally:
                        backend.randomize_seed = original_randomize_seed
                    rendered_this_run.add(scene.scene_number)
                    if downstream:
                        _write_continuity_dirty(
                            dirty_marker,
                            dirty_scenes=persisted_dirty,
                            predecessor_scene=scene.scene_number,
                            predecessor_contract=(
                                _stored_consistency_contract(
                                    scene.to_dict()
                                )
                            ),
                            predecessor_output=final_path,
                            project_dir=state.context.project_config_dir,
                        )
                    if scene.scene_number in persisted_dirty:
                        persisted_dirty.remove(scene.scene_number)
                        if persisted_dirty:
                            _write_continuity_dirty(
                                dirty_marker,
                                dirty_scenes=persisted_dirty,
                                predecessor_scene=scene.scene_number,
                                predecessor_contract=(
                                    _stored_consistency_contract(
                                        scene.to_dict()
                                    )
                                ),
                                predecessor_output=final_path,
                                project_dir=state.context.project_config_dir,
                            )
                        else:
                            dirty_marker.unlink(missing_ok=True)
                else:
                    manifest = SceneWorkflowManifest.read(state.context.artifact_layout.scene_manifest(scene.scene_number))
                    if manifest.pipeline != state.args.video_pipeline:
                        raise ValueError(
                            f"Prepared workflow pipeline {manifest.pipeline!r} does not match "
                            f"expected pipeline {state.args.video_pipeline!r}"
                        )
                    mismatches = manifest.verify(state.context.project_config_dir)
                    if mismatches:
                        raise ValueError("Prepared workflow verification failed: " + "; ".join(mismatches))
                progress.update(final_path, completed, total)
        return

    selected_workflows = _selected_video_workflows(state)
    if state.args.video_pipeline in ("ltx_msr", "ltx_ingredients"):
        ltx_workflow = selected_workflows[0]
        ltx_single_prompt_workflow = None
    else:
        ltx_workflow = selected_workflows[0]
        ltx_single_prompt_workflow = selected_workflows[1] if state.args.render_mode == "auto" else None
    video_use_case = build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            app_config_path=state.app_config_path,
            project_config_path=state.context.project_config_path,
            render_plan_path=state.plan_for_next_step,
            workflow_path=ltx_workflow,
            output_dir=state.context.ltx_dir,
            video_pipeline=state.args.video_pipeline,
            single_prompt_workflow_path=ltx_single_prompt_workflow,
            render_mode=state.args.render_mode,
            single_prompt_title=state.args.single_prompt_title,
            single_prompt_input=state.args.single_prompt_input,
            character_lora_strength=state.args.video_character_lora_strength,
            lora_1_strength_model=state.args.video_lora_1_strength_model,
            lora_1_strength_clip=state.args.video_lora_1_strength_clip,
            lora_split_enabled=state.args.lora_split_enabled,
            randomize_seed=state.args.randomize_seed,
            debug_workflows_dir=state.context.ltx_debug_dir,
            rolling_frame_profile=state.args.rolling_frame_profile,
            resolution=_get_resolution(state.args),
        ),
        console=console,
    )
    ltx_scene_numbers = {state.args.smoke_scene} if state.args.smoke_only else parse_scene_list(state.args.scenes)
    ltx_total = count_render_plan_items(state.plan_for_next_step, scene_numbers=ltx_scene_numbers)
    with RenderProgressReporter(
        VIDEO_SCENE_PROGRESS_LABEL, ltx_total, emit_scene_progress=True
    ) as ltx_progress:
        video_use_case.execute(
            RenderVideoScenesRequest(
                render_plan_path=state.plan_for_next_step,
                workflow_path=ltx_workflow,
                audio_file=state.context.input_audio,
                storyboard_dir=state.context.storyboard_dir,
                output_dir=state.context.ltx_dir,
                render_mode=state.args.render_mode,
                single_prompt_workflow_path=ltx_single_prompt_workflow,
                scene_numbers=ltx_scene_numbers,
                skip_existing=False if state.args.smoke_only else not state.args.no_skip_existing,
                anchors=WorkflowAnchorConfig(
                    single_prompt_title=state.args.single_prompt_title,
                    single_prompt_input=state.args.single_prompt_input,
                ),
                on_scene_complete=ltx_progress.update,
            )
        )


def _run_concat_video_only_stage(state: PipelineRunState) -> None:
    from .config_loader import (
        collect_render_plan_scene_clips,
        collect_render_plan_scene_raw_clips,
        rewrite_concat_list,
        write_concat_list,
    )
    layout = state.context.artifact_layout
    render_plan = json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig"))
    scene_numbers = [int(entry["scene"]) for entry in render_plan]
    canonical_clips = [layout.scene_final_video(scene_number) for scene_number in scene_numbers]
    canonical_available = [clip for clip in canonical_clips if clip.is_file()]
    if canonical_available and len(canonical_available) != len(canonical_clips):
        missing = [clip.parent.name for clip in canonical_clips if not clip.is_file()]
        raise FileNotFoundError(
            "Cannot build base variant without mixing artifact layouts; missing canonical "
            f"final.mp4 for {', '.join(missing[:10])}"
        )
    if canonical_available:
        clips = canonical_clips
    else:
        console.print(
            "[yellow]No canonical final.mp4 scene artifacts found; using the complete legacy "
            "scene layout for the base movie.[/yellow]"
        )
        clips = collect_render_plan_scene_clips(
            state.plan_for_next_step,
            state.context.ltx_dir,
            layout=None,
        )
    rewrite_concat_list(clips, state.context.artifact_layout.final_dir)
    postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
    console.print(f"Concatenating base variant: {len(clips)} scene clips")
    state.video_only_path = postprocessor.concat_clips(
        concat_list=state.context.concat_list,
        output_file=layout.video_only,
        video_only=True,
    )
    state.video_only_variants = {"base": state.video_only_path}

    optional_variants = (
        ("facefix", layout.scene_final_facefix_video, layout.video_only_facefix),
        ("upscaled", layout.scene_upscaled_video, layout.video_only_upscaled),
    )
    for variant, scene_path, output_path in optional_variants:
        variant_clips = [scene_path(scene_number) for scene_number in scene_numbers]
        available = [clip for clip in variant_clips if clip.is_file()]
        if not available:
            continue
        if len(available) != len(variant_clips):
            missing = [clip.parent.name for clip in variant_clips if not clip.is_file()]
            console.print(
                f"[yellow]Skipping {variant} movie: found {len(available)}/{len(variant_clips)} "
                f"scene clips; missing {', '.join(missing[:10])}.[/yellow]"
            )
            continue
        concat_list = write_concat_list(
            variant_clips,
            layout.final_dir,
            f"concat_{variant}.txt",
        )
        console.print(f"Concatenating {variant} variant: {len(variant_clips)} scene clips")
        state.video_only_variants[variant] = postprocessor.concat_clips(
            concat_list=concat_list,
            output_file=output_path,
            video_only=True,
        )

    try:
        raw_clips = collect_render_plan_scene_raw_clips(
            state.plan_for_next_step,
            state.context.ltx_dir,
            layout=state.context.artifact_layout,
        )
    except FileNotFoundError:
        raw_clips = clips
    if raw_clips is not clips:
        write_concat_list(raw_clips, state.context.artifact_layout.final_dir, "concat_raw.txt")
    else:
        state.context.concat_raw.write_text(
            state.context.concat_list.read_text(encoding="utf-8"), encoding="utf-8"
        )
    postprocessor.concat_clips(
        concat_list=state.context.concat_raw,
        output_file=layout.video_audio,
        video_only=False,
    )


def _run_mux_original_audio_stage(state: PipelineRunState) -> None:
    layout = getattr(state.context, "artifact_layout", None)
    variants = getattr(state, "video_only_variants", None)
    if layout is not None and not variants:
        variants = {}
        if layout.video_only.is_file():
            variants["base"] = layout.video_only
        render_plan = json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig"))
        scene_numbers = [int(entry["scene"]) for entry in render_plan]
        optional_variants = (
            ("facefix", layout.scene_final_facefix_video, layout.video_only_facefix),
            ("upscaled", layout.scene_upscaled_video, layout.video_only_upscaled),
        )
        for variant, scene_path, aggregate_path in optional_variants:
            if not aggregate_path.is_file():
                continue
            scene_clips = [scene_path(scene_number) for scene_number in scene_numbers]
            if all(clip.is_file() for clip in scene_clips):
                variants[variant] = aggregate_path
            else:
                console.print(
                    f"[yellow]Ignoring stale {aggregate_path.name}: the current render plan "
                    f"does not have a complete {variant} scene set.[/yellow]"
                )
    if variants:
        output_paths = {
            "base": layout.movie,
            "facefix": layout.movie_facefix,
            "upscaled": layout.movie_upscaled,
        }
        postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
        results: dict[str, Path] = {}
        for variant in ("base", "facefix", "upscaled"):
            video_file = variants.get(variant)
            if video_file is None:
                continue
            console.print(f"Muxing original audio into {variant} movie")
            results[variant] = postprocessor.mux_original_audio(
                video_file=video_file,
                audio_file=state.context.input_audio,
                output_file=output_paths[variant],
            )
        state.final_video_path = results.get("base") or next(iter(results.values()))
        return

    video_only_path = state.video_only_path or state.context.final_concat_video
    if state.video_only_path is None and not Path(video_only_path).exists():
        raise FileNotFoundError(f"Video-only concat not found: {video_only_path}")
    postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
    output_file = state.context.final_concat
    if Path(video_only_path).name == "video_only_upscaled.mp4":
        output_file = Path(state.context.final_concat).with_name("movie_upscaled.mp4")
    state.final_video_path = postprocessor.mux_original_audio(
        video_file=video_only_path,
        audio_file=state.context.input_audio,
        output_file=output_file,
    )


def _run_upscale_stage(state: PipelineRunState) -> None:
    from .seedvr2_pipeline import SeedVR2CompositionOptions, run_seedvr2
    from feverslop.adapters.comfyui_seedvr2_backend import ComfyUISeedVR2Backend

    config = ProjectConfig.load(state.context.project_config_path)
    if not config.upscale.enabled and not getattr(state.args, "upscale", False):
        console.print("SeedVR2 upscale disabled in project config.")
        return
    workflow_path = Path(config.upscale.workflow_path)
    if not workflow_path.is_absolute():
        workflow_path = runner_root() / workflow_path
    backend = ComfyUISeedVR2Backend(
        client=state.comfyui_client,
        workflow_path=workflow_path,
    )
    reporter = ConsoleReporter(console)
    run_seedvr2(SeedVR2CompositionOptions(
        project_config_path=state.context.project_config_path,
        render_plan_path=state.plan_for_next_step,
        backend=backend,
        skip_existing=not state.args.no_skip_existing,
        force_enabled=bool(getattr(state.args, "upscale", False)),
        resolution_override=(
            (state.args.upscale_resolution.width, state.args.upscale_resolution.height)
            if getattr(state.args, "upscale_resolution", None) is not None
            else None
        ),
        scene_numbers=parse_scene_list(getattr(state.args, "scenes", None)),
        reporter=reporter,
    ))
    console.print("[green]SeedVR2 upscale artifacts ready.[/green]")


def _run_diagnostic_scene_audio_concat_stage(state: PipelineRunState) -> None:
    postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
    postprocessor.concat_clips(
        concat_list=state.context.concat_list,
        output_file=state.context.final_concat_scene_audio_debug,
        video_only=False,
    )


def _run_timeline_export_stage(state: PipelineRunState) -> None:
    from .config_loader import collect_render_plan_scene_clips

    config = ProjectConfig.load(state.context.project_config_path)
    video = config.to_video_settings()
    legacy_openshot_stage = PipelineStage.OPENSHOT_EXPORT.value in (getattr(state.args, "stages", None) or [])
    export_format = "openshot" if legacy_openshot_stage else getattr(state.args, "timeline_format", "openshot")
    extension = "mlt" if export_format == "mlt" else "osp"
    output_dir_name = "timeline" if export_format == "mlt" else "openshot"
    layout = state.context.artifact_layout
    plan_entries = json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig"))
    has_facefix = any(
        layout.scene_final_facefix_video(int(entry.get("scene") or entry.get("scene_number"))).is_file()
        for entry in plan_entries
    )
    has_upscaled = any(
        layout.scene_upscaled_video(int(entry.get("scene") or entry.get("scene_number"))).is_file()
        for entry in plan_entries
    )
    variants = [("", False, False)]
    if has_facefix:
        variants.append(("_facefix", True, False))
    if has_upscaled:
        variants.append(("_upscaled", False, True))

    def report(completed: int, total: int, label: str) -> None:
        console.print(f"[dim]Timeline export: {completed}/{total} ({label})[/dim]")

    for suffix, prefer_facefix, prefer_upscaled in variants:
        clips = collect_render_plan_scene_clips(
            state.plan_for_next_step,
            state.context.ltx_dir,
            layout=layout,
            prefer_facefix=prefer_facefix,
            prefer_upscaled=prefer_upscaled,
        )
        output_path = state.context.project_output_dir / output_dir_name / f"{state.context.project_file_stem}{suffix}.{extension}"
        console.print(f"Timeline export ({export_format}, {suffix or 'final'}): writing {len(clips)} rendered clips")
        if export_format == "mlt":
            written = export_render_plan_to_mlt(
                render_plan_path=state.plan_for_next_step, clip_paths=clips,
                audio_path=state.context.input_audio, output_path=output_path,
                width=video.width, height=video.height, fps=video.fps,
                project_name=f"{state.context.project_file_stem}{suffix}",
            )
            if not suffix:
                state.timeline_project_path = written
        else:
            written = export_render_plan_to_openshot(
                render_plan_path=state.plan_for_next_step, clip_paths=clips,
                audio_path=state.context.input_audio, output_path=output_path,
                width=video.width, height=video.height, fps=video.fps,
                on_progress=report,
            )
            if not suffix:
                state.openshot_project_path = written
        console.print(f"[green]Timeline project written: {output_path}[/green]")


def _run_facefix_stage(state: PipelineRunState) -> None:
    from feverslop.composition.facefix_pipeline import FaceFixCompositionOptions, run_facefix

    layout = state.context.artifact_layout
    scenes_dir = layout.scenes_dir
    if not scenes_dir.is_dir():
        console.print(
            "[yellow]No scenes directory found at"
            + f" {scenes_dir}, FaceFix skipped.[/yellow]"
        )
        return

    scene_numbers = None
    if state.args.scenes:
        scene_numbers = sorted(parse_scene_list(state.args.scenes))

    options = FaceFixCompositionOptions(
        app_config_path=str(state.app_config_path),
        workflow_path=str(state.facefix_workflow),
        scenes_dir=str(scenes_dir),
        project_dir=str(state.context.project_config_dir),
        scene_numbers=scene_numbers,
        reference_images=layout.actor_sheet_images(),
        skip_existing=not state.args.no_skip_existing,
        ffmpeg_debug=getattr(state.args, "facefix_debug", False),
        use_crop_pipeline=True,
    )
    run_facefix(options, console=console)


def _run_facefix_concat_stage(state: PipelineRunState) -> None:
    console.print("FaceFix final concat uses the shared artifact-variant assembler.")
    _run_concat_video_only_stage(state)
    _run_mux_original_audio_stage(state)


STAGE_RUNNERS = {
    PipelineStage.TESTS: _run_tests_stage,
    PipelineStage.MAIN_PIPELINE: _run_main_pipeline_stage,
    PipelineStage.H3_PROMPTS: _run_h3_prompts_stage,
    PipelineStage.RENDER_PLAN: _run_render_plan_stage,
    PipelineStage.RELAY_COMPACT: _run_relay_compact_stage,
    PipelineStage.ANCHOR_FIX: _run_anchor_fix_stage,
    PipelineStage.SET_RESOLUTION: _run_set_resolution_stage,
    PipelineStage.STORYBOARD_FRAMES: _run_storyboard_frames_stage,
    PipelineStage.STORYBOARD_PAGE: _run_storyboard_page_stage,
    PipelineStage.MSR_REFERENCES: _run_msr_references_stage,
    PipelineStage.MSR_REFERENCE_SHEETS: _run_msr_reference_sheets_stage,
    PipelineStage.MSR_PROMPT_ENRICH: _run_msr_prompt_enrich_stage,
    PipelineStage.INGREDIENTS_SHEETS: _run_ingredients_sheets_stage,
    PipelineStage.LTX_PREPARE_WORKFLOWS: _run_ltx_prepare_workflows_stage,
    PipelineStage.LTX_RENDER_SCENES: _run_ltx_render_scenes_stage,
    PipelineStage.PREPARE_WORKFLOWS: _run_ltx_prepare_workflows_stage,
    PipelineStage.RENDER_SCENES: _run_ltx_render_scenes_stage,
    PipelineStage.UPSCALE: _run_upscale_stage,
    PipelineStage.CONCAT_VIDEO_ONLY: _run_concat_video_only_stage,
    PipelineStage.MUX_ORIGINAL_AUDIO: _run_mux_original_audio_stage,
    PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT: _run_diagnostic_scene_audio_concat_stage,
    PipelineStage.FACEFIX: _run_facefix_stage,
    PipelineStage.FACEFIX_CONCAT: _run_facefix_concat_stage,
    PipelineStage.EXPORT_TIMELINE: _run_timeline_export_stage,
    PipelineStage.OPENSHOT_EXPORT: _run_timeline_export_stage,
}

STAGE_LABELS = {
    PipelineStage.TESTS: "tests",
    PipelineStage.MAIN_PIPELINE: "Main pipeline",
    PipelineStage.H3_PROMPTS: "H3 prompts",
    PipelineStage.RENDER_PLAN: "Render plan",
    PipelineStage.RELAY_COMPACT: "relay compact",
    PipelineStage.ANCHOR_FIX: "anchor fix",
    PipelineStage.SET_RESOLUTION: "Set resolution",
    PipelineStage.STORYBOARD_FRAMES: "Storyboard frames",
    PipelineStage.STORYBOARD_PAGE: "Storyboard page",
    PipelineStage.MSR_REFERENCES: "MSR references",
    PipelineStage.MSR_REFERENCE_SHEETS: "MSR reference sheets",
    PipelineStage.MSR_PROMPT_ENRICH: "MSR prompt enrichment",
    PipelineStage.INGREDIENTS_SHEETS: "Ingredients scene sheets",
    PipelineStage.LTX_PREPARE_WORKFLOWS: "Prepare LTX workflows",
    PipelineStage.LTX_RENDER_SCENES: "Video render",
    PipelineStage.PREPARE_WORKFLOWS: "Prepare workflows",
    PipelineStage.RENDER_SCENES: "Render scenes",
    PipelineStage.UPSCALE: "SeedVR2 upscale",
    PipelineStage.CONCAT_VIDEO_ONLY: "Final concat video-only",
    PipelineStage.MUX_ORIGINAL_AUDIO: "Mux original audio",
    PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT: "Diagnostic scene-audio concat",
    PipelineStage.FACEFIX: "FaceFix postprocessing",
    PipelineStage.FACEFIX_CONCAT: "FaceFix final concat",
    PipelineStage.EXPORT_TIMELINE: "Timeline export",
    PipelineStage.OPENSHOT_EXPORT: "OpenShot project export",
}


def run_unittest_suite() -> None:
    subprocess.run(["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests"], check=True, cwd=runner_root())


def _scene_progress_callback(progress: RenderProgressReporter):
    def update(scene_number: int, completed: int, total: int) -> None:
        progress.update(Path(f"scene_{scene_number:04}.json"), completed, total)

    return update


def write_step(message: str) -> None:
    console.print()
    console.print(f"==> {message}")


def resolve_pipeline_stages(args: argparse.Namespace) -> list[PipelineStage]:
    selected = getattr(args, "stages", None)
    if selected:
        aliases = {
            PipelineStage.PREPARE_WORKFLOWS.value: PipelineStage.LTX_PREPARE_WORKFLOWS,
            PipelineStage.RENDER_SCENES.value: PipelineStage.LTX_RENDER_SCENES,
            PipelineStage.EXPORT_TIMELINE.value: PipelineStage.EXPORT_TIMELINE,
        }
        return [aliases.get(stage, PipelineStage(stage)) for stage in selected]

    # --set-resolution is a special mode: just update config + render plan + re-prepare
    set_res = getattr(args, "set_resolution", None)
    if set_res is not None:
        return [PipelineStage.SET_RESOLUTION]

    stages: list[PipelineStage] = []
    if not args.skip_tests:
        stages.append(PipelineStage.TESTS)
    if not args.skip_main_pipeline:
        stages.append(PipelineStage.MAIN_PIPELINE)
    else:
        console.print("Skipping main pipeline; using existing timeline, prompts, and render plan.")
    if not args.skip_relay_compact and args.render_mode != "single_prompt":
        stages.append(PipelineStage.RELAY_COMPACT)
    if not args.skip_anchor_fix:
        stages.append(PipelineStage.ANCHOR_FIX)
    if args.video_pipeline in ("ltx_msr", "minimax-h3-r2v"):
        if not args.skip_msr_reference_render:
            stages.append(PipelineStage.MSR_REFERENCES)
        else:
            console.print("Skipping MSR reference rendering; using existing reference manifests.")
        stages.append(PipelineStage.MSR_REFERENCE_SHEETS)
        if args.video_pipeline == "minimax-h3-r2v" and not args.skip_main_pipeline:
            stages.append(PipelineStage.H3_PROMPTS)
            stages.append(PipelineStage.RENDER_PLAN)
        if args.video_pipeline == "ltx_msr" and not args.skip_msr_prompt_enrichment:
            stages.append(PipelineStage.MSR_PROMPT_ENRICH)
        elif args.video_pipeline == "ltx_msr":
            console.print("Skipping MSR prompt enrichment; using existing MSR prompt fields.")
    elif args.video_pipeline == "ltx_ingredients":
        if not args.skip_msr_reference_render:
            stages.append(PipelineStage.MSR_REFERENCES)
        else:
            console.print("Skipping MSR reference rendering; using existing reference manifests.")
        stages.append(PipelineStage.MSR_REFERENCE_SHEETS)
        if not args.skip_msr_prompt_enrichment:
            stages.append(PipelineStage.MSR_PROMPT_ENRICH)
        else:
            console.print("Skipping MSR prompt enrichment; using existing MSR prompt fields.")
        if not getattr(args, "skip_ingredients_sheets", False):
            stages.append(PipelineStage.INGREDIENTS_SHEETS)
        else:
            console.print("Skipping Ingredients sheets; using existing sheets or references.")
    else:
        if not args.skip_storyboard:
            stages.append(PipelineStage.STORYBOARD_FRAMES)
        if not args.skip_storyboard_page:
            stages.append(PipelineStage.STORYBOARD_PAGE)
    if not args.skip_ltx:
        if args.video_pipeline in ("ltx_msr", "ltx_ingredients"):
            stages.append(PipelineStage.LTX_PREPARE_WORKFLOWS)
        stages.append(PipelineStage.LTX_RENDER_SCENES)
    if not args.skip_final_concat:
        config_path = getattr(args, "project_config", None)
        if config_path is None and getattr(args, "project_root", None):
            config_path = Path(args.project_root) / "config.json"
        config_upscale_enabled = False
        if config_path and Path(config_path).is_file():
            config_upscale_enabled = ProjectConfig.load(config_path).upscale.enabled
        upscale_enabled = bool(getattr(args, "upscale", False)) or config_upscale_enabled
        if not args.skip_facefix:
            stages.append(PipelineStage.FACEFIX)
        else:
            console.print("Skipping FaceFix postprocessing.")
        if upscale_enabled:
            stages.append(PipelineStage.UPSCALE)
        stages.append(PipelineStage.CONCAT_VIDEO_ONLY)
        stages.append(PipelineStage.MUX_ORIGINAL_AUDIO)
        if args.diagnostic_original_audio_mux:
            stages.append(PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT)
        elif args.no_original_audio_mux:
            console.print("--no-original-audio-mux is deprecated; original-audio muxing is now always used for final concat.")
        if not getattr(args, "skip_openshot_export", False):
            stages.append(PipelineStage.EXPORT_TIMELINE)
        else:
            console.print("Skipping OpenShot project export.")
    elif not args.skip_facefix:
        stages.append(PipelineStage.FACEFIX)
    else:
        console.print("Skipping FaceFix postprocessing.")
    return stages


def _initial_render_plan(context: PipelineRunContext, args: argparse.Namespace, stages: list[PipelineStage]) -> Path:
    upstream_stages = {PipelineStage.MAIN_PIPELINE, PipelineStage.RELAY_COMPACT, PipelineStage.ANCHOR_FIX, PipelineStage.MSR_REFERENCE_SHEETS, PipelineStage.INGREDIENTS_SHEETS}
    render_dir = context.render_dir
    legacy_base = render_dir / f"render_plan_{context.song_id}.json"
    legacy_references = render_dir / f"render_plan_{context.song_id}_refs.json"
    legacy_ingredients = render_dir / f"render_plan_{context.song_id}_ingredients.json"
    if stages in ([PipelineStage.OPENSHOT_EXPORT], [PipelineStage.EXPORT_TIMELINE]):
        for plan_path, legacy_paths in (
            (context.render_plan, [legacy_base]),
            (context.reference_plan, [legacy_references]),
            (context.ingredients_plan, [legacy_ingredients]),
            (context.anchored_plan, []),
            (context.compact_plan, []),
        ):
            existing = context.artifact_layout.find_plan(plan_path, legacy_paths=legacy_paths)
            if existing:
                return existing
        return context.render_plan
    if args.video_pipeline == "ltx_msr" and not upstream_stages.intersection(stages):
        existing = context.artifact_layout.find_plan(context.reference_plan, legacy_paths=[legacy_references])
        if existing:
            return existing
    if args.video_pipeline == "ltx_ingredients" and not upstream_stages.intersection(stages):
        existing = context.artifact_layout.find_plan(context.ingredients_plan, legacy_paths=[legacy_ingredients])
        if existing:
            return existing
    # MiniMax R2V needs reference paths (actor_msr_paths, location_msr_path) from
    # an existing MSR/ingredients plan so it can patch them into its workflow.
    if args.video_pipeline == "minimax-h3-r2v":
        if context.render_plan.is_file():
            scenes = JsonArtifactStore().read_json(context.render_plan)
            if scenes and all((scene.get("h3") or {}).get("prompt") for scene in scenes):
                return context.render_plan
        for plan_path, legacy_paths in (
            (context.ingredients_plan, [legacy_ingredients]),
            (context.reference_plan, [legacy_references]),
            (context.render_plan, [legacy_base]),
        ):
            existing = context.artifact_layout.find_plan(plan_path, legacy_paths=legacy_paths)
            if existing:
                return existing
        return context.render_plan
    return context.artifact_layout.find_plan(context.render_plan, legacy_paths=[legacy_base]) or context.render_plan
