"""Final-assembly stage runners (cutless assembly, concat, mux, upscale,
diagnostic, timeline export, facefix) for the M-20 split of
``stage_runners`` (issue #1194).

Moved mechanically from ``feverslop.composition.stage_runners`` (P5).
``stage_runners`` re-exports these so existing imports and
``patch("...stage_runners.X")`` targets keep working.
"""
from __future__ import annotations

import json
from pathlib import Path

from feverslop.adapters.cutless_assembly import CutlessAssemblyService
from feverslop.adapters.reporting import ConsoleReporter
from feverslop.adapters.video_postprocessor import (
    VideoPostProcessor,
    final_video_postprocessor,
)
from feverslop.application.mlt_exporter import export_render_plan_to_mlt
from feverslop.application.openshot_exporter import export_render_plan_to_openshot
from feverslop.composition.cutless_assembly import assemble_declared_cutless_group
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.render_plan import RenderPlan
from feverslop.tools.storyboard_page import parse_scene_list

from ..arg_parser import PipelineStage
from ..config_loader import PipelineRunState, runner_root
from .progress import _report, console


def _assemble_declared_cutless_groups(
    render_plan: list[dict],
    clips: list[Path],
    *,
    output_dir: Path,
    postprocessor: VideoPostProcessor,
) -> list[Path]:
    """Replace explicitly declared continuation groups with hard-cut movies."""
    clips_by_segment = {
        str((entry.get("metadata") or {}).get("segment_id") or entry.get("segment_id") or ""): clip
        for entry, clip in zip(render_plan, clips)
    }
    groups: list[dict] = []
    seen: set[str] = set()
    for entry in render_plan:
        for group in (entry.get("metadata") or {}).get("continuation_groups") or []:
            group_id = str(group.get("group_id") or "").strip()
            if group_id and group_id not in seen:
                groups.append(group)
                seen.add(group_id)
    if not groups:
        return clips

    assembled = list(clips)
    for index, group in enumerate(groups, start=1):
        segment_ids = [
            str(segment.get("segment_id") or "").strip()
            for segment in group.get("segments") or []
        ]
        if not segment_ids or any(segment_id not in clips_by_segment for segment_id in segment_ids):
            _report(
                f"[yellow]Skipping cutless group {group.get('group_id', index)}: "
                "rendered segment clips are not individually addressable.[/yellow]",
            )
            continue
        group_clips = {segment_id: clips_by_segment[segment_id] for segment_id in segment_ids}
        output_file = output_dir / f"cutless_{index:04d}.mp4"
        diagnostics_file = output_dir / f"cutless_{index:04d}.json"
        assemble_declared_cutless_group(
            group=group,
            clips_by_segment=group_clips,
            frame_count=postprocessor._frame_count,
            extract_first_frame=postprocessor.extract_first_frame,
            extract_last_frame=postprocessor.extract_last_frame,
            assembly_service=CutlessAssemblyService(postprocessor),
            output_file=output_file,
            diagnostics_file=diagnostics_file,
            fps=int(render_plan[0].get("fps") or 24),
            duplicate_policy=str(group.get("duplicate_policy") or "reject"),
            reporter=ConsoleReporter(console),
        )
        first_clip_index = min(assembled.index(clips_by_segment[segment_id]) for segment_id in segment_ids)
        assembled = [
            clip for clip in assembled
            if clip not in group_clips.values() or clip == assembled[first_clip_index]
        ]
        assembled[first_clip_index] = output_file
    return assembled


def _stage_final_postprocessor(state: PipelineRunState) -> VideoPostProcessor:
    """Build the final-assembly postprocessor with the project FFmpeg timeout."""
    return final_video_postprocessor(
        AppConfig.load(state.app_config_path).comfyui.ffmpeg_timeout_seconds
    )


def _run_concat_video_only_stage(state: PipelineRunState) -> None:
    from feverslop.domain.scene_recovery import require_ready_scenes
    from ..config_loader import (
        collect_render_plan_scene_clips,
        collect_render_plan_scene_raw_clips,
        rewrite_concat_list,
        write_concat_list,
    )
    layout = state.context.artifact_layout
    render_plan = json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig"))
    args = getattr(state, "args", None)
    selected_scenes = parse_scene_list(getattr(args, "scenes", None))
    if selected_scenes is None and getattr(args, "smoke_only", False):
        selected_scenes = {int(args.smoke_scene)}
    render_plan = RenderPlan.from_dicts(render_plan).select(
        scene_numbers=selected_scenes,
    ).to_dicts()
    require_ready_scenes(
        render_plan,
        project_path=getattr(getattr(state, "context", None), "project_config_dir", None),
    )
    scene_numbers = [int(entry["scene"]) for entry in render_plan]
    canonical_clips = [layout.scene_final_video(scene_number) for scene_number in scene_numbers]
    canonical_available = [clip for clip in canonical_clips if clip.is_file()]
    if canonical_available and len(canonical_available) != len(canonical_clips):
        missing = [clip.parent.name for clip in canonical_clips if not clip.is_file()]
        raise FileNotFoundError(
            "Cannot build base variant without mixing artifact layouts; missing canonical "
            f"final.mp4 for {', '.join(missing[:10])}",
        )
    if canonical_available:
        clips = canonical_clips
    else:
        _report(
            "[yellow]No canonical final.mp4 scene artifacts found; using the complete legacy "
            "scene layout for the base movie.[/yellow]",
        )
        clips = collect_render_plan_scene_clips(
            state.plan_for_next_step,
            state.context.ltx_dir,
            layout=None,
        )
    clips = _assemble_declared_cutless_groups(
        render_plan,
        clips,
        output_dir=layout.final_dir,
        postprocessor=_stage_final_postprocessor(state),
    )
    rewrite_concat_list(clips, state.context.artifact_layout.final_dir)
    postprocessor = _stage_final_postprocessor(state)
    _report(f"Concatenating base variant: {len(clips)} scene clips")
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
            _report(
                f"[yellow]Skipping {variant} movie: found {len(available)}/{len(variant_clips)} "
                f"scene clips; missing {', '.join(missing[:10])}.[/yellow]",
            )
            continue
        concat_list = write_concat_list(
            variant_clips,
            layout.final_dir,
            f"concat_{variant}.txt",
        )
        _report(f"Concatenating {variant} variant: {len(variant_clips)} scene clips")
        state.video_only_variants[variant] = postprocessor.concat_clips(
            concat_list=concat_list,
            output_file=output_path,
            video_only=True,
            reencode=variant == "upscaled",
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
            state.context.concat_list.read_text(encoding="utf-8"), encoding="utf-8",
        )
    postprocessor.concat_clips(
        concat_list=state.context.concat_raw,
        output_file=layout.video_audio,
        video_only=False,
    )


def _run_mux_original_audio_stage(state: PipelineRunState) -> None:
    from feverslop.domain.scene_recovery import require_ready_scenes
    if getattr(state, "plan_for_next_step", None):
        require_ready_scenes(
            json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig")),
            project_path=getattr(getattr(state, "context", None), "project_config_dir", None),
        )
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
                _report(
                    f"[yellow]Ignoring stale {aggregate_path.name}: the current render plan "
                    f"does not have a complete {variant} scene set.[/yellow]",
                )
    if variants:
        output_paths = {
            "base": layout.movie,
            "facefix": layout.movie_facefix,
            "upscaled": layout.movie_upscaled,
        }
        postprocessor = _stage_final_postprocessor(state)
        results: dict[str, Path] = {}
        for variant in ("base", "facefix", "upscaled"):
            video_file = variants.get(variant)
            if video_file is None:
                continue
            _report(f"Muxing original audio into {variant} movie")
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
    postprocessor = _stage_final_postprocessor(state)
    output_file = state.context.final_concat
    if Path(video_only_path).name == "video_only_upscaled.mp4":
        output_file = Path(state.context.final_concat).with_name("movie_upscaled.mp4")
    state.final_video_path = postprocessor.mux_original_audio(
        video_file=video_only_path,
        audio_file=state.context.input_audio,
        output_file=output_file,
    )


def _run_upscale_stage(state: PipelineRunState) -> None:
    from feverslop.adapters.comfyui_seedvr2_backend import ComfyUISeedVR2Backend

    from ..seedvr2_pipeline import SeedVR2CompositionOptions, run_seedvr2

    config = ProjectConfig.load(state.context.project_config_path)
    if not config.upscale.enabled and not getattr(state.args, "upscale", False):
        _report("SeedVR2 upscale disabled in project config.")
        return
    config.upscale.validate_resources()
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
    _report("[green]SeedVR2 upscale artifacts ready.[/green]")


def _run_diagnostic_scene_audio_concat_stage(state: PipelineRunState) -> None:
    postprocessor = _stage_final_postprocessor(state)
    postprocessor.concat_clips(
        concat_list=state.context.concat_list,
        output_file=state.context.final_concat_scene_audio_debug,
        video_only=False,
    )


def _run_timeline_export_stage(state: PipelineRunState) -> None:
    from ..config_loader import collect_render_plan_scene_clips

    config = ProjectConfig.load(state.context.project_config_path)
    video = config.to_video_settings()
    legacy_openshot_stage = PipelineStage.OPENSHOT_EXPORT.value in (getattr(state.args, "stages", None) or [])
    requested_format = "openshot" if legacy_openshot_stage else getattr(state.args, "timeline_format", "both")
    export_formats = [requested_format] if requested_format != "both" else ["mlt", "openshot"]
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
        _report(f"[dim]Timeline export: {completed}/{total} ({label})[/dim]")

    for export_format in export_formats:
        extension = "mlt" if export_format == "mlt" else "osp"
        output_dir_name = "timeline" if export_format == "mlt" else "openshot"
        for suffix, prefer_facefix, prefer_upscaled in variants:
            clips = collect_render_plan_scene_clips(
                state.plan_for_next_step,
                state.context.ltx_dir,
                layout=layout,
                prefer_facefix=prefer_facefix,
                prefer_upscaled=prefer_upscaled,
            )
            output_path = state.context.project_output_dir / output_dir_name / f"{state.context.project_file_stem}{suffix}.{extension}"
            _report(f"Timeline export ({export_format}, {suffix or 'final'}): writing {len(clips)} rendered clips")
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
            _report(f"[green]Timeline project written: {output_path}[/green]")


def _run_facefix_stage(state: PipelineRunState) -> None:
    from feverslop.composition.facefix_pipeline import (
        FaceFixCompositionOptions,
        run_facefix,
    )

    layout = state.context.artifact_layout
    scenes_dir = layout.scenes_dir
    if not scenes_dir.is_dir():
        _report(
            "[yellow]No scenes directory found at"
             f" {scenes_dir}, FaceFix skipped.[/yellow]",
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
    _report("FaceFix final concat uses the shared artifact-variant assembler.")
    _run_concat_video_only_stage(state)
    _run_mux_original_audio_stage(state)

