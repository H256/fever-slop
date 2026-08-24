from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from feverslop.adapters.comfyui_seedvr2_backend import (
    ComfyUISeedVR2Backend,
    SeedVR2RenderSettings,
)
from feverslop.adapters.reporting import NullReporter
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.seedvr2 import (
    SeedVR2Pass,
    SeedVR2Segment,
    plan_seedvr2_passes,
    plan_seedvr2_segments,
)
from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.utils.sub_step_progress import SubStepProgress


class SeedVR2Backend(Protocol):
    def render(self, **kwargs) -> Path: ...


@dataclass(frozen=True)
class SeedVR2CompositionOptions:
    project_config_path: str | Path
    render_plan_path: str | Path
    backend: SeedVR2Backend | None = None
    probe_size: Callable[[Path], tuple[int, int]] | None = None
    probe_duration: Callable[[Path], float | None] | None = None
    skip_existing: bool = True
    force_enabled: bool = False
    resolution_override: tuple[int, int] | None = None
    scene_numbers: set[int] | None = None
    reporter: Any = field(default_factory=NullReporter)


def _probe_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = (json.loads(result.stdout).get("streams") or [])[0]
    return int(stream["width"]), int(stream["height"])


def _probe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        value = float(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return value if value >= 0 else None


def _resolve_vae_temporal_size(configured: int, duration_seconds: float | None) -> int:
    if duration_seconds is None:
        return configured
    if duration_seconds > 10.0:
        return min(configured, 16)
    if duration_seconds > 6.0:
        return min(configured, 32)
    return configured


def _source_clip(layout: SceneArtifactLayout, scene_number: int) -> Path:
    candidates = [layout.scene_final_facefix_video(scene_number), layout.scene_final_video(scene_number)]
    legacy_dirs = []
    for root in (layout.render_dir, layout.output_dir / "movie"):
        if not root.is_dir():
            continue
        legacy_dirs.append(root)
        legacy_dirs.extend(
            directory
            for directory in sorted(root.iterdir())
            if directory.is_dir() and directory.name != layout.scenes_dir.name
        )
    for directory in legacy_dirs:
        candidates.extend((
            directory / "facefix" / f"scene_{scene_number:04d}.mp4",
            directory / "facefix" / "final" / f"scene_{scene_number:04d}.mp4",
        ))
    for directory in legacy_dirs:
        candidates.extend((
            directory / f"scene_{scene_number:04d}.mp4",
            directory / "final" / f"scene_{scene_number:04d}.mp4",
        ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No final or FaceFix clip found for scene {scene_number}; searched: {searched}")


def _pass_path(layout: SceneArtifactLayout, scene_number: int, pass_number: int, final: bool) -> Path:
    if final:
        return layout.scene_upscaled_video(scene_number)
    return layout.scene_dir(scene_number) / f"upscale_pass_{pass_number:02d}.mp4"


def _pass_record(item: SeedVR2Pass, output: Path, pass_number: int) -> dict:
    return {
        "pass": pass_number,
        "input_width": item.input_size[0],
        "input_height": item.input_size[1],
        "output_width": item.output_size[0],
        "output_height": item.output_size[1],
        "scale": item.scale,
        "output": str(output),
    }


def _pass_segment_cap(upscale, pass_spec: SeedVR2Pass, final_size: tuple[int, int]) -> float:
    final_area = final_size[0] * final_size[1]
    pass_area = pass_spec.output_size[0] * pass_spec.output_size[1]
    return max(1.0, upscale.segment_duration_seconds * final_area / pass_area)


def _usable_artifact(
    path: Path,
    probe_duration: Callable[[Path], float | None] | None = None,
    *,
    require_probe: bool = False,
) -> bool:
    try:
        if not path.is_file():
            return False
        if path.stat().st_size <= 0:
            return False
        if probe_duration is None:
            return True
        duration = probe_duration(path)
        return duration is not None and duration > 0.0 or duration is None and not require_probe
    except OSError:
        return False


def _render_segmented_pass(
    *,
    backend: SeedVR2Backend,
    postprocessor: VideoPostProcessor,
    segments: list[SeedVR2Segment],
    scene_dir: Path,
    pass_number: int,
    scene_number: int,
    pass_spec: SeedVR2Pass,
    settings: SeedVR2RenderSettings,
    source: Path,
    output: Path,
    reporter: Any,
    skip_existing: bool,
    probe_duration: Callable[[Path], float | None],
) -> Path:
    segment_outputs: list[Path] = []
    for segment in segments:
        segment_output = scene_dir / f"upscale_pass_{pass_number:02d}_segment_{segment.index:04d}.mp4"
        segment_outputs.append(segment_output)
        if skip_existing and _usable_artifact(segment_output, probe_duration, require_probe=True):
            reporter.message(
                f"[yellow]SeedVR2 scene {scene_number} pass {pass_number} segment {segment.index}/{len(segments)} skipped: existing {segment_output}[/yellow]",
            )
            continue
        reporter.message(
            f"[cyan]SeedVR2 scene {scene_number} pass {pass_number} segment {segment.index}/{len(segments)} starting: "
            f"{segment.start_seconds:.2f}s + {segment.duration_seconds:.2f}s[/cyan]",
        )
        segment_settings = replace(
            settings,
            trim_start_seconds=segment.start_seconds,
            trim_duration_seconds=segment.duration_seconds,
        )
        backend.render(
            source_video=source,
            output_path=segment_output,
            output_size=pass_spec.output_size,
            scene_number=scene_number,
            pass_number=pass_number,
            segment_number=segment.index,
            settings=segment_settings,
        )
        reporter.message(
            f"[green]SeedVR2 scene {scene_number} pass {pass_number} segment {segment.index}/{len(segments)} complete: {segment_output}[/green]",
        )
    concat_list = scene_dir / f"upscale_pass_{pass_number:02d}_segments.txt"
    postprocessor.write_concat_list(segment_outputs, concat_list)
    frame_count = max(1, round(sum(segment.duration_seconds for segment in segments) * settings.fps))
    if hasattr(postprocessor, "ffmpeg_timeout_seconds"):
        postprocessor.ffmpeg_timeout_seconds = max(
            120.0,
            sum(segment.duration_seconds for segment in segments) * 20.0,
        )
    return postprocessor.concat_clips(
        concat_list,
        output,
        video_only=True,
        reencode=True,
        fps=settings.fps,
        frame_count=frame_count,
    )


def run_seedvr2(options: SeedVR2CompositionOptions) -> list[Path]:
    config = ProjectConfig.load(options.project_config_path)
    if options.force_enabled and not config.upscale.enabled:
        config = replace(config, upscale=replace(config.upscale, enabled=True))
    if options.resolution_override is not None:
        width, height = options.resolution_override
        config = replace(
            config,
            upscale=replace(config.upscale, target_width=width, target_height=height),
        )
    if not config.upscale.enabled:
        return []
    layout = SceneArtifactLayout(config.project_dir)
    backend = options.backend or ComfyUISeedVR2Backend(
        client=_build_comfy_client(config),
        workflow_path=(
            Path(config.upscale.workflow_path)
            if Path(config.upscale.workflow_path).is_absolute()
            else Path(__file__).resolve().parents[3] / config.upscale.workflow_path
        ),
    )
    probe_size = options.probe_size or _probe_size
    probe_duration = options.probe_duration or _probe_duration
    postprocessor = VideoPostProcessor(
        ffmpeg_path="ffmpeg",
        audio_bitrate="320k",
    )
    plan = json.loads(Path(options.render_plan_path).read_text(encoding="utf-8-sig"))
    if options.scene_numbers is not None:
        plan = [
            entry for entry in plan
            if int(entry.get("scene") or entry.get("scene_number")) in options.scene_numbers
        ]
    outputs: list[Path] = []
    progress = SubStepProgress(options.reporter, "SeedVR2 scenes", len(plan), interval=1, verbose=True)
    progress.update(0, detail="starting", force=True)
    for scene_index, entry in enumerate(plan, start=1):
        scene_number = int(entry.get("scene") or entry.get("scene_number"))
        final_output = layout.scene_upscaled_video(scene_number)
        require_probe = probe_duration is _probe_duration
        if options.skip_existing and _usable_artifact(final_output, probe_duration):
            segment_lists = sorted(layout.scene_dir(scene_number).glob("upscale_pass_*_segments.txt"))
            if segment_lists:
                segment_list = segment_lists[-1]
                segment_outputs = [
                    Path(line.strip()[6:-1])
                    for line in segment_list.read_text(encoding="utf-8").splitlines()
                    if line.strip().startswith("file '") and line.strip().endswith("'")
                ]
                if segment_outputs and all(
                    _usable_artifact(path, probe_duration, require_probe=require_probe)
                    for path in segment_outputs
                ):
                    source = _source_clip(layout, scene_number)
                    source_duration = probe_duration(source)
                    if source_duration is not None:
                        options.reporter.message(
                            f"[cyan]SeedVR2 scene {scene_index}/{len(plan)} rebuilding existing final "
                            f"from {len(segment_outputs)} segments: {final_output}[/cyan]",
                        )
                        probed_durations = [probe_duration(path) for path in segment_outputs]
                        segment_duration = min(
                            source_duration,
                            sum(duration for duration in probed_durations if duration is not None),
                        ) if all(duration is not None for duration in probed_durations) else source_duration
                        if hasattr(postprocessor, "ffmpeg_timeout_seconds"):
                            postprocessor.ffmpeg_timeout_seconds = max(
                                120.0,
                                (segment_duration or source_duration) * 20.0,
                            )
                        postprocessor.concat_clips(
                            segment_list,
                            final_output,
                            video_only=True,
                            reencode=True,
                            fps=config.video.fps,
                            frame_count=max(1, round((segment_duration or source_duration) * config.video.fps)),
                        )
                        outputs.append(final_output)
                        progress.update(scene_index, detail=f"scene {scene_number} rebuilt", force=True)
                        continue
            options.reporter.message(
                f"[yellow]SeedVR2 scene {scene_index}/{len(plan)} skipped: existing {final_output}[/yellow]",
            )
            outputs.append(final_output)
            progress.update(scene_index, detail=f"scene {scene_number} skipped", force=True)
            continue
        source = _source_clip(layout, scene_number)
        source_duration = probe_duration(source)
        duration_suffix = f" ({source_duration:.2f}s)" if source_duration is not None else ""
        options.reporter.message(
            f"[cyan]SeedVR2 scene {scene_index}/{len(plan)} starting: source {source}{duration_suffix}[/cyan]",
        )
        source_size = probe_size(source)
        upscale = config.upscale
        max_pass_scale = upscale.max_pass_scale
        max_ai_passes = upscale.max_ai_passes
        if upscale.strategy == "single":
            max_pass_scale = max(
                2.0,
                (upscale.target_width or source_size[0]) / source_size[0],
                (upscale.target_height or source_size[1]) / source_size[1],
            )
            max_ai_passes = 1
        passes = plan_seedvr2_passes(
            source_size[0],
            source_size[1],
            target_width=upscale.target_width,
            target_height=upscale.target_height,
            default_scale=upscale.default_scale,
            max_pass_scale=max_pass_scale,
            max_ai_passes=max_ai_passes,
        )
        if not passes:
            final_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, final_output)
            options.reporter.message(
                f"[green]SeedVR2 scene {scene_index}/{len(plan)} complete: copied source (target already matched)[/green]",
            )
            outputs.append(final_output)
            progress.update(scene_index, detail=f"scene {scene_number} complete", force=True)
            continue
        scene_dir = layout.scene_dir(scene_number)
        current = source
        records: list[dict] = []
        vae_temporal_size = upscale.vae_temporal_size
        vae_temporal_overlap = min(upscale.vae_temporal_overlap, vae_temporal_size)
        settings = SeedVR2RenderSettings(
            model=upscale.model,
            vae=upscale.vae,
            denoise=upscale.denoise,
            temporal_overlap=upscale.temporal_overlap,
            split_latent=upscale.split_latent,
            vae_temporal_size=vae_temporal_size,
            vae_temporal_overlap=vae_temporal_overlap,
            color_correction=upscale.color_correction,
            seed=upscale.seed,
            fps=config.video.fps,
        )
        for pass_number, pass_spec in enumerate(passes, start=1):
            output = _pass_path(layout, scene_number, pass_number, pass_number == len(passes))
            if options.skip_existing and _usable_artifact(output, probe_duration, require_probe=require_probe):
                current = output
                records.append(_pass_record(pass_spec, output, pass_number))
                options.reporter.message(
                    f"[yellow]SeedVR2 scene {scene_index}/{len(plan)} pass {pass_number}/{len(passes)} skipped: existing {output}[/yellow]",
                )
                continue
            options.reporter.message(
                f"[cyan]SeedVR2 scene {scene_index}/{len(plan)} pass {pass_number}/{len(passes)} starting: "
                f"{pass_spec.input_size[0]}x{pass_spec.input_size[1]} -> {pass_spec.output_size[0]}x{pass_spec.output_size[1]} "
                f"vae_temporal_size={settings.vae_temporal_size}[/cyan]",
            )
            try:
                current_duration = probe_duration(current) or source_duration or 0.0
                pass_settings = replace(settings, scale_multiplier=pass_spec.scale)
                if current_duration > 0:
                    segment_cap = _pass_segment_cap(upscale, pass_spec, passes[-1].output_size)
                    segments = plan_seedvr2_segments(current_duration, max_segment_duration=segment_cap)
                else:
                    segments = []
                if current_duration > 0 and len(segments) > 1:
                    rendered = _render_segmented_pass(
                        backend=backend,
                        postprocessor=postprocessor,
                        segments=segments,
                        scene_dir=scene_dir,
                        pass_number=pass_number,
                        scene_number=scene_number,
                        pass_spec=pass_spec,
                        settings=pass_settings,
                        source=current,
                        output=output,
                        reporter=options.reporter,
                        skip_existing=options.skip_existing,
                        probe_duration=probe_duration,
                    )
                else:
                    rendered = backend.render(
                        source_video=current,
                        output_path=output,
                        output_size=pass_spec.output_size,
                        scene_number=scene_number,
                        pass_number=pass_number,
                        settings=pass_settings,
                    )
            except Exception as exc:
                options.reporter.message(
                    f"[red]SeedVR2 scene {scene_index}/{len(plan)} pass {pass_number}/{len(passes)} failed: {exc}[/red]",
                )
                raise
            current = Path(rendered)
            records.append(_pass_record(pass_spec, current, pass_number))
            options.reporter.message(
                f"[green]SeedVR2 scene {scene_index}/{len(plan)} pass {pass_number}/{len(passes)} complete: {current}[/green]",
            )
        manifest = {
            "scene": scene_number,
            "source": str(source),
            "source_duration_seconds": source_duration,
            "strategy": upscale.strategy,
            "profile": "identity",
            "settings": {
                "model": settings.model,
                "vae": settings.vae,
                "denoise": settings.denoise,
                "temporal_overlap": settings.temporal_overlap,
                "split_latent": settings.split_latent,
                "vae_temporal_size": settings.vae_temporal_size,
                "vae_temporal_overlap": settings.vae_temporal_overlap,
                "color_correction": settings.color_correction,
                "seed": settings.seed,
            },
            "passes": records,
        }
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "upscale_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        outputs.append(final_output)
        options.reporter.message(
            f"[green]SeedVR2 scene {scene_index}/{len(plan)} complete: {final_output}[/green]",
        )
        progress.update(scene_index, detail=f"scene {scene_number} complete", force=True)
    return outputs


def _build_comfy_client(config: ProjectConfig):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load(config.project_dir / "app_config.json") if (config.project_dir / "app_config.json").is_file() else AppConfig.load("app_config.json")
    return ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
