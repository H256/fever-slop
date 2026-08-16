from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import shutil
from typing import Callable, Protocol
from dataclasses import replace

from feverslop.adapters.comfyui_seedvr2_backend import (
    ComfyUISeedVR2Backend,
    SeedVR2RenderSettings,
)
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.seedvr2 import SeedVR2Pass, plan_seedvr2_passes
from feverslop.scene_artifacts import SceneArtifactLayout


class SeedVR2Backend(Protocol):
    def render(self, **kwargs) -> Path: ...


@dataclass(frozen=True)
class SeedVR2CompositionOptions:
    project_config_path: str | Path
    render_plan_path: str | Path
    backend: SeedVR2Backend | None = None
    probe_size: Callable[[Path], tuple[int, int]] | None = None
    skip_existing: bool = True
    force_enabled: bool = False
    resolution_override: tuple[int, int] | None = None


def _probe_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = (json.loads(result.stdout).get("streams") or [])[0]
    return int(stream["width"]), int(stream["height"])


def _source_clip(layout: SceneArtifactLayout, scene_number: int) -> Path:
    for candidate in (layout.scene_final_facefix_video(scene_number), layout.scene_final_video(scene_number)):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No final or FaceFix clip found for scene {scene_number}: {layout.scene_dir(scene_number)}")


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
        workflow_path=Path(config.project_dir, config.upscale.workflow_path),
    )
    probe_size = options.probe_size or _probe_size
    plan = json.loads(Path(options.render_plan_path).read_text(encoding="utf-8-sig"))
    outputs: list[Path] = []
    for entry in plan:
        scene_number = int(entry.get("scene") or entry.get("scene_number"))
        final_output = layout.scene_upscaled_video(scene_number)
        if options.skip_existing and final_output.is_file():
            outputs.append(final_output)
            continue
        source = _source_clip(layout, scene_number)
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
            outputs.append(final_output)
            continue
        scene_dir = layout.scene_dir(scene_number)
        current = source
        records: list[dict] = []
        settings = SeedVR2RenderSettings(
            model=upscale.model,
            vae=upscale.vae,
            denoise=upscale.denoise,
            temporal_overlap=upscale.temporal_overlap,
            color_correction=upscale.color_correction,
            seed=upscale.seed,
            fps=config.video.fps,
        )
        for pass_number, pass_spec in enumerate(passes, start=1):
            output = _pass_path(layout, scene_number, pass_number, pass_number == len(passes))
            if options.skip_existing and output.is_file():
                current = output
                records.append(_pass_record(pass_spec, output, pass_number))
                continue
            rendered = backend.render(
                source_video=current,
                output_path=output,
                output_size=pass_spec.output_size,
                scene_number=scene_number,
                pass_number=pass_number,
                settings=settings,
            )
            current = Path(rendered)
            records.append(_pass_record(pass_spec, current, pass_number))
        manifest = {
            "scene": scene_number,
            "source": str(source),
            "strategy": upscale.strategy,
            "profile": "identity",
            "settings": {
                "model": settings.model,
                "vae": settings.vae,
                "denoise": settings.denoise,
                "temporal_overlap": settings.temporal_overlap,
                "color_correction": settings.color_correction,
                "seed": settings.seed,
            },
            "passes": records,
        }
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "upscale_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        outputs.append(final_output)
    return outputs


def _build_comfy_client(config: ProjectConfig):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load(config.project_dir / "app_config.json") if (config.project_dir / "app_config.json").is_file() else AppConfig.load("app_config.json")
    return ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
