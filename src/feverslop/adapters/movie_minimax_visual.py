from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.application.render_video import RenderVideoScenesRequest
from feverslop.adapters.movie_visual import _references_from_ids
from feverslop.application.movie_msr_enrichment import _movie_video_prompt
from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
from feverslop.config.project_config import ProjectConfig
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.dspy_h3_prompt_builder import _format_relay_shots, _normalize_relay_segments


class ComfyUIMiniMaxMovieVisualAdapter:
    """Render and concatenate Movie scenes through a MiniMax H3 pipeline."""

    def __init__(self, *, project_dir: Path, workflow_path: str | Path, video_pipeline: str):
        self.project_dir = Path(project_dir)
        self.workflow_path = Path(workflow_path)
        self.video_pipeline = video_pipeline
        self.output_dir = self.project_dir / "output" / "movie" / video_pipeline
        self.postprocessor = VideoPostProcessor()

    def render_movie(
        self,
        *,
        project_dir: Path,
        render_plan_path: Path,
        selected_scenes: list[int] | None = None,
        concat_only: bool = False,
        continuity_keyframes: str = "none",
        on_clip_rendered: Callable[[int, int, int], None] | None = None,
    ) -> Path:
        del continuity_keyframes
        project_config = ProjectConfig.load(Path(project_dir) / "config.json")
        minimax_plan_path = self._prepare_render_plan(render_plan_path, project_dir)
        if concat_only:
            rendered = self._existing_clips(minimax_plan_path, selected_scenes)
        else:
            use_case = build_render_video_scenes_use_case(
                RenderVideoCompositionOptions(
                    app_config_path="app_config.json",
                    project_config_path=project_dir / "config.json",
                    render_plan_path=minimax_plan_path,
                    workflow_path=self.workflow_path,
                    output_dir=self.output_dir,
                    video_pipeline=self.video_pipeline,
                )
            )
            rendered = use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=minimax_plan_path,
                    workflow_path=self.workflow_path,
                    audio_file=project_config.input_audio,
                    storyboard_dir=project_dir / "output" / "movie" / "storyboard",
                    output_dir=self.output_dir,
                    scene_numbers=set(selected_scenes) if selected_scenes else None,
                    upload_audio=False,
                    on_scene_complete=(
                        lambda path, completed, total: on_clip_rendered(
                            completed, total, _scene_number(path)
                        )
                        if on_clip_rendered
                        else None
                    ),
                )
            )
        if not rendered:
            raise ValueError("Movie render plan has no MiniMax scenes to concatenate")
        concat_list = self.postprocessor.write_concat_list(rendered, self.output_dir / "concat_list.txt")
        return self.postprocessor.concat_clips(
            concat_list,
            project_dir / "output" / "movie" / f"{project_dir.name}.mp4",
            video_only=False,
            reencode=True,
        )

    def _prepare_render_plan(self, render_plan_path: Path, project_dir: Path) -> Path:
        """Materialize MiniMax scenes with H3 prompts and reference paths."""
        source = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
        if isinstance(source, list):
            plan = {"shots": source}
        else:
            plan = dict(source)
        manifest = _load_movie_manifest(project_dir)
        scenes = []
        for index, raw_scene in enumerate(plan.get("shots") or plan.get("scenes") or [], start=1):
            scene = dict(raw_scene)
            scene["scene"] = int(scene.get("scene") or scene.get("scene_number") or index)
            scene["references"] = scene.get("references") or _references_from_ids(scene, manifest, project_dir)
            if self.video_pipeline == "minimax-h3-r2v" and not scene["references"].get("actor_msr_paths"):
                raise ValueError(f"Movie scene {scene['scene']} is missing actor MSR references")
            scene["h3"] = {
                **dict(scene.get("h3") or {}),
                "prompt": _h3_movie_prompt(scene),
            }
            scenes.append(scene)
        plan["shots"] = scenes
        output = self.output_dir / "render_plan_h3.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output

    def _existing_clips(self, render_plan_path: Path, selected_scenes: list[int] | None) -> list[Path]:
        plan = JsonArtifactStore().read_render_plan(render_plan_path)
        scenes = plan if isinstance(plan, list) else (plan.get("shots") or plan.get("scenes") or [])
        selected = set(selected_scenes or [])
        result = []
        for scene in scenes:
            number = int(scene.get("scene") or scene.get("scene_number"))
            if selected and number not in selected:
                continue
            candidates = (
                self.output_dir / "final" / f"scene_{number:04}.mp4",
                self.output_dir / f"scene_{number:04}.mp4",
                self.output_dir / f"scene_{number:04}" / "final.mp4",
            )
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                raise ValueError(f"Cannot build final movie; missing rendered MiniMax scene clip: {path}")
            result.append(path)
        return result


def _scene_number(path: Path) -> int:
    return int(path.parent.name.rsplit("_", 1)[-1])


def _load_movie_manifest(project_dir: Path) -> dict:
    path = project_dir / "movie" / "references" / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _h3_movie_prompt(scene: dict) -> str:
    """Build a compact H3 prompt while preserving existing enriched prompts."""
    existing = str((scene.get("h3") or {}).get("prompt") or "").strip()
    relay_prompt = _format_relay_shots(_normalize_relay_segments(scene))
    references = scene.get("references") or {}
    reference_paths = list(references.get("actor_msr_paths") or references.get("actor_sheet_paths") or [])
    location_path = references.get("location_msr_path") or references.get("location_sheet_path")
    if location_path:
        reference_paths.append(location_path)
    reference_prompt = (
        "Reference files:\n" + "\n".join(f"<Picture {index}>: {path}" for index, path in enumerate(reference_paths, start=1))
        if reference_paths else ""
    )
    if existing:
        return "\n\n".join(part for part in (existing, reference_prompt, relay_prompt) if part)
    descriptions = list(references.get("actor_reference_descriptions") or [])
    location = references.get("location_reference_description") or {}
    definitions = []
    for index, item in enumerate(descriptions, start=1):
        name = str(item.get("name") or item.get("id") or f"Subject {index}")
        description = str(item.get("visual_description") or item.get("image_prompt") or name).strip()
        definitions.append(f"<Subject {index}> ({name}): {description} Source references: <Picture {index}>.")
    if location:
        index = len(descriptions) + 1
        name = str(location.get("name") or location.get("id") or "Environment")
        description = str(location.get("visual_description") or location.get("image_prompt") or name).strip()
        definitions.append(f"<Subject {index}> ({name}): {description} Source references: <Picture {index}>.")
    prompt = str((scene.get("ltx") or {}).get("msr_global_prompt") or "").strip()
    prompt = re.sub(r"Reference image (\d+)", r"<Picture \1>", prompt, flags=re.IGNORECASE)
    action = _movie_video_prompt(scene, bible={}, manifest={})
    parts = ["subject_definitions:", "\n".join(definitions), f"summary: {action}", prompt]
    parts.extend(part for part in (reference_prompt, relay_prompt) if part)
    return "\n\n".join(part for part in parts if part.strip())