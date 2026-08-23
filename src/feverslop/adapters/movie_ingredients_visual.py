from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from feverslop.adapters.comfyui_ingredients_video_backend import (
    ComfyUIIngredientsVideoRenderBackend,
)
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIMovieIngredientsVisualAdapter:
    """Visual adapter that renders a movie using the ingredients video pipeline."""

    def __init__(
        self,
        *,
        backend: ComfyUIIngredientsVideoRenderBackend,
    ):
        self.backend = backend

    def render_movie(
        self,
        *,
        project_dir: Path,
        render_plan_path: Path,
        on_clip_rendered: Callable[[int, int, int], None] | None = None,
    ) -> Path:
        project_dir = Path(project_dir)
        plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
        output_dir = self.backend.output_dir
        final_output = project_dir / "output" / "movie" / f"{_movie_slug(plan, project_dir)}.mp4"
        scenes = self._movie_scenes(plan, project_dir=project_dir)
        if not scenes:
            raise ValueError("Movie ingredients render plan has no shots or scenes to render")

        rendered = []
        renderable_scenes = [scene for scene in scenes if not (output_dir / f"scene_{int(scene['scene']):04}.mp4").exists()]
        total = len(renderable_scenes) or len(scenes)
        rendered_count = 0
        for scene in scenes:
            scene_number = int(scene["scene"])
            clip_path = output_dir / f"scene_{scene_number:04}.mp4"
            if clip_path.exists():
                rendered.append(clip_path)
                continue
            clip_path = self.backend.render_video(
                VideoRenderRequest(
                    scene=scene,
                    scene_number=scene_number,
                    prompt=_ingredients_prompt(scene),
                    workflow_path=self.backend.workflow_path,
                    output_dir=output_dir,
                    audio_file=Path(),
                    storyboard_dir=Path(),
                    upload_audio=False,
                ),
            )
            rendered.append(clip_path)
            rendered_count += 1
            if on_clip_rendered is not None:
                on_clip_rendered(rendered_count, total, scene_number)

        postprocessor = self.backend.postprocessor
        concat_list = postprocessor.write_concat_list(rendered, output_dir / "concat_list.txt")
        return postprocessor.concat_clips(concat_list, final_output, video_only=False, reencode=True)

    def _movie_scenes(self, plan: dict, *, project_dir: Path) -> list[dict]:
        raw_scenes = plan.get("scenes") or plan.get("shots") or []
        resolution = plan.get("resolution") or {}
        fps = int(plan.get("fps") or 24)
        cursor = 0.0
        scenes = []
        for index, raw in enumerate(raw_scenes, start=1):
            scene = dict(raw)
            scene["scene"] = int(scene.get("scene") or index)
            scene["fps"] = int(scene.get("fps") or fps)
            scene["width"] = int(scene.get("width") or resolution.get("width") or 1280)
            scene["height"] = int(scene.get("height") or resolution.get("height") or 704)
            duration = float(scene.get("duration_seconds") or 1.0)
            scene["duration_seconds"] = duration
            scene["frame_count"] = int(scene.get("frame_count") or max(1, round(duration * scene["fps"])))
            scene["abs_start_seconds"] = float(scene.get("abs_start_seconds", cursor) or 0.0)
            scenes.append(scene)
            cursor = scene["abs_start_seconds"] + duration
        return scenes


def _ingredients_prompt(scene: dict) -> str:
    ltx = scene.get("ltx") or {}
    ingredients = scene.get("ingredients") or {}
    static_prompt = str(ltx.get("static_prompt") or "").strip()
    global_prompt = str(ingredients.get("global_prompt") or scene.get("ingredients_global_prompt") or "").strip()
    scene_desc = str(ltx.get("ingredients_scene_sheet_description") or "").strip()
    target_prompt = str(ltx.get("ingredients_target_prompt") or "").strip()
    if static_prompt:
        return static_prompt
    if global_prompt:
        return global_prompt
    if scene_desc and target_prompt:
        return scene_desc + "\n" + target_prompt
    if scene_desc:
        return scene_desc
    if target_prompt:
        return target_prompt
    return str(scene.get("description") or "").strip()


def _movie_slug(plan: dict, project_dir: Path) -> str:
    title = str(plan.get("title") or "").strip()
    if not title:
        return project_dir.name
    return "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-") or project_dir.name
