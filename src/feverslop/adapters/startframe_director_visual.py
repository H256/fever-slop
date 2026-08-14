from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image

from feverslop.adapters.movie_visual import write_local_placeholder_clip

class LocalStartframeDirectorVisualAdapter:
    def render_movie(
        self,
        *,
        project_dir: Path,
        render_plan_path: Path,
        selected_scenes: list[int] | None = None,
        concat_only: bool = False,
        continuity_keyframes: str = "none",
        on_clip_rendered: Callable[[int, int, int], None] | None = None,
        on_startframe_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        project_dir = Path(project_dir)
        final_dir = project_dir / "output" / "movie" / "storyboard" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        scenes = _read_render_plan(render_plan_path)
        clip_dir = project_dir / "output" / "movie" / "startframe-director" / "final"
        clip_dir.mkdir(parents=True, exist_ok=True)
        total = len(scenes) or 1
        for index, scene in enumerate(scenes, start=1):
            scene_number = int(scene.get("scene") or index)
            image = Image.new("RGB", (int(scene.get("width") or 1280), int(scene.get("height") or 704)), "white")
            image.save(final_dir / f"scene_{scene_number:04}.png")
            write_local_placeholder_clip(
                clip_dir / f"scene_{scene_number:04}.mp4",
                duration_seconds=float(scene.get("duration_seconds") or 1.0),
            )
            if on_startframe_step is not None:
                on_startframe_step({"kind": "validated-startframe", "completed": index, "total": total, "scene": scene_number})
        if on_startframe_step is not None:
            on_startframe_step({"kind": "validation", "completed": total, "total": total, "scene": int(scenes[-1].get("scene") or total) if scenes else 1})
        output_dir = project_dir / "output" / "movie"
        output_dir.mkdir(parents=True, exist_ok=True)
        final = output_dir / f"{project_dir.name}.mp4"
        write_local_placeholder_clip(final)
        if on_clip_rendered is not None:
            on_clip_rendered(1, 1, int(scenes[0].get("scene") or 1) if scenes else 1)
        return final


def _read_render_plan(path: Path) -> list[dict[str, Any]]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [scene for scene in data if isinstance(scene, dict)]
    return []
