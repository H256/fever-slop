from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.ports.rendering import VideoRenderRequest


class LocalMovieVisualAdapter:
    """Local placeholder adapter for movie production tests and offline Studio use."""

    def render_movie(self, *, project_dir: Path, render_plan_path: Path) -> Path:
        output_dir = project_dir / "output" / "movie"
        output_dir.mkdir(parents=True, exist_ok=True)
        final = output_dir / f"{project_dir.name}.mp4"
        final.write_bytes(b"feverslop movie placeholder\n" + render_plan_path.read_bytes())
        return final


class ComfyUIMovieVisualAdapter:
    def __init__(
        self,
        *,
        client: object,
        workflow_path: str | Path,
        render_queue=None,
        asset_uploader=None,
        postprocessor: VideoPostProcessor | None = None,
        model_resolver=None,
        fps: int = 24,
        workflow: dict | None = None,
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.render_queue = render_queue
        self.asset_uploader = asset_uploader
        self.postprocessor = postprocessor or VideoPostProcessor()
        self.model_resolver = model_resolver
        self.fps = int(fps)
        self.workflow = workflow

    def render_movie(self, *, project_dir: Path, render_plan_path: Path) -> Path:
        project_dir = Path(project_dir)
        plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
        output_dir = project_dir / "output" / "movie" / "ltx_msr"
        final_output = project_dir / "output" / "movie" / f"{_movie_slug(plan, project_dir)}.mp4"
        scenes = self._movie_scenes(plan, project_dir=project_dir)
        if not scenes:
            raise ValueError("Movie render plan has no shots or scenes to render")

        backend = ComfyUIMSRVideoRenderBackend(
            client=self.client,
            workflow_path=self.workflow_path,
            output_dir=output_dir,
            project_dir=project_dir,
            asset_uploader=self.asset_uploader,
            render_queue=self.render_queue,
            postprocessor=self.postprocessor,
            model_resolver=self.model_resolver,
            debug_workflows_dir=project_dir / "output" / "movie" / "ltx_msr_debug",
            workflow=self.workflow,
            workflow_label=self.workflow_path,
        )
        rendered = []
        for scene in scenes:
            rendered.append(
                backend.render_video(
                    VideoRenderRequest(
                        scene=scene,
                        scene_number=int(scene["scene"]),
                        prompt=str((scene.get("ltx") or {}).get("original_style_i2v_prompt") or scene.get("description") or ""),
                        workflow_path=self.workflow_path,
                        output_dir=output_dir,
                        audio_file=project_dir / "movie" / "ltx_native_audio.wav",
                        storyboard_dir=project_dir / "movie" / "storyboard",
                        upload_audio=False,
                    )
                )
            )

        concat_list = self.postprocessor.write_concat_list(rendered, output_dir / "concat_list.txt")
        return self.postprocessor.concat_clips(concat_list, final_output, video_only=False)

    def _movie_scenes(self, plan: dict[str, Any], *, project_dir: Path) -> list[dict[str, Any]]:
        raw_scenes = plan.get("scenes") or plan.get("shots") or []
        resolution = plan.get("resolution") or {}
        fps = int(plan.get("fps") or self.fps)
        reference_manifest = _load_reference_manifest(project_dir)
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
            scene["ltx"] = {
                **dict(scene.get("ltx") or {}),
                "original_style_i2v_prompt": _movie_scene_prompt(scene),
            }
            scene["references"] = scene.get("references") or _references_from_ids(scene.get("reference_ids") or {}, reference_manifest, project_dir)
            if not scene.get("references"):
                raise ValueError(f"Movie shot {scene['scene']} is missing MSR references")
            scenes.append(scene)
            cursor = scene["abs_start_seconds"] + duration
        return scenes


def _movie_scene_prompt(scene: dict[str, Any]) -> str:
    ltx = dict(scene.get("ltx") or {})
    prompt = str(ltx.get("original_style_i2v_prompt") or ltx.get("base_prompt") or "").strip()
    if prompt:
        return prompt
    parts = [
        scene.get("description"),
        scene.get("action"),
        scene.get("camera"),
        scene.get("expression"),
        scene.get("location"),
    ]
    return ", ".join(str(part).strip() for part in parts if str(part or "").strip())


def _movie_slug(plan: dict[str, Any], project_dir: Path) -> str:
    title = str(plan.get("title") or "").strip()
    if not title:
        return project_dir.name
    return "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-") or project_dir.name


def _load_reference_manifest(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "movie" / "references" / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _references_from_ids(reference_ids: dict[str, Any], manifest: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    actor_ids = [str(actor_id) for actor_id in reference_ids.get("actors") or []]
    location_id = str(reference_ids.get("location") or "")
    actors = {str(actor.get("id")): actor for actor in manifest.get("actors") or []}
    locations = {str(location.get("id")): location for location in manifest.get("locations") or []}
    actor_paths = [_required_manifest_path(actors, actor_id, "actor", project_dir) for actor_id in actor_ids]
    location_path = _required_manifest_path(locations, location_id, "location", project_dir) if location_id else ""
    if not actor_paths or not location_path:
        return {}
    return {
        "actor_msr_paths": [path.as_posix() for path in actor_paths],
        "location_msr_path": location_path.as_posix(),
    }


def _required_manifest_path(items: dict[str, dict], item_id: str, kind: str, project_dir: Path) -> Path:
    item = items.get(item_id)
    if not item:
        raise ValueError(f"Movie {kind} reference id is missing from manifest: {item_id}")
    value = str(item.get("msr_sheet_path") or item.get("path") or "").strip()
    if not value:
        raise ValueError(f"Movie {kind} reference {item_id} has no rendered MSR sheet path")
    path = Path(value)
    return path if path.is_absolute() else project_dir / path
