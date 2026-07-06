from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.ports.rendering import VideoRenderRequest


class LocalMovieVisualAdapter:
    """Local placeholder adapter for movie production tests and offline Studio use."""

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
        output_dir = project_dir / "output" / "movie"
        output_dir.mkdir(parents=True, exist_ok=True)
        final = output_dir / f"{project_dir.name}.mp4"
        final.write_bytes(b"feverslop movie placeholder\n" + render_plan_path.read_bytes())
        if on_clip_rendered is not None:
            plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
            total = len(plan.get("shots") or plan.get("scenes") or []) or 1
            for completed in range(1, total + 1):
                on_clip_rendered(completed, total, completed)
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
        i2v_workflow_path: str | Path | None = None,
        i2v_workflow: dict | None = None,
        continuity_keyframes: str = "none",
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.render_queue = render_queue
        self.asset_uploader = asset_uploader
        self.postprocessor = postprocessor or VideoPostProcessor()
        self.model_resolver = model_resolver
        self.fps = int(fps)
        self.workflow = workflow
        self.i2v_workflow_path = Path(i2v_workflow_path) if i2v_workflow_path is not None else None
        self.i2v_workflow = i2v_workflow
        self.continuity_keyframes = _continuity_keyframes(continuity_keyframes)

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
        project_dir = Path(project_dir)
        plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
        output_dir = project_dir / "output" / "movie" / "ltx_msr"
        final_output = project_dir / "output" / "movie" / f"{_movie_slug(plan, project_dir)}.mp4"
        scenes = self._movie_scenes(plan, project_dir=project_dir)
        if not scenes:
            raise ValueError("Movie render plan has no shots or scenes to render")

        backend = self._build_backend(
            workflow_path=self.workflow_path,
            workflow=self.workflow,
            output_dir=output_dir,
            project_dir=project_dir,
        )
        i2v_backend = self._build_i2v_backend(output_dir=output_dir, project_dir=project_dir)
        selected = {int(scene) for scene in selected_scenes or []}
        keyframe_mode = _continuity_keyframes(continuity_keyframes if continuity_keyframes != "none" else self.continuity_keyframes)
        if selected and keyframe_mode == "last-to-start":
            _validate_selected_continuity_dependencies(scenes, output_dir=output_dir, selected=selected)
        rendered = []
        renderable_scenes = [scene for scene in scenes if not selected or int(scene["scene"]) in selected or not (output_dir / f"scene_{int(scene['scene']):04}.mp4").exists()]
        total = len(renderable_scenes) or len(scenes)
        rendered_count = 0
        for index, scene in enumerate(scenes):
            scene_number = int(scene["scene"])
            clip_path = output_dir / f"scene_{scene_number:04}.mp4"
            if concat_only and not clip_path.exists():
                raise ValueError(f"Cannot build final movie; missing rendered movie scene clip: {clip_path}")
            should_render = not concat_only and (not selected or scene_number in selected or not clip_path.exists())
            if should_render:
                use_i2v = self._attach_continuity_startframe(
                    scene,
                    scene_number=scene_number,
                    previous_clip=output_dir / f"scene_{scene_number - 1:04}.mp4",
                    previous_scene=scenes[index - 1] if index > 0 else None,
                    project_dir=project_dir,
                    mode=keyframe_mode,
                    selected=selected,
                )
                scene_backend = i2v_backend if use_i2v and i2v_backend is not None else backend
                clip_path = scene_backend.render_video(
                    VideoRenderRequest(
                        scene=scene,
                        scene_number=scene_number,
                        prompt=str((scene.get("ltx") or {}).get("original_style_i2v_prompt") or scene.get("description") or ""),
                        workflow_path=scene_backend.workflow_path,
                        output_dir=output_dir,
                        audio_file=project_dir / "movie" / "ltx_native_audio.wav",
                        storyboard_dir=project_dir / "movie" / "storyboard",
                        upload_audio=False,
                    )
                )
                rendered_count += 1
                if on_clip_rendered is not None:
                    on_clip_rendered(rendered_count, total, scene_number)
            rendered.append(clip_path)

        concat_list = self.postprocessor.write_concat_list(rendered, output_dir / "concat_list.txt")
        return self.postprocessor.concat_clips(concat_list, final_output, video_only=False, reencode=True)

    def _build_backend(self, *, workflow_path: Path, workflow: dict | None, output_dir: Path, project_dir: Path) -> ComfyUIMSRVideoRenderBackend:
        return ComfyUIMSRVideoRenderBackend(
            client=self.client,
            workflow_path=workflow_path,
            output_dir=output_dir,
            project_dir=project_dir,
            asset_uploader=self.asset_uploader,
            render_queue=self.render_queue,
            postprocessor=self.postprocessor,
            model_resolver=self.model_resolver,
            debug_workflows_dir=project_dir / "output" / "movie" / "ltx_msr_debug",
            workflow=workflow,
            workflow_label=workflow_path,
        )

    def _build_i2v_backend(self, *, output_dir: Path, project_dir: Path) -> ComfyUIMSRVideoRenderBackend | None:
        if self.i2v_workflow_path is None and self.i2v_workflow is None:
            return None
        workflow_path = self.i2v_workflow_path or self.workflow_path
        return ComfyUIMSRVideoRenderBackend(
            client=self.client,
            workflow_path=workflow_path,
            output_dir=output_dir,
            project_dir=project_dir,
            asset_uploader=self.asset_uploader,
            render_queue=self.render_queue,
            postprocessor=self.postprocessor,
            model_resolver=self.model_resolver,
            debug_workflows_dir=project_dir / "output" / "movie" / "ltx_msr_debug",
            workflow=self.i2v_workflow,
            workflow_label=workflow_path,
            preroll_frames=0,
        )

    def _attach_continuity_startframe(
        self,
        scene: dict[str, Any],
        *,
        scene_number: int,
        previous_clip: Path,
        previous_scene: dict[str, Any] | None,
        project_dir: Path,
        mode: str,
        selected: set[int],
    ) -> bool:
        if mode != "last-to-start" or scene_number <= 1:
            return False
        if _transition_from_previous(scene) != "continuous":
            return False
        if not _continuous_transition_is_valid(scene, previous_scene=previous_scene):
            return False
        if not previous_clip.exists():
            detail = " for selected re-render" if selected else ""
            raise ValueError(f"Cannot use last-frame continuity{detail}; missing previous movie scene clip: {previous_clip}")
        output_file = project_dir / "output" / "movie" / "keyframes" / f"scene_{scene_number - 1:04}_to_{scene_number:04}_start.png"
        startframe_path = self.postprocessor.extract_last_frame(previous_clip, output_file)
        keyframes = dict(scene.get("keyframes") or {})
        keyframes["startframe_path"] = startframe_path.as_posix()
        keyframes["startframe_source_scene"] = scene_number - 1
        keyframes["startframe_mode"] = "last_frame_from_previous"
        scene["keyframes"] = keyframes
        ltx = dict(scene.get("ltx") or {})
        ltx["msr_continuity_handoff_prompt"] = _continuity_handoff_prompt(previous_scene)
        ltx["msr_continuity_handoff_frames"] = 18
        ltx["msr_continuity_msr_frame_count"] = 17
        ltx["msr_continuity_guide_frame_idx"] = 18
        scene["ltx"] = ltx
        return True

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
            scene["transition_from_previous"] = _transition_from_previous(scene)
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


def _continuity_keyframes(value: object) -> str:
    mode = str(value or "none").strip().lower()
    if mode not in {"none", "last-to-start"}:
        raise ValueError("continuity_keyframes must be one of: last-to-start, none")
    return mode


def _transition_from_previous(scene: dict[str, Any]) -> str:
    transition = str(scene.get("transition_from_previous") or "cut").strip().lower().replace("_", "-")
    return "continuous" if transition == "continuous" else "cut"


def _continuity_handoff_prompt(previous_scene: dict[str, Any] | None) -> str:
    fallback = "Hold the previous scene end state as the shot begins."
    if not previous_scene:
        return fallback
    ltx = previous_scene.get("ltx") or {}
    relays = ltx.get("msr_prompt_relay") or []
    if relays and isinstance(relays[-1], dict):
        prompt = str(relays[-1].get("prompt") or "").strip()
        if prompt:
            return prompt
    for key in ("original_style_i2v_prompt", "base_prompt"):
        prompt = str(ltx.get(key) or "").strip()
        if prompt:
            return prompt
    return str(previous_scene.get("description") or "").strip() or fallback


def _continuous_transition_is_valid(scene: dict[str, Any], *, previous_scene: dict[str, Any] | None) -> bool:
    if int(scene.get("scene") or 0) <= 1 or not previous_scene:
        return False
    current_refs = scene.get("reference_ids") if isinstance(scene.get("reference_ids"), dict) else {}
    previous_refs = previous_scene.get("reference_ids") if isinstance(previous_scene.get("reference_ids"), dict) else {}
    current_location = str(scene.get("location_id") or current_refs.get("location") or "").strip()
    previous_location = str(previous_scene.get("location_id") or previous_refs.get("location") or "").strip()
    if current_location and previous_location and current_location != previous_location:
        return False
    current_actors = _actor_ids(scene, current_refs)
    previous_actors = _actor_ids(previous_scene, previous_refs)
    if current_actors and previous_actors and not current_actors.intersection(previous_actors):
        return False
    return True


def _actor_ids(scene: dict[str, Any], refs: dict[str, Any]) -> set[str]:
    return {str(item).strip() for item in (scene.get("actor_ids") or refs.get("actors") or []) if str(item).strip()}


def _validate_selected_continuity_dependencies(scenes: list[dict[str, Any]], *, output_dir: Path, selected: set[int]) -> None:
    scenes_by_number = {int(scene["scene"]): scene for scene in scenes}
    for scene_number in sorted(selected):
        scene = scenes_by_number.get(scene_number)
        if not scene or scene_number <= 1 or _transition_from_previous(scene) != "continuous":
            continue
        previous_clip = output_dir / f"scene_{scene_number - 1:04}.mp4"
        if not previous_clip.exists():
            raise ValueError(f"Cannot use last-frame continuity for selected re-render; missing previous movie scene clip: {previous_clip}")


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
    actor_items = [_required_manifest_item(actors, actor_id, "actor") for actor_id in actor_ids]
    location_item = _required_manifest_item(locations, location_id, "location") if location_id else {}
    actor_paths = [_required_manifest_path(actor, project_dir) for actor in actor_items]
    location_path = _required_manifest_path(location_item, project_dir) if location_item else ""
    if not actor_paths or not location_path:
        return {}
    return {
        "actor_msr_paths": [path.as_posix() for path in actor_paths],
        "location_msr_path": location_path.as_posix(),
        "actor_reference_descriptions": [_reference_description(actor) for actor in actor_items],
        "location_reference_description": _reference_description(location_item),
    }


def _required_manifest_item(items: dict[str, dict], item_id: str, kind: str) -> dict:
    item = items.get(item_id)
    if not item:
        raise ValueError(f"Movie {kind} reference id is missing from manifest: {item_id}")
    return item


def _required_manifest_path(item: dict, project_dir: Path) -> Path:
    value = str(item.get("msr_sheet_path") or item.get("path") or "").strip()
    if not value:
        raise ValueError(f"Movie reference {item.get('id')} has no rendered MSR sheet path")
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _reference_description(item: dict) -> dict[str, str]:
    return {
        "id": str(item.get("id") or ""),
        "name": str(item.get("name") or item.get("id") or ""),
        "role": str(item.get("role") or ""),
        "visual_description": str(item.get("visual_description") or ""),
        "image_prompt": str(item.get("image_prompt") or ""),
    }
