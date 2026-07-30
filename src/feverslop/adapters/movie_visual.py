from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.adapters.postprocessor_frame_extractor import (
    PostprocessorFrameExtractor,
)
from feverslop.domain.visual_consistency import (
    ReferenceAnchor,
    SceneConsistencyContract,
    can_handoff,
    expand_handoff_selection,
    validate_scene_sequence,
)
from feverslop.domain.visual_consistency_runtime import (
    reference_look_id,
    resolve_reference_look,
)
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.domain.movie_utils import transition_from_previous
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
        continuity_handoff_factory=None,
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
        self.continuity_handoff_factory = (
            continuity_handoff_factory
            or _default_continuity_handoff_factory
        )

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
            contracts = [
                contract
                for scene in scenes
                if (contract := _continuity_contract(scene)) is not None
            ]
            selected = expand_handoff_selection(contracts, selected)
        if selected and keyframe_mode == "last-to-start":
            _validate_selected_continuity_dependencies(
                scenes,
                output_dir=output_dir,
                selected=selected,
            )
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
            workflow_profile=workflow_path.stem,
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
            workflow_profile=workflow_path.stem,
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
        previous_contract = _continuity_contract(previous_scene)
        current_contract = _continuity_contract(scene)
        if (
            previous_contract is None
            or current_contract is None
            or previous_contract.scene + 1 != current_contract.scene
            or not can_handoff(previous_contract, current_contract)
        ):
            return False
        output_file = project_dir / "output" / "movie" / "keyframes" / f"scene_{scene_number - 1:04}_to_{scene_number:04}_start.png"
        attached = self.continuity_handoff_factory(
            self.postprocessor,
            project_dir,
            bool(selected),
        ).execute(
            previous_contract,
            current_contract,
            previous_clip,
            output_file,
            scene,
            handoff_prompt=_continuity_handoff_prompt(previous_scene),
        )
        scene.clear()
        scene.update(attached)
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
            scene["references"] = scene.get("references") or _references_from_ids(
                scene,
                reference_manifest,
                project_dir,
            )
            scene["transition_from_previous"] = transition_from_previous(scene.get("transition_from_previous"))
            if not scene.get("references"):
                raise ValueError(f"Movie shot {scene['scene']} is missing MSR references")
            scenes.append(scene)
            cursor = scene["abs_start_seconds"] + duration
        validate_scene_sequence(scenes)
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


def _validate_selected_continuity_dependencies(scenes: list[dict[str, Any]], *, output_dir: Path, selected: set[int]) -> None:
    scenes_by_number = {int(scene["scene"]): scene for scene in scenes}
    for scene_number in sorted(selected):
        scene = scenes_by_number.get(scene_number)
        previous_scene = scenes_by_number.get(scene_number - 1)
        previous_contract = _continuity_contract(previous_scene)
        current_contract = _continuity_contract(scene)
        if (
            previous_contract is None
            or current_contract is None
            or previous_contract.scene + 1 != current_contract.scene
            or not can_handoff(previous_contract, current_contract)
        ):
            continue
        if scene_number - 1 in selected:
            continue
        previous_clip = output_dir / f"scene_{scene_number - 1:04}.mp4"
        if not previous_clip.exists():
            raise ValueError(f"Cannot use last-frame continuity for selected re-render; missing previous movie scene clip: {previous_clip}")


def _continuity_contract(
    scene: dict[str, Any] | None,
) -> SceneConsistencyContract | None:
    if not scene:
        return None
    stored = scene.get("visual_consistency")
    if isinstance(stored, dict):
        try:
            return SceneConsistencyContract.from_dict(stored)
        except (KeyError, TypeError, ValueError):
            return None
    reference_ids = (
        scene.get("reference_ids")
        if isinstance(scene.get("reference_ids"), dict)
        else {}
    )
    references = (
        scene.get("references")
        if isinstance(scene.get("references"), dict)
        else {}
    )
    actor_ids = (
        scene.get("actor_ids")
        or reference_ids.get("actors")
        or references.get("actor_ids")
        or references.get("actor_msr_paths")
        or references.get("actor_sheet_paths")
        or []
    )
    location_id = str(
        scene.get("location_id")
        or reference_ids.get("location")
        or references.get("location_id")
        or references.get("location_msr_path")
        or references.get("location_sheet_path")
        or ""
    ).strip()
    actors = tuple(
        _synthetic_anchor(str(actor_id).strip(), kind="actor")
        for actor_id in actor_ids
        if str(actor_id).strip()
    )
    if not actors or not location_id:
        return None
    return SceneConsistencyContract.create(
        scene=int(scene.get("scene") or 0),
        mode="msr",
        workflow_profile="movie-continuity",
        actors=actors,
        location=_synthetic_anchor(location_id, kind="location"),
        transition_from_previous=transition_from_previous(
            scene.get("transition_from_previous")
        ),
    )


def _synthetic_anchor(value: str, *, kind: str) -> ReferenceAnchor:
    return ReferenceAnchor(
        id=value,
        kind=kind,
        look_id="default",
        asset_role=(
            "identity-reference"
            if kind == "actor"
            else "environment-reference"
        ),
        asset_sha256=hashlib.sha256(value.encode("utf-8")).hexdigest(),
        prompt_anchor=value,
    )


def _default_continuity_handoff_factory(
    postprocessor,
    project_dir: Path,
    selected_rerender: bool,
):
    module = importlib.import_module(
        "feverslop.application.continuity_handoff"
    )
    return module.ContinuityHandoffUseCase(
        PostprocessorFrameExtractor(
            postprocessor,
            project_dir=project_dir,
            selected_rerender=selected_rerender,
        )
    )


def _load_reference_manifest(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "movie" / "references" / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _references_from_ids(scene: dict[str, Any], manifest: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    reference_ids = scene.get("reference_ids") or {}
    actor_ids = [str(actor_id) for actor_id in reference_ids.get("actors") or []]
    location_id = str(reference_ids.get("location") or "")
    actors = {str(actor.get("id")): actor for actor in manifest.get("actors") or []}
    locations = {str(location.get("id")): location for location in manifest.get("locations") or []}
    actor_items = [
        resolve_reference_look(
            _required_manifest_item(actors, actor_id, "actor"),
            reference_look_id(scene, kind="actor", semantic_id=actor_id),
        )
        for actor_id in actor_ids
    ]
    location_item = (
        resolve_reference_look(
            _required_manifest_item(locations, location_id, "location"),
            reference_look_id(
                scene,
                kind="location",
                semantic_id=location_id,
            ),
        )
        if location_id
        else {}
    )
    actor_paths = [_required_manifest_path(actor, project_dir) for actor in actor_items]
    location_path = _required_manifest_path(location_item, project_dir) if location_item else ""
    if not actor_paths or not location_path:
        return {}
    return {
        "actor_ids": actor_ids,
        "location_id": location_id,
        "actor_msr_paths": [
            path.relative_to(project_dir.resolve()).as_posix()
            for path in actor_paths
        ],
        "location_msr_path": location_path.relative_to(
            project_dir.resolve()
        ).as_posix(),
        "actor_reference_descriptions": [_reference_description(actor) for actor in actor_items],
        "location_reference_description": _reference_description(location_item),
    }


def _required_manifest_item(items: dict[str, dict], item_id: str, kind: str) -> dict:
    item = items.get(item_id)
    if not item:
        raise ValueError(f"Movie {kind} reference id is missing from manifest: {item_id}")
    return item


def _required_manifest_path(item: dict, project_dir: Path) -> Path:
    value = str(
        (
            item.get("sheet_path")
            if item.get("look_id") != "default"
            else (
                item.get("msr_sheet_path")
                or item.get("path")
                or item.get("sheet_path")
            )
        )
        or ""
    ).strip()
    if not value:
        raise ValueError(f"Movie reference {item.get('id')} has no rendered MSR sheet path")
    path = Path(value)
    resolved = (
        path.resolve()
        if path.is_absolute()
        else (project_dir / path).resolve()
    )
    if not resolved.is_relative_to(project_dir.resolve()):
        raise ValueError(
            f"Movie reference {item.get('id')!r} path must be inside the "
            f"project: {value}"
        )
    return resolved


def _reference_description(item: dict) -> dict[str, str]:
    return {
        "id": str(item.get("id") or ""),
        "name": str(item.get("name") or item.get("id") or ""),
        "role": str(item.get("role") or ""),
        "visual_description": str(item.get("visual_description") or ""),
        "image_prompt": str(item.get("image_prompt") or ""),
    }
