from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from feverslop.ports.rendering import ImageRenderRequest, WorkflowAnchorConfig


class LocalMovieI2VEditVisualAdapter:
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
        output_dir = Path(project_dir) / "output" / "movie"
        output_dir.mkdir(parents=True, exist_ok=True)
        final = output_dir / f"{Path(project_dir).name}.mp4"
        final.write_bytes(b"feverslop movie i2v-edit placeholder\n" + Path(render_plan_path).read_bytes())
        if on_clip_rendered is not None:
            on_clip_rendered(1, 1, 1)
        return final


class ComfyUIMovieI2VEditVisualAdapter:
    def __init__(
        self,
        *,
        base_image_backend,
        edit_backend,
        artifact_store,
        video_use_case,
        workflow_path: Path,
        edit_workflow_path: Path,
        i2v_workflow_path: Path,
        postprocessor,
    ):
        self.base_image_backend = base_image_backend
        self.edit_backend = edit_backend
        self.artifact_store = artifact_store
        self.video_use_case = video_use_case
        self.workflow_path = Path(workflow_path)
        self.edit_workflow_path = Path(edit_workflow_path)
        self.i2v_workflow_path = Path(i2v_workflow_path)
        self.postprocessor = postprocessor

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
        storyboard_dir = self._render_startframes(project_dir=project_dir, render_plan_path=render_plan_path, on_startframe_step=on_startframe_step)
        ltx_dir = project_dir / "output" / "movie" / "ltx_i2v"
        rendered = self.video_use_case.execute(
            SimpleNamespace(
                render_plan_path=render_plan_path,
                workflow_path=self.i2v_workflow_path,
                single_prompt_workflow_path=self.i2v_workflow_path,
                audio_file=project_dir / "movie" / "ltx_native_audio.wav",
                storyboard_dir=storyboard_dir,
                output_dir=ltx_dir,
                render_mode="single_prompt",
                limit=None,
                scene_numbers=None,
                skip_existing=True,
                uploaded_audio_name=None,
                upload_audio=False,
                upload_startframes=True,
                anchors=WorkflowAnchorConfig(),
                on_scene_complete=on_clip_rendered,
            )
        )
        concat_list = self.postprocessor.write_concat_list(rendered, ltx_dir / "concat_list.txt")
        return self.postprocessor.concat_clips(
            concat_list,
            project_dir / "output" / "movie" / f"{project_dir.name}.mp4",
            video_only=False,
            reencode=True,
        )

    def _render_startframes(
        self,
        *,
        project_dir: Path,
        render_plan_path: Path,
        on_startframe_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        scenes = self.artifact_store.read_render_plan(render_plan_path)
        total_steps = _startframe_step_count(scenes)
        completed_steps = 0
        base_dir = project_dir / "output" / "movie" / "storyboard" / "base"
        edit_dir = project_dir / "output" / "movie" / "storyboard" / "edit"
        final_dir = project_dir / "output" / "movie" / "storyboard" / "final"
        base_dir.mkdir(parents=True, exist_ok=True)
        edit_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        for scene in scenes:
            scene_number = int(scene["scene"])
            base_path = self.base_image_backend.render_image(
                ImageRenderRequest(
                    scene=scene,
                    scene_number=scene_number,
                    prompt=str((scene.get("z_image") or {}).get("prompt") or ""),
                    workflow_path=self.workflow_path,
                    output_dir=base_dir,
                    width=int(scene.get("width") or 1280),
                    height=int(scene.get("height") or 704),
                )
            )
            completed_steps += 1
            _notify_startframe_step(
                on_startframe_step,
                kind="base",
                completed=completed_steps,
                total=total_steps,
                scene_number=scene_number,
            )
            current_path = base_path
            for edit_pass in (scene.get("movie") or {}).get("edit_passes") or []:
                current_path = self.edit_backend.render_edit(
                    prompt=str(edit_pass.get("prompt") or ""),
                    scene_number=scene_number,
                    plate_image=current_path,
                    character_image=project_dir / str(edit_pass["reference_image_path"]),
                    output_dir=edit_dir,
                    pass_number=int(edit_pass["pass"]),
                )
                completed_steps += 1
                _notify_startframe_step(
                    on_startframe_step,
                    kind="edit",
                    completed=completed_steps,
                    total=total_steps,
                    scene_number=scene_number,
                    actor_id=str(edit_pass.get("actor_id") or ""),
                    pass_number=int(edit_pass["pass"]),
                )
            final_path = final_dir / f"scene_{scene_number:04}.png"
            final_path.write_bytes(Path(current_path).read_bytes())
        return final_dir


def _startframe_step_count(scenes: list[dict[str, Any]]) -> int:
    total = 0
    for scene in scenes:
        total += 1
        total += len((scene.get("movie") or {}).get("edit_passes") or [])
    return total


def _notify_startframe_step(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    kind: str,
    completed: int,
    total: int,
    scene_number: int,
    actor_id: str = "",
    pass_number: int | None = None,
) -> None:
    if callback is None:
        return
    event: dict[str, Any] = {
        "kind": kind,
        "completed": completed,
        "total": total,
        "scene": scene_number,
    }
    if actor_id:
        event["actor_id"] = actor_id
    if pass_number is not None:
        event["pass"] = pass_number
    callback(event)
