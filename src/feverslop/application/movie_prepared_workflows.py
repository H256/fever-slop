from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from feverslop.ports.workflow import WorkflowMaterializationRequest
from feverslop.scene_artifacts import SceneArtifactLayout


def prepare_movie_workflows(
    *,
    project_dir: Path,
    render_plan_path: Path,
    pipeline: str,
    scenes: list[dict[str, Any]],
    selected_scenes: Iterable[int] | None,
    materializer: Any,
    prompt_for_scene: Callable[[dict[str, Any]], str],
) -> Path:
    selected = {int(number) for number in selected_scenes or ()}
    chosen = [scene for scene in scenes if not selected or int(scene["scene"]) in selected]
    errors: list[str] = []
    for scene in chosen:
        number = int(scene["scene"])
        for relay in (scene.get("ltx") or {}).get("prompt_relay") or []:
            if str(relay.get("state") or "").strip().lower() == "singing":
                prompt = str(relay.get("prompt") or "").lower()
                if "sing" not in prompt or ("lip sync" not in prompt and "lip-sync" not in prompt):
                    errors.append(f"scene {number}: singing relay requires singing and lip sync")
        if pipeline == "ltx_ingredients":
            anchors = scene.get("ingredients_scene_sheet_anchors") or []
            target = str(
                scene.get("ingredients_target_prompt")
                or (scene.get("ltx") or {}).get("ingredients_target_prompt")
                or ""
            )
            unbound = sorted(
                str(anchor.get("id") or "") for anchor in anchors
                if f"`{str(anchor.get('id') or '')}`" not in target
            )
            if unbound:
                errors.append(
                    f"scene {number}: target description does not bind anchors {', '.join(unbound)}"
                )
    if errors:
        raise ValueError("Cannot prepare scene workflows:\n- " + "\n- ".join(errors))
    layout = SceneArtifactLayout(project_dir)
    paths = [
        path
        for scene in chosen
        for path in (
            layout.scene_workflow(int(scene["scene"])),
            layout.scene_manifest(int(scene["scene"])),
        )
    ]
    previous = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        for scene in chosen:
            materializer.prepare(
                WorkflowMaterializationRequest(
                    scene=scene,
                    prompt=prompt_for_scene(scene),
                    audio_file=None,
                    render_plan_path=Path(render_plan_path),
                    pipeline=pipeline,
                )
            )
    except Exception:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise
    return layout.scenes_dir


def render_prepared_movie_workflows(
    *,
    project_dir: Path,
    scenes: list[dict[str, Any]],
    selected_scenes: Iterable[int] | None,
    renderer: Any,
    postprocessor: Any,
    legacy_dirs: Iterable[Path] = (),
    on_clip_rendered: Callable[[int, int, int], None] | None = None,
) -> Path:
    layout = SceneArtifactLayout(project_dir)
    selected = {int(number) for number in selected_scenes or ()}
    to_render = [scene for scene in scenes if not selected or int(scene["scene"]) in selected]
    missing = [
        int(scene["scene"])
        for scene in to_render
        if not layout.scene_manifest(int(scene["scene"])).is_file()
        or not layout.scene_workflow(int(scene["scene"])).is_file()
    ]
    if missing:
        numbers = ", ".join(str(number) for number in missing)
        raise FileNotFoundError(
            f"Prepared movie scenes missing: {numbers}; prepare them first with --write-debug-workflows"
        )

    for completed, scene in enumerate(to_render, start=1):
        scene_number = int(scene["scene"])
        renderer.render(layout.scene_workflow(scene_number))
        if on_clip_rendered is not None:
            on_clip_rendered(completed, len(to_render), scene_number)

    clips: list[Path] = []
    for scene in scenes:
        scene_number = int(scene["scene"])
        clip = layout.find_scene_final_video(scene_number, legacy_dirs=legacy_dirs)
        if clip is None:
            raise FileNotFoundError(f"Cannot build final movie; missing rendered movie scene {scene_number}")
        clips.append(clip)
    concat_list = postprocessor.write_concat_list(clips, layout.render_dir / "concat_list.txt")
    layout.final_dir.mkdir(parents=True, exist_ok=True)
    return postprocessor.concat_clips(concat_list, layout.movie, video_only=False, reencode=True)
