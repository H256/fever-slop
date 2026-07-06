from __future__ import annotations

import json
from pathlib import Path

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter


class OfflineMovieAssetUploader:
    def resolve_audio_name(self, *_args, **_kwargs) -> str:
        raise AssertionError("offline movie workflow debug must not resolve audio")

    def resolve_reference_image_name(self, image_path, *, upload_references=True) -> str:
        return Path(image_path).name


def write_movie_debug_workflows(
    *,
    project_dir: Path,
    render_plan_path: Path,
    workflow_path: Path,
    workflow: dict,
    output_dir: Path,
) -> Path:
    project_dir = Path(project_dir)
    render_plan_path = Path(render_plan_path)
    workflow_path = Path(workflow_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(render_plan_path.read_text(encoding="utf-8"))
    scenes = ComfyUIMovieVisualAdapter(client=object(), workflow_path=workflow_path, workflow=workflow)._movie_scenes(plan, project_dir=project_dir)
    backend = ComfyUIMSRVideoRenderBackend(
        client=object(),
        workflow_path=workflow_path,
        output_dir=project_dir / "output" / "movie" / "ltx_msr",
        project_dir=project_dir,
        asset_uploader=OfflineMovieAssetUploader(),
        workflow=workflow,
        workflow_label=workflow_path,
    )
    for scene in scenes:
        scene_number = int(scene["scene"])
        patched = backend.build_workflow(
            scene,
            prompt=str((scene.get("ltx") or {}).get("original_style_i2v_prompt") or scene.get("description") or ""),
        )
        (output_dir / f"scene_{scene_number:04}_workflow.json").write_text(
            json.dumps(patched, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return output_dir
