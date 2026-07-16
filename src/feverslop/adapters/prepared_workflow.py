from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from feverslop.domain.prepared_workflow import (
    PreparedSceneWorkflow,
    SceneWorkflowManifest,
    StoredArtifact,
)
from feverslop.domain.postprocessing import TrimSpec
from feverslop.ports.workflow import WorkflowMaterializationRequest
from feverslop.scene_artifacts import SceneArtifactLayout


def _write_json_temp(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    return temporary


class WorkflowMaterializer:
    """Build and persist a render-ready workflow without submitting it."""

    def __init__(self, backend: Any, layout: SceneArtifactLayout):
        self.backend = backend
        self.layout = layout

    def prepare(self, request: WorkflowMaterializationRequest) -> PreparedSceneWorkflow:
        scene = request.scene
        scene_number = int(scene["scene"])
        manifest_path = self.layout.scene_manifest(scene_number)
        seed = int(request.seed) if request.seed is not None else int(self.backend._seed_for_scene(scene_number))
        original_uploader = self.backend.asset_uploader
        recording_uploader = _RecordingUploader(original_uploader, self.layout.project_dir)
        self.backend.asset_uploader = recording_uploader
        try:
            audio_name = None
            if request.audio_file is not None:
                audio_name = recording_uploader.resolve_audio_name(
                    request.audio_file, upload_audio=True, uploaded_audio_name=None,
                )
            rolling = self.backend._rolling_spec(scene)
            old_offset = self.backend.seed_offset
            old_randomize = self.backend.randomize_seed
            try:
                self.backend.seed_offset = seed - scene_number
                self.backend.randomize_seed = False
                workflow = self.backend.build_workflow(
                    scene, prompt=request.prompt, comfy_audio_name=audio_name, rolling=rolling,
                )
            finally:
                self.backend.seed_offset = old_offset
                self.backend.randomize_seed = old_randomize
            workflow = self.backend.model_resolver.resolve_workflow_models(
                workflow, workflow_path=self.backend.workflow_label,
            )
        finally:
            self.backend.asset_uploader = original_uploader

        workflow_path = self.layout.scene_workflow(scene_number)
        temporary_workflow = _write_json_temp(workflow_path, workflow)
        temporary_manifest: Path | None = None
        try:
            assets = self._manifest_assets(scene, request.audio_file, recording_uploader.names)
            manifest = SceneWorkflowManifest.create(
                project_dir=self.layout.project_dir,
                scene=scene_number,
                pipeline=request.pipeline,
                workflow_path=temporary_workflow,
                template_path=self.backend.workflow_label,
                render_plan_path=request.render_plan_path,
                assets=assets,
                seed=seed,
                fps=int(scene.get("fps") or _rolling_value(rolling, "fps") or 0),
                frame_count=int(scene.get("frame_count") or _rolling_value(rolling, "render_frame_count") or 0),
                render_frame_count=int(_rolling_value(rolling, "render_frame_count") or scene.get("frame_count") or 0),
                trim_front_frames=int(_rolling_value(rolling, "trim_front_frames") or 0),
                width=int(scene.get("width") or 0),
                height=int(scene.get("height") or 0),
            )
            workflow_artifact = StoredArtifact.from_path(
                workflow_path, project_dir=self.layout.project_dir,
            ) if workflow_path.is_file() else replace(
                manifest.workflow,
                path=workflow_path.resolve().relative_to(self.layout.project_dir.resolve()).as_posix(),
            )
            manifest = replace(
                manifest,
                workflow=replace(workflow_artifact, sha256=manifest.workflow.sha256),
            )
            temporary_manifest = manifest.write(
                _write_json_temp(manifest_path, manifest.to_dict())
            )
            os.replace(temporary_workflow, workflow_path)
            os.replace(temporary_manifest, manifest_path)
        finally:
            temporary_workflow.unlink(missing_ok=True)
            if temporary_manifest is not None:
                temporary_manifest.unlink(missing_ok=True)
        return PreparedSceneWorkflow(scene_number, workflow_path.parent, workflow_path, manifest_path)

    def _manifest_assets(
        self, scene: dict[str, Any], audio_file: Path | None, uploaded_names: dict[Path, str],
    ) -> list[tuple[str, str | Path, str]]:
        assets: list[tuple[str, str | Path, str]] = []
        if audio_file is not None:
            audio_path = self._project_path(audio_file).resolve()
            assets.append(("audio", audio_path, uploaded_names[audio_path]))

        references = scene.get("references") or {}
        candidates: list[tuple[str, str | Path]] = []
        ingredients = scene.get("ingredients_scene_sheet")
        if ingredients:
            candidates.append(("ingredients_sheet", ingredients))
        actor_paths = references.get("actor_msr_paths") or references.get("actor_sheet_paths") or []
        candidates.extend(("actor_sheet", path) for path in actor_paths)
        location = references.get("location_msr_path") or references.get("location_sheet_path")
        if location:
            candidates.append(("location_sheet", location))
        keyframes = scene.get("keyframes") or {}
        startframe = keyframes.get("startframe_path") or keyframes.get("start_frame_path")
        if startframe:
            candidates.append(("startframe", startframe))
        for role, value in candidates:
            path = self._project_path(value).resolve()
            if path in uploaded_names:
                assets.append((role, path, uploaded_names[path]))
        return assets

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.layout.project_dir / path


class PreparedWorkflowRenderer:
    """Verify and queue the exact JSON produced by :class:`WorkflowMaterializer`."""

    def __init__(
        self, *, project_dir: str | Path, render_queue: Any, postprocessor: Any,
        expected_pipeline: str,
    ):
        self.project_dir = Path(project_dir)
        self.render_queue = render_queue
        self.postprocessor = postprocessor
        self.expected_pipeline = expected_pipeline

    def render(self, prepared_workflow_path: str | Path) -> Path:
        workflow_path = Path(prepared_workflow_path)
        manifest = SceneWorkflowManifest.read(workflow_path.with_name("manifest.json"))
        if manifest.pipeline != self.expected_pipeline:
            raise ValueError(
                f"Prepared workflow pipeline {manifest.pipeline!r} does not match "
                f"expected pipeline {self.expected_pipeline!r}"
            )
        manifest_workflow_path = manifest.workflow.resolve(self.project_dir).resolve()
        if workflow_path.resolve() != manifest_workflow_path:
            raise ValueError(
                f"Prepared path {workflow_path} does not match manifest workflow {manifest_workflow_path}"
            )
        mismatches = manifest.verify(self.project_dir)
        if mismatches:
            raise ValueError("Prepared workflow verification failed: " + "; ".join(mismatches))
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        layout = SceneArtifactLayout(self.project_dir)
        raw_path = layout.scene_raw_video(manifest.scene)
        final_path = layout.scene_final_video(manifest.scene)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=raw_path.parent, suffix=".mp4", delete=False) as handle:
            temporary_raw = Path(handle.name)
        try:
            downloaded = self.render_queue.queue_workflow_and_download_first_video(
                workflow, scene_number=manifest.scene, output_path=temporary_raw,
            )
            os.replace(downloaded, raw_path)
        finally:
            temporary_raw.unlink(missing_ok=True)
        with NamedTemporaryFile(dir=final_path.parent, suffix=".mp4", delete=False) as handle:
            temporary_final = Path(handle.name)
        try:
            self.postprocessor.trim_clip(TrimSpec(
                source_file=raw_path,
                output_file=temporary_final,
                fps=manifest.fps,
                trim_front_frames=manifest.trim_front_frames,
                keep_frames=manifest.frame_count,
                scene=manifest.scene,
            ))
            os.replace(temporary_final, final_path)
        finally:
            temporary_final.unlink(missing_ok=True)
        return final_path


def _rolling_value(rolling: Any, key: str) -> Any:
    if isinstance(rolling, dict):
        return rolling.get(key)
    return getattr(rolling, key, None)


class _RecordingUploader:
    def __init__(self, delegate: Any, project_dir: Path):
        self.delegate = delegate
        self.project_dir = project_dir
        self.names: dict[Path, str] = {}

    def resolve_audio_name(self, path: str | Path, **kwargs: Any) -> str:
        name = self.delegate.resolve_audio_name(path, **kwargs)
        self.names[self._path(path)] = name
        return name

    def resolve_reference_image_name(self, path: str | Path, **kwargs: Any) -> str:
        name = self.delegate.resolve_reference_image_name(path, **kwargs)
        self.names[self._path(path)] = name
        return name

    def resolve_startframe_name(self, path: str | Path, **kwargs: Any) -> str:
        name = self.delegate.resolve_startframe_name(path, **kwargs)
        self.names[self._path(path)] = name
        return name

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.project_dir / path).resolve()
