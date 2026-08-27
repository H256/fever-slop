from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.adapters.reporting import ConsoleReporter
from feverslop.application.effective_render_plan import project_effective_plan
from feverslop.domain.artifact_hash import sha256_file
from feverslop.domain.continuity import BoundaryFrameManifest
from feverslop.domain.render_plan import RenderPlan
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.rendering import (
    VideoRenderBackend,
    VideoRenderRequest,
    WorkflowAnchorConfig,
)
from feverslop.ports.reporting import Reporter
from feverslop.utils.io import file_is_valid
from feverslop.utils.io import atomic_write_json, read_json_object


@dataclass(frozen=True)
class RenderVideoScenesRequest:
    render_plan_path: Path
    workflow_path: Path
    audio_file: Path
    storyboard_dir: Path
    output_dir: Path
    render_mode: str = "single_prompt"
    single_prompt_workflow_path: Path | None = None
    limit: int | None = None
    scene_numbers: set[int] | None = None
    skip_existing: bool = True
    uploaded_audio_name: str | None = None
    upload_audio: bool = True
    upload_startframes: bool = True
    anchors: WorkflowAnchorConfig = WorkflowAnchorConfig()
    on_scene_complete: Callable[[Path, int, int], None] | None = None
    canonical_plan_path: Path | None = None


class RenderVideoScenesUseCase:
    def __init__(
        self,
        backend: VideoRenderBackend,
        artifact_store: ArtifactStore,
        console: Any | None = None,
        reporter: Reporter | None = None,
    ):
        self.backend = backend
        self.artifact_store = artifact_store
        self.reporter = reporter or (ConsoleReporter(console) if console is not None else None)

    def execute(self, request: RenderVideoScenesRequest) -> list[Path]:
        render_plan_data = self.artifact_store.read_render_plan(request.render_plan_path)
        canonical_plan_data = (
            self.artifact_store.read_render_plan(request.canonical_plan_path)
            if request.canonical_plan_path is not None
            else None
        )
        plan = RenderPlan.from_dicts(
            project_effective_plan(render_plan_data, canonical_plan_data),
        ).select(
            scene_numbers=request.scene_numbers,
            limit=request.limit,
        )

        rendered: list[Path] = []
        rendered_by_segment: dict[str, Path] = {}
        total = len(plan.scenes)
        for scene in plan.scenes:
            scene_payload = scene.to_dict()
            technical_segment_id = str(
                scene_payload.get("technical_segment_id")
                or scene_payload.get("segment_id")
                or "",
            ).strip()
            predecessor_id = str(
                scene_payload.get("continuation_predecessor_id") or "",
            ).strip()
            predecessor_clip: Path | None = None
            boundary_is_current = False
            if predecessor_id and getattr(self.backend, "pipeline_name", "") == "minimax-h3-r2v":
                predecessor_clip = rendered_by_segment.get(predecessor_id)
                if predecessor_clip is None or not predecessor_clip.is_file():
                    raise ValueError(
                        f"Continuation segment {technical_segment_id or scene.scene_number} "
                        f"requires rendered predecessor {predecessor_id}",
                    )
                boundary_is_current = _continuation_boundary_is_current(
                    scene_number=scene.scene_number,
                    predecessor_clip=predecessor_clip,
                    output_dir=request.output_dir,
                    backend=self.backend,
                )
            if predecessor_id and getattr(self.backend, "pipeline_name", "") == "minimax-h3-r2v":
                assert predecessor_clip is not None
                if boundary_is_current:
                    scene_payload = _restore_r2v_continuation_anchor(
                        scene_payload,
                        predecessor_id=predecessor_id,
                        scene_number=scene.scene_number,
                        output_dir=request.output_dir,
                        backend=self.backend,
                    )
                else:
                    scene_payload = _attach_r2v_continuation_anchor(
                        scene_payload,
                        predecessor_id=predecessor_id,
                        predecessor_clip=predecessor_clip,
                        output_dir=request.output_dir,
                        backend=self.backend,
                    )
                scene = type(scene).from_dict(scene_payload)
            video_request = VideoRenderRequest(
                scene=scene_payload,
                scene_number=scene.scene_number,
                prompt=scene.video_prompt,
                workflow_path=request.workflow_path,
                output_dir=request.output_dir,
                audio_file=request.audio_file,
                storyboard_dir=request.storyboard_dir,
                render_mode=request.render_mode,
                single_prompt_workflow_path=request.single_prompt_workflow_path,
                skip_existing=request.skip_existing,
                uploaded_audio_name=request.uploaded_audio_name,
                upload_audio=request.upload_audio,
                upload_startframes=request.upload_startframes,
                anchors=request.anchors,
                render_plan_path=request.render_plan_path,
            )
            final_path = request.output_dir / "final" / f"scene_{scene.scene_number:04}.mp4"
            direct_path = request.output_dir / f"scene_{scene.scene_number:04}.mp4"
            per_scene_path = request.output_dir / f"scene_{scene.scene_number:04}" / "final.mp4"
            existing_path = (
                final_path if file_is_valid(final_path)
                else (direct_path if file_is_valid(direct_path)
                      else (per_scene_path if file_is_valid(per_scene_path) else None))
            )
            if predecessor_id and not (scene_payload.get("keyframes") or {}).get(
                "boundary_frame_manifest"
            ):
                existing_path = None
            if request.skip_existing and existing_path:
                ensure_manifest = getattr(self.backend, "ensure_scene_manifest", None)
                if ensure_manifest is not None:
                    ensure_manifest(video_request)
                rendered.append(existing_path)
                self._log_scene_available(existing_path, len(rendered), total, skipped=True)
                if request.on_scene_complete:
                    request.on_scene_complete(existing_path, len(rendered), total)
                if technical_segment_id:
                    rendered_by_segment[technical_segment_id] = existing_path
                continue

            randomize_seed = bool(getattr(self.backend, "randomize_seed", False))
            if randomize_seed:
                scene_payload["seed"] = random.SystemRandom().randint(0, 2**63 - 1)
                render_plan_data = [
                    (
                        scene_payload
                        if int(item["scene"]) == scene.scene_number
                        else item
                    )
                    for item in render_plan_data
                ]
                self.artifact_store.write_render_plan(request.render_plan_path, render_plan_data)
                original_randomize_seed = self.backend.randomize_seed
                self.backend.randomize_seed = False
                try:
                    output_path = self.backend.render_video(video_request)
                finally:
                    self.backend.randomize_seed = original_randomize_seed
            else:
                output_path = self.backend.render_video(video_request)
            if predecessor_id and not boundary_is_current:
                boundary_manifest = (scene_payload.get("keyframes") or {}).get(
                    "boundary_frame_manifest"
                )
                if isinstance(boundary_manifest, dict):
                    atomic_write_json(
                        request.output_dir
                        / f"scene_{scene.scene_number:04}"
                        / "continuation_boundary.json",
                        boundary_manifest,
                    )
            rendered.append(output_path)
            if technical_segment_id:
                rendered_by_segment[technical_segment_id] = output_path
            self._log_scene_available(output_path, len(rendered), total, skipped=False)
            if request.on_scene_complete:
                request.on_scene_complete(output_path, len(rendered), total)

        return rendered

    def _log_scene_available(self, output_path: Path, completed: int, total: int, *, skipped: bool) -> None:
        if self.reporter is None:
            return
        verb = "Available" if skipped else "Rendered"
        self.reporter.message(
            f"[green]OK[/green] {verb} scene {completed}/{total}: [cyan]{output_path}[/cyan]",
        )


def _attach_r2v_continuation_anchor(
    scene: dict[str, Any],
    *,
    predecessor_id: str,
    predecessor_clip: Path,
    output_dir: Path,
    backend: Any,
) -> dict[str, Any]:
    """Persist a verified predecessor boundary for the production R2V backend."""
    postprocessor = getattr(backend, "postprocessor", None)
    if postprocessor is None:
        raise ValueError("R2V continuation requires a video postprocessor")
    project_dir = getattr(backend, "project_dir", None)
    source_clip = Path(predecessor_clip).resolve()
    if not source_clip.is_file():
        raise ValueError(f"Missing continuation predecessor clip: {source_clip}")
    previous_scene_number = int(scene["scene"]) - 1
    match = re.search(r"scene_(\d+)", source_clip.parent.name)
    if match:
        previous_scene_number = int(match.group(1))
    frame_path = output_dir / "keyframes" / (
        f"scene_{previous_scene_number:04}_to_{int(scene['scene']):04}_start.png"
    )
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    extracted = Path(postprocessor.extract_last_frame(source_clip, frame_path)).resolve()
    project = Path(project_dir).resolve() if project_dir is not None else None
    source_stored = _stored_continuation_path(source_clip, project)
    frame_stored = _stored_continuation_path(extracted, project)
    manifest = BoundaryFrameManifest.create(
        source_clip_path=source_stored,
        source_clip_sha256=sha256_file(source_clip),
        frame_index=int(getattr(postprocessor, "last_frame_index", 0) or 0),
        extractor_revision="last-frame-v1",
        frame_path=frame_stored,
        frame_sha256=sha256_file(extracted),
    )
    keyframes = dict(scene.get("keyframes") or {})
    keyframes.update({
        "continuity_anchor_path": extracted.as_posix(),
        "startframe_path": extracted.as_posix(),
        "startframe_source_scene": previous_scene_number,
        "startframe_mode": "last_frame_from_previous",
        "startframe_source_clip_path": source_stored,
        "startframe_source_clip_sha256": sha256_file(source_clip),
        "startframe_extractor": "last-frame-v1",
        "startframe_sha256": sha256_file(extracted),
        "boundary_frame_manifest": manifest.to_dict(),
        "continuation_predecessor_id": predecessor_id,
    })
    result = dict(scene)
    result["keyframes"] = keyframes
    return result


def _stored_continuation_path(path: Path, project_dir: Path | None) -> str:
    if project_dir is not None and path.is_relative_to(project_dir):
        return path.relative_to(project_dir).as_posix()
    return path.as_posix()


def _restore_r2v_continuation_anchor(
    scene: dict[str, Any],
    *,
    predecessor_id: str,
    scene_number: int,
    output_dir: Path,
    backend: Any,
) -> dict[str, Any]:
    """Reattach a current persisted boundary to a rerender request."""
    sidecar = output_dir / f"scene_{int(scene_number):04}" / "continuation_boundary.json"
    manifest = BoundaryFrameManifest.from_dict(read_json_object(sidecar))
    project_dir = getattr(backend, "project_dir", None)
    frame_path = Path(manifest.frame_path)
    if project_dir is not None and not frame_path.is_absolute():
        frame_path = Path(project_dir) / frame_path
    frame_path = frame_path.resolve()
    if not frame_path.is_file() or sha256_file(frame_path) != manifest.frame_sha256:
        raise ValueError(f"Persisted continuation anchor is invalid: {frame_path}")

    keyframes = dict(scene.get("keyframes") or {})
    keyframes.update({
        "continuity_anchor_path": frame_path.as_posix(),
        "startframe_path": frame_path.as_posix(),
        "startframe_mode": "last_frame_from_previous",
        "startframe_source_clip_path": manifest.source_clip_path,
        "startframe_source_clip_sha256": manifest.source_clip_sha256,
        "startframe_extractor": manifest.extractor_revision,
        "startframe_sha256": manifest.frame_sha256,
        "boundary_frame_manifest": manifest.to_dict(),
        "continuation_predecessor_id": predecessor_id,
    })
    result = dict(scene)
    result["keyframes"] = keyframes
    return result


def _continuation_boundary_is_current(
    *,
    scene_number: int,
    predecessor_clip: Path,
    output_dir: Path,
    backend: Any,
) -> bool:
    sidecar = output_dir / f"scene_{int(scene_number):04}" / "continuation_boundary.json"
    try:
        manifest = BoundaryFrameManifest.from_dict(read_json_object(sidecar))
        if manifest.source_clip_sha256 != sha256_file(predecessor_clip):
            return False
        project_dir = getattr(backend, "project_dir", None)
        frame = (
            Path(project_dir) / manifest.frame_path
            if project_dir is not None and not Path(manifest.frame_path).is_absolute()
            else Path(manifest.frame_path)
        )
        return frame.is_file() and sha256_file(frame) == manifest.frame_sha256
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
