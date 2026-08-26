from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.minimax_h3_frames import (
    _frames_from_duration as _frames_from_duration_,
)
from feverslop.domain.postprocessing import TrimSpec
from feverslop.domain.prepared_workflow import SceneWorkflowManifest, StoredArtifact
from feverslop.errors import FeverSlopValidationError
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIMiniMaxH3VideoRenderBackend:
    """Base render backend for MiniMax H3 video generation on ComfyUI."""

    MIN_DIMENSION = 512
    MAX_DIMENSION = 2048

    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        preroll_frames: int = 0,
        tail_loss_frames: int = 0,
        postprocess: bool = True,
        ffmpeg_path: str = "ffmpeg",
        postprocess_reencode: bool = True,
        ffmpeg_debug: bool = False,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        postprocessor: VideoPostProcessor | None = None,
        model_resolver=None,
        video_settings: VideoSettings | None = None,
        project_dir: str | Path | None = None,
        workflow: dict | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.workflow = deepcopy(workflow) if workflow is not None else None
        self.output_dir = Path(output_dir)
        self.raw_output_dir = self.output_dir / "raw"
        self.project_dir = Path(project_dir) if project_dir is not None else None
        self.preroll_frames = max(0, int(preroll_frames))
        self.tail_loss_frames = max(0, int(tail_loss_frames))
        self.postprocess = bool(postprocess)
        self.postprocess_reencode = bool(postprocess_reencode)
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.postprocessor = postprocessor or VideoPostProcessor(
            ffmpeg_path=ffmpeg_path,
            reencode=postprocess_reencode,
            debug=ffmpeg_debug,
        )
        self.model_resolver = model_resolver
        self.video_settings = video_settings
        self.progress_callback = progress_callback

    def _progress(self, stage: str) -> None:
        callback = self.progress_callback
        if callback is not None:
            callback(str(stage))

    def load_workflow(self) -> dict:
        """Load the ComfyUI workflow JSON (in-memory or from file)."""
        if self.workflow is not None:
            return deepcopy(self.workflow)
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _frames_from_duration(seconds: float) -> int:
        """Convert seconds to a 17N+5-constrained frame count at 24 fps."""
        return _frames_from_duration_(seconds)

    @staticmethod
    def _validate_resolution(width: int, height: int) -> None:
        """Raise ``FeverSlopValidationError`` if resolution is outside allowed range."""
        if width < ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION or width > ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION:
            raise FeverSlopValidationError(
                f"Width {width} is outside allowed range "
                f"[{ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION}, "
                f"{ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION}]",
            )
        if height < ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION or height > ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION:
            raise FeverSlopValidationError(
                f"Height {height} is outside allowed range "
                f"[{ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION}, "
                f"{ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION}]",
            )

    @staticmethod
    def _patch_minimax_core(
        patcher: WorkflowPatcher,
        prompt: str,
        width: int,
        height: int,
        frames: int,
    ) -> None:
        """Set prompt, resolution, and frame count on the MiniMax H3 core node."""
        for target_class in ("MiniMaxH3Video", "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo"):
            try:
                _, node = patcher.find_nodes_by_class_type(target_class)[0]
                node.setdefault("inputs", {})["prompt"] = prompt
                node.setdefault("inputs", {})["width"] = width
                node.setdefault("inputs", {})["height"] = height
                node.setdefault("inputs", {})["length"] = frames
                return
            except IndexError:
                continue
        raise KeyError(
            "Workflow has neither MiniMaxH3Video nor MiniMaxH3ReferenceToVideo "
            "nor MiniMaxH3ImageToVideo node",
        )

    @staticmethod
    def _patch_megapixels(
        patcher: WorkflowPatcher,
        megapixels: float,
        *,
        explicit: bool = False,
    ) -> None:
        """Patch the megapixels anchor (#MEGAPIXELS or #MEGAPIXEL).

        Tries the plural anchor first (R2V convention), then falls back to
        the singular anchor (T2V convention).
        """
        value = float(megapixels) if explicit else math.floor(float(megapixels) * 10) / 10
        if patcher.try_set_existing_input_by_title("#MEGAPIXELS", "megapixels", value):
            return
        if patcher.try_set_existing_input_by_title("#MEGAPIXEL", "megapixels", value):
            return
        raise KeyError("Workflow has neither #MEGAPIXELS nor #MEGAPIXEL anchor")

    @staticmethod
    def _patch_seed(patcher: WorkflowPatcher, seed: int) -> None:
        """Patch the #SEED anchor with the given seed value.

        Tries noise_seed, seed, and value inputs in preference order.
        """
        for input_name in ("noise_seed", "seed", "value"):
            if patcher.try_set_existing_input_by_title("#SEED", input_name, seed):
                return
        patcher.set_input_by_title("#SEED", "noise_seed", seed)

    @staticmethod
    def _patch_save_video(
        patcher: WorkflowPatcher,
        scene_number: int,
    ) -> None:
        """Patch the #SAVE_VIDEO anchor with a deterministic filename prefix.

        Uses a per-scene subfolder so the ComfyUI output lands under the canonical
        artifact layout: render/scenes/scene_NNNN/raw/00001.mp4.
        """
        patcher.set_input_by_title(
            "#SAVE_VIDEO",
            "filename_prefix",
            f"scene_{scene_number:04}/raw",
        )

    @staticmethod
    def _decode_packed_latent(patcher: WorkflowPatcher) -> None:
        """Validate that both VAEDecode and VAEDecodeAudio nodes exist in the workflow.

        No-op when both are present; subclasses handle edge cases where one is missing.
        """
        has_video = bool(patcher.find_nodes_by_class_type("VAEDecode"))
        has_audio = bool(patcher.find_nodes_by_class_type("VAEDecodeAudio"))
        if not has_video:
            raise KeyError("Workflow is missing VAEDecode node")
        if not has_audio:
            raise KeyError("Workflow is missing VAEDecodeAudio node")

    def _postprocess_with_audio(self, raw_output: Path, spec: TrimSpec) -> Path:
        """Trim raw output using ffmpeg via the postprocessor.

        H3 output is a single MP4 with synced audio+video; ffmpeg trims both streams.
        """
        output = self.postprocessor.trim_clip(spec)
        if spec.extract_boundary_frames and self.project_dir is not None:
            self._persist_boundary_frames(spec.scene)
        return output

    def _persist_boundary_frames(self, scene_number: int) -> None:
        manifest_path = self.output_dir / f"scene_{scene_number:04}" / "manifest.json"
        if not manifest_path.is_file() or self.project_dir is None:
            return
        scene_dir = self.output_dir / f"scene_{scene_number:04}"
        first_frame_path = scene_dir / "firstframe.png"
        last_frame_path = scene_dir / "lastframe.png"
        if not first_frame_path.is_file() or not last_frame_path.is_file():
            return
        manifest = SceneWorkflowManifest.read(manifest_path)
        manifest = replace(
            manifest,
            first_frame_path=(
                StoredArtifact.from_path(first_frame_path, project_dir=self.project_dir)
                if manifest.first_frame_path is None else manifest.first_frame_path
            ),
            last_frame_path=(
                StoredArtifact.from_path(last_frame_path, project_dir=self.project_dir)
                if manifest.last_frame_path is None else manifest.last_frame_path
            ),
        )
        manifest.write(manifest_path)

    def _write_scene_manifest(
        self,
        request: VideoRenderRequest,
        workflow_path: Path,
        *,
        pipeline: str,
        workflow: dict,
        assets: list[tuple],
    ) -> None:
        if self.project_dir is None or request.render_plan_path is None:
            return
        scene = request.scene
        SceneWorkflowManifest.create(
            project_dir=self.project_dir,
            scene=request.scene_number,
            pipeline=pipeline,
            workflow_path=workflow_path,
            template_path=self.workflow_path,
            render_plan_path=request.render_plan_path,
            assets=assets,
            seed=self._workflow_seed(workflow),
            fps=int(scene.get("fps") or 24),
            frame_count=int(scene.get("frame_count") or 0),
            width=int(scene.get("width") or 0),
            height=int(scene.get("height") or 0),
        ).write(workflow_path.with_name("manifest.json"))

    def ensure_scene_manifest(self, request: VideoRenderRequest) -> None:
        scene_dir = self.output_dir / f"scene_{request.scene_number:04}"
        workflow_path = scene_dir / "workflow.json"
        manifest_path = scene_dir / "manifest.json"
        if manifest_path.is_file() or not workflow_path.is_file():
            return
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        self._write_scene_manifest(
            request,
            workflow_path,
            pipeline=self.pipeline_name,
            workflow=workflow,
            assets=self._manifest_assets(request),
        )

    def _manifest_assets(self, request: VideoRenderRequest) -> list[tuple]:
        return []

    @staticmethod
    def _workflow_seed(workflow: dict) -> int:
        _, seed_node = WorkflowPatcher(workflow).find_node_by_meta_title("#SEED")
        inputs = seed_node.get("inputs", {})
        for name in ("noise_seed", "seed", "value"):
            if name in inputs:
                return int(inputs[name])
        raise ValueError("MiniMax workflow #SEED node has no seed input")

    @staticmethod
    def _reference_asset(role: str, path: str | Path) -> tuple:
        source = Path(path)
        comfy_name = (
            "feverslop/references/"
            + ComfyUIVideoAssetUploader.content_addressed_name(source)
        )
        return role, source, comfy_name

    def build_workflow(self, scene: dict, **kwargs) -> dict:
        """Build a patched workflow dict for rendering.

        Protocol: build_workflow(self, scene: dict, **kwargs) -> dict.
        Subclasses must implement.
        """
        raise NotImplementedError("build_workflow must be implemented by subclass")

    def render_video(self, scene: dict, **kwargs) -> Path:
        """Render one video scene.

        Subclasses must implement to satisfy the VideoRenderBackend protocol.
        """
        raise NotImplementedError("render_video must be implemented by subclass")
