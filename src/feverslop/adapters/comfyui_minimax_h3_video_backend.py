from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.minimax_h3_frames import _frames_from_duration as _frames_from_duration_
from feverslop.domain.postprocessing import TrimSpec
from feverslop.errors import FeverSlopValidationError
from feverslop.config.video_settings import VideoSettings


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

    def load_workflow(self) -> dict:
        """Load the ComfyUI workflow JSON (in-memory or from file)."""
        if self.workflow is not None:
            return deepcopy(self.workflow)
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _frames_from_duration(seconds: float) -> int:
        """Convert seconds to a 17N+1-constrained frame count at 24 fps."""
        return _frames_from_duration_(seconds)

    @staticmethod
    def _validate_resolution(width: int, height: int) -> None:
        """Raise ``FeverSlopValidationError`` if resolution is outside allowed range."""
        if width < ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION or width > ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION:
            raise FeverSlopValidationError(
                f"Width {width} is outside allowed range "
                f"[{ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION}, "
                f"{ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION}]"
            )
        if height < ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION or height > ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION:
            raise FeverSlopValidationError(
                f"Height {height} is outside allowed range "
                f"[{ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION}, "
                f"{ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION}]"
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
            "nor MiniMaxH3ImageToVideo node"
        )

    @staticmethod
    def _patch_megapixels(
        patcher: WorkflowPatcher,
        megapixels: float,
    ) -> None:
        """Patch the megapixels anchor (#MEGAPIXELS or #MEGAPIXEL).

        Tries the plural anchor first (R2V convention), then falls back to
        the singular anchor (T2V convention).
        """
        rounded = round(megapixels, 1)
        if patcher.try_set_existing_input_by_title("#MEGAPIXELS", "megapixels", rounded):
            return
        if patcher.try_set_existing_input_by_title("#MEGAPIXEL", "megapixels", rounded):
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
        return self.postprocessor.trim_clip(spec)

    def build_workflow(self, scene: dict, **kwargs) -> dict:
        """Build a patched workflow dict for rendering.

        Protocol: build_workflow(self, scene: dict, **kwargs) -> dict.
        Subclasses must implement."""
        raise NotImplementedError("build_workflow must be implemented by subclass")

    def render_video(self, scene: dict, **kwargs) -> Path:
        """Render one video scene.

        Subclasses must implement to satisfy the VideoRenderBackend protocol."""
        raise NotImplementedError("render_video must be implemented by subclass")
