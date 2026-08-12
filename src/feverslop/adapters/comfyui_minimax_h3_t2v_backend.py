from __future__ import annotations

from pathlib import Path
import json
import random

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_minimax_h3_video_backend import ComfyUIMiniMaxH3VideoRenderBackend
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.postprocessing import TrimSpec
from feverslop.path_utils import coerce_local_path
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIMiniMaxH3T2VBackend(ComfyUIMiniMaxH3VideoRenderBackend):
    """MiniMax H3 text-to-video backend using FL2VA + FeverSlop meta-anchors.

    Subclass of ComfyUIMiniMaxH3VideoRenderBackend. Patches #PROMPT, #SEED,
    #FRAMECOUNT, #MEGAPIXEL, #SAVE_VIDEO, #T2V_START, #T2V_END, #T2V_TEXT.
    """

    MAX_FRAMES = 2  # start + end
    FPS = 24
    pipeline_name = "minimax-h3-t2v"

    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        seed_offset: int = 100000,
        randomize_seed: bool = False,
        debug_workflows_dir: str | Path | None = None,
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
        video_settings=None,
        project_dir: str | Path | None = None,
        workflow: dict | None = None,
        workflow_label: str | Path | None = None,
    ):
        super().__init__(
            client=client,
            workflow_path=workflow_path,
            output_dir=output_dir,
            preroll_frames=preroll_frames,
            tail_loss_frames=tail_loss_frames,
            postprocess=postprocess,
            ffmpeg_path=ffmpeg_path,
            postprocess_reencode=postprocess_reencode,
            ffmpeg_debug=ffmpeg_debug,
            asset_uploader=asset_uploader,
            render_queue=render_queue,
            postprocessor=postprocessor,
            model_resolver=model_resolver,
            video_settings=video_settings,
            project_dir=project_dir,
            workflow=workflow,
        )
        self.seed_offset = int(seed_offset)
        self.randomize_seed = bool(randomize_seed)
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None
        self.workflow_label = Path(workflow_label) if workflow_label is not None else self.workflow_path
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    # -----------------------------------------------------------------------
    # High-level entry points
    # -----------------------------------------------------------------------

    def build_workflow(
        self,
        scene: dict,
        *,
        prompt: str,
        duration_seconds: float | None = None,
        width: int | None = None,
        height: int | None = None,
        start_frame_path: str | Path | None = None,
        end_frame_path: str | Path | None = None,
        text_data: str | None = None,
    ) -> dict:
        """Build a patched T2V workflow dict from *scene*.

        Patches FeverSlop meta-anchors:
        - ``#PROMPT``  -> ``prompt``
        - ``#SEED``    -> ``noise_seed``
        - ``#FRAMECOUNT`` -> ``value``
        - ``#MEGAPIXEL`` -> ``megapixels`` (computed from width x height when given)
        - ``#T2V_START`` -> ``image``
        - ``#T2V_END`` -> ``image``
        - ``#T2V_TEXT`` -> ``value``
        - ``#SAVE_VIDEO`` -> ``filename_prefix``
        """
        self._validate_scene(scene)
        scene_number = int(scene.get("scene", 0))

        patcher = WorkflowPatcher(self.load_workflow())

        # -- core patching
        patcher.set_input_by_title("#PROMPT", "prompt", str(prompt).strip())
        self._patch_seed(patcher, self._seed_for_scene(scene))
        self._patch_save_video(patcher, scene_number)

        # -- optional: megapixels
        if width is not None and height is not None:
            megapixels = (int(width) * int(height)) / 1_000_000
            self._patch_megapixels(patcher, megapixels)

        # -- optional: frame count
        if duration_seconds is not None:
            patcher.set_input_by_title(
                "#FRAMECOUNT", "value", int(round(float(duration_seconds) * 24))
            )

        # -- optional: start/end frames
        if start_frame_path is not None:
            image_name = self.asset_uploader.resolve_reference_image_name(start_frame_path)
            self._patch_t2v_start(patcher, image_name)

        if end_frame_path is not None:
            image_name = self.asset_uploader.resolve_reference_image_name(end_frame_path)
            self._patch_t2v_end(patcher, image_name)

        # -- optional: text_data
        if text_data is not None:
            self._patch_t2v_text(patcher, text_data)

        return patcher.get()

    def render_video(self, request: VideoRenderRequest) -> Path:
        """Complete render flow for one T2V scene."""
        self._validate_scene(request.scene)
        scene_number = int(request.scene_number)

        # -- compute duration
        duration_seconds: float | None = None
        raw_duration = request.scene.get("duration_seconds")
        if raw_duration is not None:
            duration_seconds = float(raw_duration)

        # -- resolve start/end frame paths from scene
        start_frame_path = self._resolve_start_frame(request.scene)
        end_frame_path = self._resolve_end_frame(request.scene)

        # -- build workflow
        workflow = self.build_workflow(
            request.scene,
            prompt=request.prompt,
            duration_seconds=duration_seconds,
            width=int(request.scene.get("width", 0) or 0) or None,
            height=int(request.scene.get("height", 0) or 0) or None,
            start_frame_path=start_frame_path,
            end_frame_path=end_frame_path,
            text_data=request.scene.get("keyframe_text"),
        )

        # -- resolve model references
        workflow = self.model_resolver.resolve_workflow_models(
            workflow,
            workflow_path=self.workflow_label,
        )

        # -- per-scene output directory
        scene_dir = self.output_dir / f"scene_{scene_number:04}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # -- scene workflow.json (production artifact)
        workflow_path = scene_dir / "workflow.json"
        workflow_path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_scene_manifest(
            request,
            workflow_path,
            pipeline=self.pipeline_name,
            workflow=workflow,
            assets=self._manifest_assets(request),
        )

        # -- debug write
        self._write_debug_workflow(scene_number, workflow)

        # -- queue and download
        raw_output = self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=scene_dir / "raw.mp4",
        )

        if not self.postprocess:
            return raw_output

        # -- postprocess trim: use render plan frame_count for audio sync,
        # fall back to 17N+5 rounding for backward compat
        scene_frame_count = request.scene.get("frame_count")
        if scene_frame_count:
            keep_frames = int(scene_frame_count)
        else:
            keep_frames = self._frames_from_duration(
                duration_seconds if duration_seconds else 5.0
            )
        return self._postprocess_with_audio(
            raw_output,
            TrimSpec(
                source_file=raw_output,
                output_file=scene_dir / "final.mp4",
                fps=self.FPS,
                trim_front_frames=int(self.preroll_frames),
                keep_frames=keep_frames,
                scene=scene_number,
            ),
        )

    # -----------------------------------------------------------------------
    # T2V-specific patching helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _patch_t2v_start(patcher: WorkflowPatcher, image_name: str) -> None:
        """Patch the #T2V_START anchor with an image name."""
        patcher.set_input_by_title("#T2V_START", "image", image_name)

    @staticmethod
    def _patch_t2v_end(patcher: WorkflowPatcher, image_name: str) -> None:
        """Patch the #T2V_END anchor with an image name."""
        patcher.set_input_by_title("#T2V_END", "image", image_name)

    @staticmethod
    def _patch_t2v_text(patcher: WorkflowPatcher, text: str) -> None:
        """Patch the #T2V_TEXT anchor with first/last frame description."""
        patcher.set_input_by_title("#T2V_TEXT", "value", text)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_scene(self, scene: dict) -> None:
        """T2V has no actor reference requirement."""
        pass  # Always valid -- T2V is text-driven

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _seed_for_scene(self, scene: int | dict) -> int:
        if self.randomize_seed:
            return random.randint(0, 2**63 - 1)
        if isinstance(scene, dict) and scene.get("seed") is not None:
            return int(scene["seed"])
        scene_number = int(scene.get("scene", 0)) if isinstance(scene, dict) else int(scene)
        return self.seed_offset + int(scene_number)

    @staticmethod
    def _resolve_project_path(path: str | Path, project_dir: Path | None = None) -> Path:
        return coerce_local_path(path, base_dir=project_dir) if project_dir else coerce_local_path(path)

    def _write_debug_workflow(self, scene_number: int, workflow: dict) -> None:
        if self.debug_workflows_dir is None:
            return
        self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
        (
            self.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json"
        ).write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _resolve_start_frame(self, scene: dict) -> str | Path | None:
        keyframes = scene.get("keyframes") or {}
        path = keyframes.get("startframe_path")
        if path is None:
            return None
        if self.project_dir is not None:
            return self._resolve_project_path(path, self.project_dir)
        return coerce_local_path(path)

    def _resolve_end_frame(self, scene: dict) -> str | Path | None:
        keyframes = scene.get("keyframes") or {}
        path = keyframes.get("endframe_path")
        if path is None:
            return None
        if self.project_dir is not None:
            return self._resolve_project_path(path, self.project_dir)
        return coerce_local_path(path)

    def _manifest_assets(self, request: VideoRenderRequest) -> list[tuple]:
        assets: list[tuple] = []
        if start := self._resolve_start_frame(request.scene):
            assets.append(self._reference_asset("startframe", start))
        if end := self._resolve_end_frame(request.scene):
            assets.append(self._reference_asset("endframe", end))
        return assets
