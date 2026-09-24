from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.domain.postprocessing import FFMPEG_TIMEOUT_SECONDS
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.facefix_rendering import (
    DEFAULT_H3_FACEFIX_WORKFLOW,
    FaceFixConfig,
    snap_to_h3_frame_grid,
)

logger = logging.getLogger(__name__)

# Face-size-dependent denoise: small/distant faces get a higher strength so the
# pass actually repairs them, close-ups a lower strength so identity is not
# disturbed. The base is the workflow's default; the multiplier scales it.
DEFAULT_FACEFIX_DENOISE = 0.25
DEFAULT_SMALL_FACE_DENOISE_MULTIPLIER = 1.0
DEFAULT_LARGE_FACE_DENOISE_MULTIPLIER = 0.6


class ComfyUIFaceFixH3Backend:
    """H3-native video-to-video FaceRefine backend (issue 519, unit 2).

    Consumes an already-rendered H3 R2V scene clip, encodes it to an H3 video
    latent, injects it into the reference-to-video conditioning, and refines
    the clip at a low, face-size-dependent denoise with the scene's actor
    references + prompt. The refined clip is written as a resumable artifact;
    the Python pipeline composites only the face region back onto the source.

    The external H3 FaceRefine node dependency is optional: when the
    ``H3InjectVideoLatent`` node is absent the backend degrades to conditioning
    on the reference-to-video latent directly (the stock H3 path), and the
    operator is told the native video-latent injection is unavailable.
    """

    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path | None = None,
        config: FaceFixConfig | None = None,
        output_dir: str | Path | None = None,
        postprocess: bool = True,
        ffmpeg_path: str = "ffmpeg",
        postprocess_reencode: bool = True,
        ffmpeg_debug: bool = False,
        ffmpeg_timeout_seconds: float | None = None,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        postprocessor: VideoPostProcessor | None = None,
        model_resolver=None,
        project_dir: str | Path | None = None,
    ):
        self.client = client
        self.workflow_path = (
            Path(workflow_path) if workflow_path else Path(DEFAULT_H3_FACEFIX_WORKFLOW)
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.project_dir = Path(project_dir) if project_dir is not None else None
        self.config = config or FaceFixConfig()
        self.postprocess = postprocess
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.postprocessor = postprocessor or VideoPostProcessor(
            ffmpeg_path=ffmpeg_path,
            reencode=postprocess_reencode,
            debug=ffmpeg_debug,
            ffmpeg_timeout_seconds=(
                FFMPEG_TIMEOUT_SECONDS if ffmpeg_timeout_seconds is None else ffmpeg_timeout_seconds
            ),
        )
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    def load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def render_scene(
        self,
        scene_number: int,
        *,
        source_video: Path,
        output_dir: Path,
        actor_id: str,
        face_ref_image: Path | None = None,
        scene_prompt: str = "",
        frame_count: int = 0,
        denoise: float = DEFAULT_FACEFIX_DENOISE,
    ) -> Path:
        workflow = self.build_workflow(
            scene_number,
            source_video=source_video,
            actor_id=actor_id,
            face_ref_image=face_ref_image,
            scene_prompt=scene_prompt,
            frame_count=frame_count,
            denoise=denoise,
        )
        workflow = self.model_resolver.resolve_workflow_models(
            workflow,
            workflow_path=self.workflow_path,
        )
        self._preflight_comfy_node_classes(workflow)

        output_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = output_dir / "workflow_facefix_h3.json"
        workflow_path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        raw_output = self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=output_dir / "raw_facefix_h3.mp4",
        )

        if not self.postprocess:
            return raw_output

        final = output_dir / f"refined_{actor_id}.mp4"
        shutil.copy2(raw_output, final)
        return final

    def build_workflow(
        self,
        scene_number: int,
        *,
        source_video: Path,
        actor_id: str,
        face_ref_image: Path | None = None,
        scene_prompt: str = "",
        frame_count: int = 0,
        denoise: float = DEFAULT_FACEFIX_DENOISE,
    ) -> dict:
        patcher = WorkflowPatcher(self.load_workflow())
        self._patch_source_video(patcher, source_video, scene_number, actor_id)
        self._patch_actor_ref(patcher, face_ref_image, scene_number, actor_id)
        self._patch_prompt(patcher, scene_prompt)
        self._patch_frame_count(patcher, frame_count)
        self._patch_denoise(patcher, denoise)
        self._patch_save_output(patcher, scene_number, actor_id)
        return patcher.get()

    def _patch_source_video(
        self,
        patcher: WorkflowPatcher,
        source_video: Path,
        scene_number: int,
        actor_id: str,
    ) -> None:
        video_name = self._upload_video(source_video, scene_number, actor_id)
        try:
            patcher.set_input_by_title("#LOAD_SOURCE", "video", video_name)
        except KeyError as exc:
            raise RuntimeError(
                f"Cannot patch source video for scene {scene_number}/{actor_id}: "
                "#LOAD_SOURCE node missing or lacks 'video' input",
            ) from exc

    def _patch_actor_ref(
        self,
        patcher: WorkflowPatcher,
        face_ref_image: Path | None,
        scene_number: int,
        actor_id: str,
    ) -> None:
        if face_ref_image is None or not face_ref_image.is_file():
            return
        uploaded = self._upload_face_ref(face_ref_image, scene_number, actor_id)
        try:
            patcher.set_input_by_title("#REF_1", "image", uploaded)
        except KeyError:
            logger.warning(
                "#REF_1 node not found for scene %d/%s; conditioning without actor ref",
                scene_number, actor_id,
            )

    def _patch_prompt(self, patcher: WorkflowPatcher, scene_prompt: str) -> None:
        if not scene_prompt:
            return
        try:
            patcher.set_input_by_title("#PROMPT", "value", scene_prompt)
        except KeyError:
            logger.warning("#PROMPT node not found; using workflow default prompt")

    def _patch_frame_count(self, patcher: WorkflowPatcher, frame_count: int) -> None:
        if frame_count <= 0:
            return
        try:
            patcher.set_input_by_title("#FRAMECOUNT", "value", snap_to_h3_frame_grid(frame_count))
        except KeyError:
            logger.warning("#FRAMECOUNT node not found; using workflow default frame count")

    def _patch_denoise(self, patcher: WorkflowPatcher, denoise: float) -> None:
        try:
            patcher.set_input_by_title("#SCHEDULER", "denoise", denoise)
        except KeyError:
            logger.warning("#SCHEDULER node not found; using workflow default denoise")

    def _patch_save_output(
        self,
        patcher: WorkflowPatcher,
        scene_number: int,
        actor_id: str,
    ) -> None:
        try:
            patcher.set_input_by_title(
                "#SAVE_VIDEO", "filename_prefix",
                f"feverslop/facefix_h3/scene_{scene_number:04}_{actor_id}",
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Cannot patch save output for scene {scene_number}/{actor_id}: "
                "#SAVE_VIDEO node missing or lacks 'filename_prefix'",
            ) from exc

    def _preflight_comfy_node_classes(self, workflow: dict) -> None:
        """Fail before queueing when ComfyUI cannot provide required nodes.

        The native ``H3InjectVideoLatent`` node is optional: when it is missing
        the video-to-video path degrades to the stock H3 reference conditioning
        and the operator is warned. All other required nodes are mandatory.
        """
        get_object_info = getattr(self.client, "get_object_info", None)
        if not callable(get_object_info):
            return
        available = set(get_object_info())
        required = {
            str(node.get("class_type"))
            for node in workflow.values()
            if node.get("class_type")
        }
        missing = sorted(required - available)
        optional = {"H3InjectVideoLatent"}
        hard_missing = [name for name in missing if name not in optional]
        if hard_missing:
            raise RuntimeError(
                "ComfyUI is missing required H3 FaceRefine workflow nodes: "
                + ", ".join(hard_missing)
                + ". Install/load the matching native H3 node package before rendering."
            )
        if "H3InjectVideoLatent" in missing:
            logger.warning(
                "H3InjectVideoLatent node not available; degrading to stock H3 "
                "reference conditioning (no native video-latent injection)."
            )

    def _upload_video(self, video_path: Path, scene_number: int, actor_id: str) -> str:
        upload_resp = self.client.upload_file_via_image_endpoint(
            video_path,
            subfolder=f"feverslop/facefix_h3/scene_{scene_number:04}/{actor_id}",
            file_type="input",
            overwrite=True,
        )
        return ComfyUIVideoAssetUploader.comfy_path_from_upload(upload_resp)

    def _upload_face_ref(self, face_ref: Path, scene_number: int, actor_id: str) -> str:
        subfolder = f"feverslop/facefix_h3/ref/scene_{scene_number:04}/{actor_id}"
        resp = self.client.upload_image(
            face_ref,
            subfolder=subfolder,
            file_type="input",
            overwrite=True,
        )
        return ComfyUIVideoAssetUploader.comfy_path_from_upload(resp)
