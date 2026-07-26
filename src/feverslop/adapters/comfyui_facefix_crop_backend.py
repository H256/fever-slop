from __future__ import annotations

from pathlib import Path
import json
import shutil

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.facefix_rendering import FaceFixConfig


class ComfyUIFaceFixCropBackend:
    """Renders face-repaired video via the LTXV FaceFix Crop ComfyUI workflow.

    Takes face-crop MP4 + anchor PNGs, patches the crop workflow, and runs
    the LTXV LoopingSampler with anchor conditioning. CLIP runs on CPU.
    """

    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path,
        config: FaceFixConfig | None = None,
        output_dir: str | Path | None = None,
        postprocess: bool = True,
        ffmpeg_path: str = "ffmpeg",
        postprocess_reencode: bool = True,
        ffmpeg_debug: bool = False,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        postprocessor: VideoPostProcessor | None = None,
        model_resolver=None,
        project_dir: str | Path | None = None,
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
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
        )
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    def load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def render_scene(
        self,
        scene_number: int,
        *,
        face_crop_mp4: Path,
        anchors_dir: Path,
        output_dir: Path,
        actor_id: str,
    ) -> Path:
        workflow = self.build_workflow(scene_number, face_crop_mp4=face_crop_mp4, anchors_dir=anchors_dir, actor_id=actor_id)
        workflow = self.model_resolver.resolve_workflow_models(
            workflow,
            workflow_path=self.workflow_path,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = output_dir / "workflow_facefix_crop.json"
        workflow_path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        raw_output = self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=output_dir / "raw_facefix_crop.mp4",
        )

        if not self.postprocess:
            return raw_output

        final = output_dir / f"repaired_{actor_id}.mp4"
        shutil.copy2(raw_output, final)
        return final

    def build_workflow(
        self,
        scene_number: int,
        *,
        face_crop_mp4: Path,
        anchors_dir: Path,
        actor_id: str,
    ) -> dict:
        patcher = WorkflowPatcher(self.load_workflow())
        self._patch_video_input(patcher, face_crop_mp4, scene_number, actor_id)
        self._patch_anchors_dir(patcher, anchors_dir)
        self._patch_facefix_params(patcher)
        self._patch_save_output(patcher, scene_number, actor_id)
        return patcher.get()

    def _patch_video_input(self, patcher: WorkflowPatcher, face_crop_mp4: Path, scene_number: int, actor_id: str) -> None:
        video_name = self._upload_video(face_crop_mp4, scene_number, actor_id)
        try:
            patcher.set_input_by_title("#LOAD_VIDEO", "video", video_name)
        except KeyError:
            patcher.set_input_by_title("#LOAD_VIDEO", "videopath", video_name)

    def _patch_anchors_dir(self, patcher: WorkflowPatcher, anchors_dir: Path) -> None:
        try:
            anchor_count = len(list(anchors_dir.glob("*.png")))
        except OSError:
            anchor_count = 0

        comfy_subfolder = self._upload_anchors(anchors_dir)
        try:
            patcher.set_input_by_title("#FACE_REFS", "directory", comfy_subfolder)
        except KeyError:
            pass

        if anchor_count > 0:
            indices = []
            for i in range(0, anchor_count, 16):
                indices.append(str(i))
            keyframes = ",".join(indices) if indices else "0"
            patcher.try_set_existing_input_by_title(
                "#LOOPING_SAMPLER", "optional_cond_image_indices", keyframes
            )

    def _patch_facefix_params(self, patcher: WorkflowPatcher) -> None:
        cfg = self.config
        patcher.try_set_existing_input_by_title("#LOOPING_SAMPLER", "guiding_strength", cfg.guiding_strength)
        patcher.try_set_existing_input_by_title("#LOOPING_SAMPLER", "cond_image_strength", cfg.cond_image_strength)
        patcher.try_set_existing_input_by_title("#LOOPING_SAMPLER", "temporal_tile_size", cfg.temporal_tile_size)
        patcher.try_set_existing_input_by_title("#LOOPING_SAMPLER", "temporal_overlap", cfg.temporal_overlap)
        patcher.try_set_existing_input_by_title(
            "#LOOPING_SAMPLER", "temporal_overlap_cond_strength", cfg.temporal_overlap_cond_strength
        )

    def _patch_save_output(self, patcher: WorkflowPatcher, scene_number: int, actor_id: str) -> None:
        patcher.set_input_by_title(
            "#SAVE_VIDEO", "filename_prefix",
            f"ltx_facefix_crop/scene_{scene_number:04}_{actor_id}",
        )

    def _upload_video(self, video_path: Path, scene_number: int, actor_id: str) -> str:
        upload_resp = self.client.upload_file_via_image_endpoint(
            video_path,
            subfolder=f"feverslop/facefix_crop/scene_{scene_number:04}/{actor_id}",
            file_type="input",
            overwrite=True,
        )
        return ComfyUIVideoAssetUploader.comfy_path_from_upload(upload_resp)

    def _upload_anchors(self, anchors_dir: Path) -> str:
        target_subfolder = f"feverslop/facefix_crop/anchors/{anchors_dir.name}"
        for png in sorted(anchors_dir.glob("*.png")):
            self.client.upload_image(
                png,
                subfolder=target_subfolder,
                file_type="input",
                overwrite=True,
            )
        return target_subfolder
