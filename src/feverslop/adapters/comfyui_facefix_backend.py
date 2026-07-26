"""DEPRECATED: Use comfyui_facefix_crop_backend.py for the crop-and-composite approach.

This backend feeds full-res videos to LTX with actor sheets as conditioning,
which causes OOM on 32GB VRAM. The replacement uses InsightFace-based face
crops at 768x768 with feather-composite postprocessing.
"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import shutil

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.facefix_rendering import FaceFixConfig, FaceFixSceneRequest


class ComfyUIFaceFixRenderBackend:
    """Renders face-refined video via the LTXV FaceFix ComfyUI workflow.

    This adapter takes already-rendered scene videos, uploads them and optional
    face reference images to ComfyUI, patches the FaceFix workflow, and runs
    the LTXV LoopingSampler with face conditioning.
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
        workflow: dict | None = None,
        workflow_label: str | Path | None = None,
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.workflow = deepcopy(workflow) if workflow is not None else None
        self.workflow_label = Path(workflow_label) if workflow_label is not None else self.workflow_path
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.project_dir = Path(project_dir) if project_dir is not None else None
        self.config = config or FaceFixConfig()
        self.postprocess = postprocess
        self.postprocess_reencode = postprocess_reencode
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.postprocessor = postprocessor or VideoPostProcessor(
            ffmpeg_path=ffmpeg_path,
            reencode=postprocess_reencode,
            debug=ffmpeg_debug,
        )
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()
        self._face_batch_size = 0

    def load_workflow(self) -> dict:
        if self.workflow is not None:
            return deepcopy(self.workflow)
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def render_scene(self, scene: dict, *, request: FaceFixSceneRequest) -> Path:
        scene_number = request.scene_number
        scene_dir = request.output_dir
        workflow = self.build_workflow(scene, request=request)
        workflow = self.model_resolver.resolve_workflow_models(
            workflow,
            workflow_path=self.workflow_label,
        )
        self._write_workflow_json(scene_dir, scene_number, workflow)

        raw_output = self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=scene_dir / "raw_facefix.mp4",
        )
        if not self.postprocess:
            return raw_output

        final = scene_dir / "final_facefix.mp4"
        shutil.copy2(raw_output, final)
        return final

    def build_workflow(self, scene: dict, *, request: FaceFixSceneRequest) -> dict:
        patcher = WorkflowPatcher(self.load_workflow())
        self._patch_video_input(patcher, request)
        self._patch_reference_images(patcher, request)
        self._patch_facefix_params(patcher)
        self._patch_save_output(patcher, request.scene_number)
        return patcher.get()

    def _patch_video_input(self, patcher: WorkflowPatcher, request: FaceFixSceneRequest) -> None:
        video_name = self._upload_video(request.source_video)
        try:
            patcher.set_input_by_title("#LOAD_VIDEO", "video", video_name)
        except KeyError:
            patcher.set_input_by_title("#LOAD_VIDEO", "videopath", video_name)

    def _patch_reference_images(self, patcher: WorkflowPatcher, request: FaceFixSceneRequest) -> None:
        uploaded = self._upload_face_references(request.reference_images, request.scene_number)
        if uploaded:
            scaled_ids = []
            for path in uploaded:
                load_id = str(patcher.find_free_node_id())
                patcher.add_node(load_id, {
                    "inputs": {"image": path, "choose_folder_to_upload": "upload", "upload_folder": ""},
                    "class_type": "LoadImage",
                    "_meta": {"title": f"#FACE_REF_{load_id}"},
                })

                crop_id = str(patcher.find_free_node_id())
                patcher.add_node(crop_id, {
                    "inputs": {
                        "image": [load_id, 0],
                        "crop_padding_factor": 0.25,
                        "cascade_xml": "haarcascade_frontalface_default.xml",
                    },
                    "class_type": "Image Crop Face",
                    "_meta": {"title": f"#FACE_CROP_{crop_id}"},
                })

                scale_id = str(patcher.find_free_node_id())
                patcher.add_node(scale_id, {
                    "inputs": {
                        "image": [crop_id, 0],
                        "upscale_method": "lanczos",
                        "width": 512,
                        "height": 512,
                        "crop": "center",
                    },
                    "class_type": "ImageScale",
                    "_meta": {"title": f"#FACE_SCALE_{scale_id}"},
                })
                scaled_ids.append(scale_id)

            if len(scaled_ids) == 1:
                batch_id = scaled_ids[0]
            else:
                batch_id = str(patcher.find_free_node_id())
                patcher.add_node(batch_id, {
                    "inputs": {
                        "inputcount": len(scaled_ids),
                        "image_1": [scaled_ids[0], 0],
                    },
                    "class_type": "ImageBatchMulti",
                    "_meta": {"title": f"#FACE_BATCH_{batch_id}"},
                })
                for i, sid in enumerate(scaled_ids[1:], start=2):
                    patcher.set_input_by_id(batch_id, f"image_{i}", [sid, 0])

            try:
                _, sampler = patcher.find_node_by_meta_title("#LOOPING_SAMPLER")
                sampler["inputs"]["optional_cond_images"] = [batch_id, 0]
                self._face_batch_size = len(scaled_ids)
            except KeyError:
                self._face_batch_size = 0
        else:
            try:
                _, sampler = patcher.find_node_by_meta_title("#LOOPING_SAMPLER")
                sampler["inputs"].pop("optional_cond_images", None)
            except KeyError:
                pass
            self._face_batch_size = 0

        try:
            patcher.remove_node_by_title("#FACE_REFS")
        except KeyError:
            pass

    def _patch_facefix_params(self, patcher: WorkflowPatcher) -> None:
        cfg = self.config
        patcher.try_set_existing_input_by_title(
            "#LOOPING_SAMPLER", "guiding_strength", cfg.guiding_strength
        )
        patcher.try_set_existing_input_by_title(
            "#LOOPING_SAMPLER", "cond_image_strength", cfg.cond_image_strength
        )
        patcher.try_set_existing_input_by_title(
            "#LOOPING_SAMPLER", "temporal_tile_size", cfg.temporal_tile_size
        )
        patcher.try_set_existing_input_by_title(
            "#LOOPING_SAMPLER", "temporal_overlap", cfg.temporal_overlap
        )
        patcher.try_set_existing_input_by_title(
            "#LOOPING_SAMPLER", "temporal_overlap_cond_strength",
            cfg.temporal_overlap_cond_strength,
        )
        if self._face_batch_size > 0:
            keyframes = cfg.keyframe_indices
            if isinstance(keyframes, str):
                indices = [int(x.strip()) for x in keyframes.split(",") if x.strip()]
                valid = [str(i) for i in indices if i < self._face_batch_size]
                keyframes = ",".join(valid) if valid else "0"
            patcher.try_set_existing_input_by_title(
                "#LOOPING_SAMPLER", "optional_cond_image_indices", keyframes
            )

    def _patch_save_output(self, patcher: WorkflowPatcher, scene_number: int) -> None:
        patcher.set_input_by_title(
            "#SAVE_VIDEO", "filename_prefix",
            f"ltx_facefix_raw/scene_{scene_number:04}",
        )

    def _write_workflow_json(self, scene_dir: Path, scene_number: int, workflow: dict) -> None:
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "workflow_facefix.json").write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _upload_video(self, video_path: Path) -> str:
        upload_resp = self.client.upload_file_via_image_endpoint(
            video_path,
            subfolder="feverslop/facefix/input",
            file_type="input",
            overwrite=True,
        )
        return ComfyUIVideoAssetUploader.comfy_path_from_upload(upload_resp)

    def _upload_face_references(self, images: list[Path], scene_number: int) -> list[str]:
        subfolder = f"feverslop/facefix/references/scene_{scene_number:04}"
        paths = []
        for img_path in images:
            resp = self.client.upload_image(
                img_path,
                subfolder=subfolder,
                file_type="input",
                overwrite=True,
            )
            paths.append(ComfyUIVideoAssetUploader.comfy_path_from_upload(resp))
        return paths
