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
from feverslop.domain.facefix_rendering import DEFAULT_FACEFIX_WORKFLOW, FaceFixConfig


class ComfyUIFaceFixCropBackend:
    """Renders face-repaired video via LTXV FaceFix workflow.

    Takes face-crop MP4 + anchor PNGs, patches the workflow, and runs
    the LTXV LoopingSampler with anchor conditioning. CLIP runs on CPU.
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
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        postprocessor: VideoPostProcessor | None = None,
        model_resolver=None,
        project_dir: str | Path | None = None,
    ):
        self.client = client
        self.workflow_path = Path(workflow_path) if workflow_path else Path(DEFAULT_FACEFIX_WORKFLOW)
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
        self._patch_anchors(patcher, anchors_dir)
        self._patch_facefix_params(patcher)
        self._patch_save_output(patcher, scene_number, actor_id)
        return patcher.get()

    def _patch_video_input(self, patcher: WorkflowPatcher, face_crop_mp4: Path, scene_number: int, actor_id: str) -> None:
        video_name = self._upload_video(face_crop_mp4, scene_number, actor_id)
        try:
            patcher.set_input_by_title("#LOAD_VIDEO", "video", video_name)
        except KeyError:
            patcher.set_input_by_title("#LOAD_VIDEO", "videopath", video_name)

    def _patch_anchors(self, patcher: WorkflowPatcher, anchors_dir: Path) -> None:
        anchor_pngs = sorted(anchors_dir.glob("*.png"))
        if not anchor_pngs:
            return

        uploaded_paths = self._upload_anchors(anchors_dir)
        scaled_ids = self._build_anchor_image_chain(patcher, uploaded_paths)
        batch_id = self._batch_scaled_images(patcher, scaled_ids)

        if batch_id:
            try:
                _, sampler = patcher.find_node_by_meta_title("#LOOPING_SAMPLER")
                sampler["inputs"]["optional_cond_images"] = [batch_id, 0]
            except KeyError:
                pass

            anchor_count = len(anchor_pngs)
            indices = [str(i) for i in range(0, anchor_count, 16)]
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

    def _upload_anchors(self, anchors_dir: Path) -> list[str]:
        subfolder = f"feverslop/facefix_crop/anchors/{anchors_dir.name}"
        paths = []
        for png in sorted(anchors_dir.glob("*.png")):
            resp = self.client.upload_image(
                png,
                subfolder=subfolder,
                file_type="input",
                overwrite=True,
            )
            paths.append(ComfyUIVideoAssetUploader.comfy_path_from_upload(resp))
        return paths

    def _build_anchor_image_chain(self, patcher: WorkflowPatcher, uploaded_paths: list[str]) -> list[str]:
        scaled_ids = []
        for path in uploaded_paths:
            load_id = str(patcher.find_free_node_id())
            patcher.add_node(load_id, {
                "inputs": {"image": path, "choose_folder_to_upload": "upload", "upload_folder": ""},
                "class_type": "LoadImage",
                "_meta": {"title": f"#ANCHOR_LOAD_{load_id}"},
            })

            scale_id = str(patcher.find_free_node_id())
            patcher.add_node(scale_id, {
                "inputs": {
                    "image": [load_id, 0],
                    "upscale_method": "lanczos",
                    "width": 768,
                    "height": 768,
                    "crop": "center",
                },
                "class_type": "ImageScale",
                "_meta": {"title": f"#ANCHOR_SCALE_{scale_id}"},
            })
            scaled_ids.append(scale_id)
        return scaled_ids

    def _batch_scaled_images(self, patcher: WorkflowPatcher, scaled_ids: list[str]) -> str | None:
        if not scaled_ids:
            return None
        if len(scaled_ids) == 1:
            return scaled_ids[0]

        current = scaled_ids[0]
        for i in range(1, len(scaled_ids)):
            batch_id = str(patcher.find_free_node_id())
            patcher.add_node(batch_id, {
                "inputs": {
                    "image1": [current, 0],
                    "image2": [scaled_ids[i], 0],
                },
                "class_type": "ImageBatch",
                "_meta": {"title": f"#ANCHOR_BATCH_{batch_id}"},
            })
            current = batch_id
        return current
