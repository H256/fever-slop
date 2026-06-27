from __future__ import annotations

from pathlib import Path
import json

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIMSRVideoRenderBackend:
    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        seed_offset: int = 100000,
        msr_frame_count: int = 17,
        debug_workflows_dir: str | Path | None = None,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        model_resolver=None,
    ):
        if int(msr_frame_count) not in {17, 25, 33, 41}:
            raise ValueError("msr_frame_count must be one of 17, 25, 33, 41")
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.output_dir = Path(output_dir)
        self.raw_output_dir = self.output_dir
        self.seed_offset = int(seed_offset)
        self.msr_frame_count = int(msr_frame_count)
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    def load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def render_video(self, request: VideoRenderRequest) -> Path:
        scene_number = int(request.scene_number)
        workflow = self.build_workflow(request.scene, prompt=request.prompt)
        workflow = self.model_resolver.resolve_workflow_models(workflow, workflow_path=self.workflow_path)
        return self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=self.raw_output_dir / f"scene_{scene_number:04}_raw.mp4",
        )

    def build_workflow(self, scene: dict, *, prompt: str) -> dict:
        scene_number = int(scene["scene"])
        references = scene.get("references") or {}
        actor_reference_paths = references.get("actor_msr_paths") or references.get("actor_sheet_paths", [])
        actor_paths = [Path(path) for path in actor_reference_paths]
        if not actor_paths:
            raise ValueError(f"Scene {scene_number} references at least 1 actor for ltx_msr")
        if len(actor_paths) > 4:
            raise ValueError(f"Scene {scene_number} references at most 4 actors for ltx_msr")
        location_path = references.get("location_msr_path") or references.get("location_sheet_path")
        if not location_path:
            raise ValueError(f"Scene {scene_number} is missing references.location_msr_path")

        patcher = WorkflowPatcher(self.load_workflow())
        for index, actor_path in enumerate(actor_paths, start=1):
            patcher.set_input_by_title(
                f"#MSR_ACTOR_{index}",
                "image",
                self.asset_uploader.resolve_reference_image_name(actor_path),
            )
        patcher.set_input_by_title(
            "#MSR_BACKGROUND",
            "image",
            self.asset_uploader.resolve_reference_image_name(location_path),
        )
        patcher.set_input_by_title("#PROMPT", "text", str(prompt).strip())
        patcher.set_input_by_title("#SAVE_VIDEO", "filename_prefix", f"ltx_msr_raw/scene_{scene_number:04}")
        patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "frame_count", self.msr_frame_count)
        patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "value", self.msr_frame_count)
        patcher.try_set_existing_input_by_title("#SEED", "noise_seed", self.seed_offset + scene_number)
        patcher.try_set_existing_input_by_title("#WIDTH", "value", int(scene.get("width", 0) or 0))
        patcher.try_set_existing_input_by_title("#HEIGHT", "value", int(scene.get("height", 0) or 0))

        workflow = patcher.get()
        if self.debug_workflows_dir:
            self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json").write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return workflow
