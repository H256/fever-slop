from __future__ import annotations

from pathlib import Path
import json

from comfyui_client import ComfyUIClient
from ltx_video_renderer import LTXVideoRenderer
from ports.rendering import ImageRenderRequest, VideoRenderRequest
from workflow_patcher import WorkflowPatcher


class ComfyUIImageBackend:
    def __init__(
        self,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        seed_node_title: str | None = None,
        seed_input_name: str = "seed",
        filename_prefix_input_name: str = "filename_prefix",
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.output_dir = Path(output_dir)
        self.seed_node_title = seed_node_title
        self.seed_input_name = seed_input_name
        self.filename_prefix_input_name = filename_prefix_input_name

    def load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8"))

    def render_image(self, request: ImageRenderRequest) -> Path:
        workflow = self.load_workflow()
        patcher = WorkflowPatcher(workflow)

        scene_number = int(request.scene_number)
        anchors = request.anchors

        patcher.set_input_by_title(
            anchors.positive_prompt_title,
            "text",
            request.prompt,
        )

        if anchors.negative_prompt_title:
            patcher.set_input_by_title(
                anchors.negative_prompt_title,
                "text",
                request.negative_prompt,
            )

        if anchors.character_lora_title:
            patcher.patch_lora_strength_by_title(
                anchors.character_lora_title,
                request.character_lora_strength,
            )

        if self.seed_node_title:
            patcher.set_input_by_title(
                self.seed_node_title,
                self.seed_input_name,
                scene_number,
            )

        if anchors.save_image_title:
            patcher.set_input_by_title(
                anchors.save_image_title,
                self.filename_prefix_input_name,
                f"storyboard/scene_{scene_number:04}",
            )

        prompt_id = self.client.queue_prompt(patcher.get())
        history = self.client.wait_for_completion(prompt_id)
        images = self.client.extract_output_images(history)

        if not images:
            raise RuntimeError(f"No image output for scene {scene_number}")

        first = images[0]
        return self.client.download_view_file(
            filename=first["filename"],
            subfolder=first.get("subfolder", ""),
            file_type=first.get("type", "output"),
            output_path=request.output_dir / f"scene_{scene_number:04}.png",
        )


class ComfyUIVideoBackend(LTXVideoRenderer):
    def render_video(self, request: VideoRenderRequest) -> Path:
        one_scene_plan = request.output_dir / "_single_scene_plan.json"
        one_scene_plan.parent.mkdir(parents=True, exist_ok=True)
        one_scene_plan.write_text(
            json.dumps([request.scene], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rendered = self.render_videos(
            render_plan_path=one_scene_plan,
            audio_file=request.audio_file,
            storyboard_dir=request.storyboard_dir,
            skip_existing=request.skip_existing,
            uploaded_audio_name=request.uploaded_audio_name,
            upload_audio=request.upload_audio,
            upload_startframes=request.upload_startframes,
        )
        if not rendered:
            raise RuntimeError(f"No rendered video returned for scene {request.scene_number}")
        return rendered[0]
