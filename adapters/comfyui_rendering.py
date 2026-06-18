from __future__ import annotations

from pathlib import Path
import json

from ltx_video_renderer import LTXVideoRenderer
from ports.rendering import ImageRenderRequest, VideoRenderRequest
from storyboard_renderer import StoryboardRenderer


class ComfyUIImageBackend:
    def __init__(self, renderer: StoryboardRenderer):
        self.renderer = renderer

    def render_image(self, request: ImageRenderRequest) -> Path:
        return self.renderer.render_scene_startframe(request.scene)


class ComfyUIVideoBackend:
    def __init__(self, renderer: LTXVideoRenderer):
        self.renderer = renderer

    def render_video(self, request: VideoRenderRequest) -> Path:
        one_scene_plan = request.output_dir / "_single_scene_plan.json"
        one_scene_plan.parent.mkdir(parents=True, exist_ok=True)
        one_scene_plan.write_text(
            json.dumps([request.scene], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rendered = self.renderer.render_videos(
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
