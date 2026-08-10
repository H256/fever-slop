from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class WorkflowAnchorConfig:
    positive_prompt_title: str = "#PROMPT_POSITIVE"
    positive_prompt_input: str = "text"
    negative_prompt_title: str = "#PROMPT_NEGATIVE"
    save_image_title: str = "#SAVE_IMAGE"
    character_lora_title: str | None = "#LORA_1"
    single_prompt_title: str = "#PROMPT"
    single_prompt_input: str = "text"
    width_title: str | None = "#WIDTH"
    height_title: str | None = "#HEIGHT"
    width_input: str = "value"
    height_input: str = "value"
    reference_image_title: str | None = "#IMAGE_1"
    reference_image_input: str = "image"


@dataclass(frozen=True)
class RenderBackendConfig:
    workflow_path: Path
    anchors: WorkflowAnchorConfig = WorkflowAnchorConfig()


@dataclass(frozen=True)
class ImageRenderRequest:
    scene: dict
    scene_number: int
    prompt: str
    workflow_path: Path
    output_dir: Path
    width: int | None = None
    height: int | None = None
    skip_existing: bool = True
    negative_prompt: str = ""
    character_lora_strength: float | None = None
    reference_image: Path | None = None
    anchors: WorkflowAnchorConfig = WorkflowAnchorConfig()


@dataclass(frozen=True)
class VideoRenderRequest:
    scene: dict
    scene_number: int
    prompt: str
    workflow_path: Path
    output_dir: Path
    audio_file: Path
    storyboard_dir: Path
    render_mode: str = "single_prompt"
    single_prompt_workflow_path: Path | None = None
    skip_existing: bool = True
    uploaded_audio_name: str | None = None
    upload_audio: bool = True
    upload_startframes: bool = True
    anchors: WorkflowAnchorConfig = WorkflowAnchorConfig()
    render_plan_path: Path | None = None


class ImageRenderBackend(Protocol):
    def render_image(self, request: ImageRenderRequest) -> Path:
        """Render one storyboard/startframe scene."""


class VideoRenderBackend(Protocol):
    def render_video(self, request: VideoRenderRequest) -> Path:
        """Render one video scene."""
