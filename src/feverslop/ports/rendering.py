from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class WorkflowAnchorConfig:
    positive_prompt_title: str = "#PROMPT_POSITIVE"
    negative_prompt_title: str = "#PROMPT_NEGATIVE"
    save_image_title: str = "#SAVE_IMAGE"
    character_lora_title: str | None = "#CHARACTER_LORA"
    single_prompt_title: str = "#PROMPT"
    single_prompt_input: str = "text"


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
    skip_existing: bool = True
    negative_prompt: str = ""
    character_lora_strength: float = 1.0
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


class ImageRenderBackend(Protocol):
    def render_image(self, request: ImageRenderRequest) -> Path:
        """Render one storyboard/startframe scene."""


class VideoRenderBackend(Protocol):
    def render_video(self, request: VideoRenderRequest) -> Path:
        """Render one video scene."""
