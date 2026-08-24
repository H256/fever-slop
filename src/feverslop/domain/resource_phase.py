from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class StageResource(str, Enum):
    LLM = "LLM"
    COMFYUI = "ComfyUI"


_LLM_STAGES = frozenset({
    "main_pipeline",
    "relay_compact",
    "h3_prompts",
    "msr_prompt_enrich",
    "ingredients_sheets",
})
_COMFYUI_STAGES = frozenset({
    "storyboard_frames",
    "msr_references",
    "ltx_prepare_workflows",
    "ltx_render_scenes",
    "facefix",
    "upscale",
})
_NEUTRAL_STAGES = frozenset({
    "tests",
    "sync_project_settings",
    "anchor_fix",
    "msr_reference_sheets",
    "render_plan",
    "storyboard_page",
    "concat_video_only",
    "mux_original_audio",
    "diagnostic_scene_audio_concat",
    "export_timeline",
})


@dataclass(frozen=True)
class ResourcePhase:
    stages: tuple[str, ...]
    resource: StageResource | None
    next_resource: StageResource | None


def stage_resource(stage: str) -> StageResource | None:
    if stage in _LLM_STAGES:
        return StageResource.LLM
    if stage in _COMFYUI_STAGES:
        return StageResource.COMFYUI
    if stage in _NEUTRAL_STAGES:
        return None
    raise ValueError(f"unclassified safe-resume stage: {stage}")


def select_first_resource_phase(stages: Iterable[str]) -> ResourcePhase:
    selected: list[str] = []
    resource: StageResource | None = None
    for stage in stages:
        owner = stage_resource(stage)
        if owner is not None and resource is not None and owner is not resource:
            return ResourcePhase(tuple(selected), resource, owner)
        selected.append(stage)
        if owner is not None:
            resource = owner
    return ResourcePhase(tuple(selected), resource, None)
