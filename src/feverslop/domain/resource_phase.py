from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from feverslop.domain.stages import COMFYUI_STAGES, LLM_STAGES, NEUTRAL_STAGES


class StageResource(str, Enum):
    LLM = "LLM"
    COMFYUI = "ComfyUI"


@dataclass(frozen=True)
class ResourcePhase:
    stages: tuple[str, ...]
    resource: StageResource | None
    next_resource: StageResource | None


def stage_resource(stage: str) -> StageResource | None:
    if stage in LLM_STAGES:
        return StageResource.LLM
    if stage in COMFYUI_STAGES:
        return StageResource.COMFYUI
    if stage in NEUTRAL_STAGES:
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
