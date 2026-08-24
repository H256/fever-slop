from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


_RESUME_STAGE_ORDER = (
    "tests",
    "main_pipeline",
    "relay_compact",
    "anchor_fix",
    "msr_references",
    "msr_reference_sheets",
    "h3_prompts",
    "render_plan",
    "msr_prompt_enrich",
    "ingredients_sheets",
    "ltx_prepare_workflows",
    "ltx_render_scenes",
    "facefix",
    "upscale",
    "concat_video_only",
    "mux_original_audio",
    "diagnostic_scene_audio_concat",
    "export_timeline",
)
_RESUME_STAGE_INDEX = {stage: index for index, stage in enumerate(_RESUME_STAGE_ORDER)}


class PlanAction(str, Enum):
    RUN = "RUN"
    REUSE = "REUSE"
    BLOCKED = "BLOCKED"
    NOT_SELECTED = "NOT_SELECTED"


@dataclass(frozen=True)
class ExecutionPlanItem:
    phase: str
    action: PlanAction
    reason: str
    scene: int | None = None
    stage: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    project: Path
    mode: str
    items: tuple[ExecutionPlanItem, ...]

    @property
    def blocked(self) -> bool:
        return any(item.action is PlanAction.BLOCKED for item in self.items)

    @property
    def runnable_stages(self) -> tuple[str, ...]:
        if self.blocked:
            return ()
        stages = tuple(dict.fromkeys(
            item.stage
            for item in self.items
            if item.action is PlanAction.RUN and item.stage is not None
        ))
        if self.mode == "compatibility":
            return stages
        return tuple(sorted(
            stages,
            key=lambda stage: _RESUME_STAGE_INDEX.get(stage, len(_RESUME_STAGE_INDEX)),
        ))

    @property
    def runnable_scenes(self) -> tuple[int, ...]:
        if self.blocked:
            return ()
        return tuple(sorted({
            item.scene
            for item in self.items
            if item.action is PlanAction.RUN and item.scene is not None
        }))

    def runnable_scenes_for_stage(self, stage: str) -> tuple[int, ...]:
        if self.blocked:
            return ()
        return tuple(sorted({
            item.scene
            for item in self.items
            if item.action is PlanAction.RUN
            and item.stage == stage
            and item.scene is not None
        }))
