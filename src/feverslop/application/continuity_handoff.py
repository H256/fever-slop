from __future__ import annotations

from typing import Any, Mapping

from feverslop.domain.visual_consistency import (
    SceneConsistencyContract,
    apply_continuity_handoff,
)
from feverslop.ports.visual_consistency import PreviousFramePort


class ContinuityHandoffUseCase:
    def __init__(self, frame_extractor: PreviousFramePort):
        self.frame_extractor = frame_extractor

    def execute(
        self,
        previous: SceneConsistencyContract,
        current: SceneConsistencyContract,
        previous_clip,
        output_frame,
        current_scene: Mapping[str, Any],
        *,
        handoff_prompt: str | None = None,
    ) -> dict[str, Any]:
        return apply_continuity_handoff(
            previous,
            current,
            previous_clip,
            output_frame,
            current_scene,
            frame_extractor=self.frame_extractor,
            handoff_prompt=handoff_prompt,
        )
