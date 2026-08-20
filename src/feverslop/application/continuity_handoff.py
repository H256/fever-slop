from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from feverslop.domain.artifact_hash import sha256_file
from feverslop.domain.continuity import ContinuityHandoffPayload
from feverslop.domain.visual_consistency import (
    SceneConsistencyContract,
    can_handoff,
)
from feverslop.ports.visual_consistency import PreviousFramePort

_HANDOFF_MODES = {"msr", "i2v"}


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
        if (
            previous.mode not in _HANDOFF_MODES
            or current.mode not in _HANDOFF_MODES
            or previous.scene + 1 != current.scene
            or not can_handoff(previous, current)
        ):
            raise ValueError(
                f"Scene {current.scene} does not support continuity handoff"
            )

        project_dir_raw = getattr(self.frame_extractor, "project_dir", None)
        raw_source_clip = Path(previous_clip)
        resolved_project_dir = (
            Path(project_dir_raw).resolve()
            if project_dir_raw is not None
            else None
        )
        source_clip = (
            (resolved_project_dir / raw_source_clip).resolve()
            if resolved_project_dir is not None and not raw_source_clip.is_absolute()
            else raw_source_clip.resolve()
        )
        extracted = self.frame_extractor.extract_last_frame(
            source_clip,
            Path(output_frame),
        )
        stored_source_clip = (
            source_clip.relative_to(resolved_project_dir).as_posix()
            if resolved_project_dir is not None
            and source_clip.is_relative_to(resolved_project_dir)
            else source_clip.as_posix()
        )
        scene = deepcopy(dict(current_scene))
        keyframes = dict(scene.get("keyframes") or {})
        handoff: ContinuityHandoffPayload = {
            "source_scene": previous.scene,
            "last_frame_path": extracted.as_posix(),
            "last_frame_sha256": sha256_file(extracted),
            "transition": "continuous",
            "source_clip_path": stored_source_clip,
            "source_clip_sha256": sha256_file(source_clip),
            "extractor": "last-frame-v1",
        }
        keyframes.update(
            {
                "startframe_path": extracted.as_posix(),
                "startframe_sha256": sha256_file(extracted),
                "startframe_source_scene": previous.scene,
                "startframe_mode": "last_frame_from_previous",
                "startframe_source_clip_path": stored_source_clip,
                "startframe_source_clip_sha256": sha256_file(source_clip),
                "startframe_extractor": "last-frame-v1",
                "continuity_handoff": handoff,
            }
        )
        scene["keyframes"] = keyframes
        ltx = dict(scene.get("ltx") or {})
        ltx.update(
            {
                "msr_continuity_handoff_prompt": (
                    str(handoff_prompt or "").strip()
                    or "Hold the previous scene end state as the shot begins."
                ),
                "msr_continuity_handoff_frames": 18,
                "msr_continuity_msr_frame_count": 17,
                "msr_continuity_guide_frame_idx": 18,
            }
        )
        scene["ltx"] = ltx
        return scene
