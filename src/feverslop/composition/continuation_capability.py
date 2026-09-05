from feverslop.domain.duration_capability import DurationCapability
from feverslop.domain.minimax_h3_frames import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_DURATION_SECONDS,
    MINIMAX_H3_MIN_DURATION_SECONDS,
)
from feverslop.domain.video_workflow_profile import VideoWorkflowProfile


def resolve_continuation_capability(
    pipeline: str, profile: VideoWorkflowProfile | None,
) -> DurationCapability | None:
    if profile is not None and profile.duration_capability is not None:
        return profile.duration_capability
    if pipeline != "minimax-h3-r2v":
        return None
    return DurationCapability.create(
        fps=MINIMAX_H3_FPS,
        min_seconds=MINIMAX_H3_MIN_DURATION_SECONDS,
        max_seconds=MINIMAX_H3_MAX_DURATION_SECONDS,
        preferred_seconds=MINIMAX_H3_MAX_DURATION_SECONDS,
        frame_alignment=17,
        frame_offset=5,
    )
