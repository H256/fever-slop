"""FaceFix domain models."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

DEFAULT_FACEFIX_WORKFLOW = "video_ltxv_facefix_v1.json"
# H3-native FaceRefine workflow (issue 519, unit 2). Video-to-video: the
# rendered source clip is encoded to an H3 video latent, refined with the
# scene's actor references + prompt at a low denoise, and the refined face
# region is composited back by the Python pipeline.
DEFAULT_H3_FACEFIX_WORKFLOW = "video_minimax_h3_facefix_v1.json"
DEFAULT_KEYFRAME_INDICES = "0,16,32,48"

# H3 pipelines whose rendered scenes are refined by the H3-native FaceRefine
# pass rather than the LTXV crop path. The issue scope is MiniMax H3 R2V;
# other H3 variants stay on the crop path until explicitly added.
_H3_FACEFIX_PIPELINES = frozenset({"minimax-h3-r2v"})


class FaceFixBackendKind(Enum):
    """Which FaceFix backend refines a scene.

    - LTXV_CROP: the existing crop-and-composite LTXV LoopingSampler path.
    - H3_FACEFIX: the H3-native video-to-video FaceRefine pass (issue 519).
    """

    LTXV_CROP = "ltxv_crop"
    H3_FACEFIX = "h3_facefix"


def select_facefix_backend(
    video_pipeline: str,
    override: str | None = None,
) -> FaceFixBackendKind:
    """Select the FaceFix backend for a scene's video pipeline.

    An explicit ``override`` (one of the ``FaceFixBackendKind`` values) wins
    over the video-pipeline heuristic. Otherwise H3 R2V scenes route to the
    H3-native FaceRefine pass and everything else keeps the LTXV crop path.
    """
    if override is not None:
        return FaceFixBackendKind(override)
    if video_pipeline in _H3_FACEFIX_PIPELINES:
        return FaceFixBackendKind.H3_FACEFIX
    return FaceFixBackendKind.LTXV_CROP


# MiniMax H3 samples on a 16k+5 frame grid (e.g. 5, 21, 37, 201, 217). A
# source clip whose frame count does not sit on this grid must be snapped
# before an H3 video-to-video pass; the original count is preserved by the
# Python compositing step (U2) so the refined latent never changes scene
# duration or frame count.
H3_FRAME_GRID_STRIDE = 16
H3_FRAME_GRID_OFFSET = 5


def is_h3_frame_grid_compatible(frame_count: int) -> bool:
    """True when ``frame_count`` sits on H3's 16k+5 grid."""
    return frame_count > 0 and (frame_count - H3_FRAME_GRID_OFFSET) % H3_FRAME_GRID_STRIDE == 0


def snap_to_h3_frame_grid(frame_count: int) -> int:
    """Snap ``frame_count`` to the nearest 16k+5 value (>= 5).

    Used to size the H3 video-to-video latent. The compositing step maps the
    refined frames back onto the original frame count, so snapping only affects
    the render, not the final artifact.
    """
    if frame_count <= 0:
        return H3_FRAME_GRID_OFFSET
    k = round((frame_count - H3_FRAME_GRID_OFFSET) / H3_FRAME_GRID_STRIDE)
    return max(H3_FRAME_GRID_OFFSET, H3_FRAME_GRID_STRIDE * k + H3_FRAME_GRID_OFFSET)


@dataclass(frozen=True)
class FaceFixConfig:
    """Configuration for the LTXV FaceFix postprocessing pass."""

    workflow_path: Path = field(default_factory=lambda: Path(DEFAULT_FACEFIX_WORKFLOW))
    keyframe_indices: str = DEFAULT_KEYFRAME_INDICES
    guiding_strength: float = 0.2
    cond_image_strength: float = 0.5
    temporal_tile_size: int = 56
    temporal_overlap: int = 24
    temporal_overlap_cond_strength: float = 0.5
    adain_factor: float = 0.0
    face_reference_folder: str | None = None
    postprocess: bool = True
    ffmpeg_path: str = "ffmpeg"


@dataclass(frozen=True)
class FaceFixSceneRequest:
    """Single-scene FaceFix request.

    output_dir is the scene dir (e.g. render/scenes/scene_0001).
    """

    scene_number: int
    source_video: Path
    reference_images: Sequence[Path] = ()
    output_dir: Path = Path()
