"""FaceFix domain models."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

DEFAULT_FACEFIX_WORKFLOW = "video_ltxv_facefix_v1.json"
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
