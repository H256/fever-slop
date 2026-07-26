from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


DEFAULT_FACEFIX_WORKFLOW = "video_ltxv_facefix_v1.json"
DEFAULT_KEYFRAME_INDICES = "0,16,32,48"


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
    """Single-scene FaceFix request."""

    scene_number: int
    source_video: Path
    reference_images: Sequence[Path] = ()
    output_dir: Path = Path(".")

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"scene_{self.scene_number:04}_facefix.mp4"
