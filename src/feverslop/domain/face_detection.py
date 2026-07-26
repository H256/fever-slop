from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaceEmbedding:
    """InsightFace recognition embedding for a single actor."""

    actor_id: str
    embedding: np.ndarray


@dataclass(frozen=True)
class FaceBox:
    """Detected face bounding box with recognition data."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    actor_id: str | None = None
    embedding: np.ndarray | None = None


@dataclass(frozen=True)
class FaceTrackEntry:
    """Single tracked face position for one frame."""

    frame_index: int
    box: FaceBox
    crop_path: Path | None = None


@dataclass(frozen=True)
class FaceCropResult:
    """Result of tracking one actor's face through a video."""

    actor_id: str
    entries: list[FaceTrackEntry]
    anchor_paths: list[Path]
    crop_frames_dir: Path
    crop_mp4_path: Path | None = None


@dataclass(frozen=True)
class FaceRepairData:
    """Repaired face frames for compositing back into original."""

    actor_id: str
    repaired_frames_dir: Path
    track_entries: list[FaceTrackEntry]
    crop_size: int
