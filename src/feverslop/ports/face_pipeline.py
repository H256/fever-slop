from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Protocol

from feverslop.domain.face_detection import (
    BoundingBox,
    FaceDetection,
    FaceEmbedding,
    FaceLandmarks,
)


class FaceDetectorPort(Protocol):
    """Detects faces and extracts embeddings from frames."""

    def detect_faces(
        self,
        frame: np.ndarray,
    ) -> list[FaceDetection]:
        """Detect all faces in a frame.

        Returns list of FaceDetection objects with bounding boxes, scores,
        landmarks, and embeddings. Empty list if no faces found.

        IMPORTANT: Returns ALL detections found; filtering happens later.
        """

    def extract_embedding(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray | None:
        """Extract face embedding for identity matching.

        Returns normalized embedding vector or None if extraction fails.
        """


class FaceIdentityPort(Protocol):
    """Manages identity references and performs face verification."""

    def register_reference(self, embedding: np.ndarray, actor_id: str) -> None:
        """Register a reference face embedding for an actor."""

    def verify_identity(
        self,
        detection_embedding: np.ndarray,
    ) -> tuple[str | None, float | None]:
        """Compare detection against registered references.

        Returns (actor_id, similarity_score) for best match,
        or (None, None) if no match found.
        """

    def get_actor_embedding(self, actor_id: str) -> np.ndarray | None:
        """Get reference embedding for a specific actor."""

    def get_all_embeddings(self) -> list[FaceEmbedding]:
        """Get all registered face embeddings."""


class FrameSourcePort(Protocol):
    """Reads frames from video sources."""

    def open(self, path: Path) -> None:
        """Open a video file for reading."""

    def read_frame(self, frame_index: int) -> np.ndarray | None:
        """Read a specific frame by index.

        Returns RGB numpy array (H, W, 3) or None if frame not available.
        """

    def close(self) -> None:
        """Close the video file and release resources."""

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""

    @property
    def frame_size(self) -> tuple[int, int]:
        """Video frame size as (width, height)."""

    @property
    def fps(self) -> float:
        """Frames per second."""


class VideoEncoderPort(Protocol):
    """Encodes processed frames back to video."""

    def open(
        self,
        output_path: Path,
        frame_size: tuple[int, int],
        fps: float,
        frame_count: int,
    ) -> None:
        """Open encoder for writing."""

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame."""

    def close(self) -> None:
        """Finalize and close the video file."""


class DebugArtifactPort(Protocol):
    """Writes debug artifacts for pipeline inspection."""

    def write_debug_image(
        self,
        frame_index: int,
        image: np.ndarray,
        label: str,
    ) -> Path:
        """Save a debug image.

        Returns the path to the saved file.
        """

    def write_detection_overlay(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: list[FaceDetection],
        decision_reason: str,
        extra_info: dict[str, str] | None = None,
    ) -> Path:
        """Save frame with detection boxes drawn.

        extra_info adds additional pipeline context lines to the overlay.
        Returns the path to the saved file.
        """

    def write_crop(self, frame_index: int, crop: np.ndarray, label: str = "crop") -> Path:
        """Save a face crop for inspection."""

    def write_mask(self, frame_index: int, mask: np.ndarray, label: str = "mask") -> Path:
        """Save a composite mask for inspection."""

    def write_composite(
        self,
        frame_index: int,
        original: np.ndarray,
        processed: np.ndarray,
        mask: np.ndarray,
    ) -> Path:
        """Save composite result showing original, processed, and mask."""


class FaceMaskPort(Protocol):
    """Generates and manipulates face masks for compositing."""

    def generate_mask(
        self,
        frame: np.ndarray,
        box: BoundingBox,
        landmarks: FaceLandmarks | None = None,
        feather_radius: int = 16,
    ) -> np.ndarray:
        """Generate a radial feather mask for face compositing.

        Returns grayscale mask (H, W) with values 0-255.
        """

    def smooth_mask_temporal(
        self,
        previous_mask: np.ndarray,
        current_mask: np.ndarray,
        alpha: float = 0.70,
    ) -> np.ndarray:
        """Apply temporal smoothing to mask.

        alpha controls how much to keep from previous mask (0-1).
        """

    def validate_mask(self, mask: np.ndarray, min_nonzero_ratio: float = 0.01) -> bool:
        """Validate that mask has meaningful content."""
