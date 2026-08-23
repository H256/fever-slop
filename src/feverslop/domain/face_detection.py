from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


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


# ---------------------------------------------------------------------------
# New domain models and functions for face pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0
        return self.width / self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def clamp(self, frame_width: int, frame_height: int) -> BoundingBox:
        x1 = max(0.0, min(self.x1, float(frame_width - 1)))
        y1 = max(0.0, min(self.y1, float(frame_height - 1)))
        x2 = max(x1 + 1.0, min(self.x2, float(frame_width)))
        y2 = max(y1 + 1.0, min(self.y2, float(frame_height)))
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class FaceLandmarks:
    """5-point face landmarks: left_eye, right_eye, nose, left_mouth, right_mouth."""

    points: list[tuple[float, float]]

    def __post_init__(self):
        # Ensure exactly 5 points
        if len(self.points) != 5:
            raise ValueError(f"Expected 5 landmarks, got {len(self.points)}")


@dataclass(frozen=True)
class FaceDetection:
    box: BoundingBox
    score: float
    landmarks: FaceLandmarks | None = None
    embedding: np.ndarray | None = None  # type: ignore[name-defined]


class TrackState(str, Enum):
    UNCONFIRMED = "UNCONFIRMED"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"


@dataclass
class FaceTrack:
    track_id: int
    state: TrackState
    box: BoundingBox
    smoothed_box: BoundingBox
    detection_score: float
    identity_score: float | None
    confirmed_frames: int
    missing_frames: int
    last_frame_index: int
    current_detection: FaceDetection | None = None


class RejectReason(str, Enum):
    NO_DETECTION = "NO_DETECTION"
    LOW_DETECTION_SCORE = "LOW_DETECTION_SCORE"
    INVALID_BOX = "INVALID_BOX"
    INVALID_LANDMARKS = "INVALID_LANDMARKS"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    FACE_TOO_LARGE = "FACE_TOO_LARGE"
    ASPECT_RATIO_INVALID = "ASPECT_RATIO_INVALID"
    TRACK_NOT_CONFIRMED = "TRACK_NOT_CONFIRMED"
    TRACK_MISMATCH = "TRACK_MISMATCH"
    IDENTITY_NOT_AVAILABLE = "IDENTITY_NOT_AVAILABLE"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    INVALID_CROP = "INVALID_CROP"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    INVALID_PROCESSED_CROP = "INVALID_PROCESSED_CROP"
    MASK_FAILED = "MASK_FAILED"
    COMPOSITE_FAILED = "COMPOSITE_FAILED"


@dataclass(frozen=True)
class FaceProcessingPolicy:
    min_detection_score: float = 0.3
    min_identity_score: float | None = None
    min_face_width: float = 16.0
    min_face_height: float = 16.0
    max_face_frame_ratio: float = 0.80
    min_face_aspect_ratio: float = 0.3
    max_face_aspect_ratio: float = 2.5
    track_confirmation_frames: int = 3
    track_max_missing_frames: int = 6
    min_track_iou: float = 0.20
    max_center_distance_ratio: float = 0.25
    face_crop_expansion: float = 1.4
    mask_feather_radius: int = 16
    mask_temporal_smoothing: float = 0.70
    enable_identity_check: bool = False
    debug_output: bool = True


@dataclass(frozen=True)
class FaceProcessingDecision:
    should_process: bool
    detection: FaceDetection | None
    track_id: int | None
    detection_score: float | None
    identity_score: float | None
    reject_reason: RejectReason | None

    @classmethod
    def accept(
        cls,
        detection: FaceDetection,
        track_id: int,
        detection_score: float,
        identity_score: float | None = None,
    ) -> FaceProcessingDecision:
        return cls(
            should_process=True,
            detection=detection,
            track_id=track_id,
            detection_score=detection_score,
            identity_score=identity_score,
            reject_reason=None,
        )

    @classmethod
    def reject(cls, reason: RejectReason) -> FaceProcessingDecision:
        return cls(
            should_process=False,
            detection=None,
            track_id=None,
            detection_score=None,
            identity_score=None,
            reject_reason=reason,
        )


@dataclass(frozen=True)
class FrameResult:
    frame: np.ndarray  # type: ignore[name-defined]
    processed: bool
    detection_score: float | None = None
    identity_score: float | None = None
    track_id: int | None = None
    box: BoundingBox | None = None
    expanded_box: BoundingBox | None = None
    reject_reason: RejectReason | None = None

    @classmethod
    def unchanged(cls, frame: np.ndarray, reject_reason: RejectReason) -> FrameResult:  # type: ignore[name-defined]
        return cls(
            frame=frame,
            processed=False,
            detection_score=None,
            identity_score=None,
            track_id=None,
            box=None,
            expanded_box=None,
            reject_reason=reject_reason,
        )

    @classmethod
    def processed(
        cls,
        frame: np.ndarray,  # type: ignore[name-defined]
        detection_score: float,
        identity_score: float | None,
        track_id: int,
        box: BoundingBox,
        expanded_box: BoundingBox,
    ) -> FrameResult:
        return cls(
            frame=frame,
            processed=True,
            detection_score=detection_score,
            identity_score=identity_score,
            track_id=track_id,
            box=box,
            expanded_box=expanded_box,
            reject_reason=None,
        )


# ---------------------------------------------------------------------------
# Geometry functions (pure Python + numpy, no framework imports)
# ---------------------------------------------------------------------------


def box_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_a.area + box_b.area - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def normalized_center_distance(
    box_a: BoundingBox,
    box_b: BoundingBox,
    frame_width: int,
    frame_height: int,
) -> float:
    """Calculate normalized center distance between two boxes."""
    diag = math.sqrt(frame_width ** 2 + frame_height ** 2)
    if diag == 0:
        return 0.0
    return math.dist(box_a.center, box_b.center) / diag


def valid_landmark_geometry(landmarks: FaceLandmarks) -> bool:
    """Validate 5-point landmark geometry.

    Expected order: left_eye, right_eye, nose, left_mouth, right_mouth
    (relative to the person, NOT image coordinates)
    """
    if len(landmarks.points) < 5:
        return False

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks.points[:5]

    # Check all finite
    for point in landmarks.points[:5]:
        if not all(math.isfinite(v) for v in point):
            return False

    # Eye distance should be reasonable (not collapsed, not too wide)
    eye_dist = math.dist(left_eye, right_eye)
    if eye_dist < 5.0:  # minimum eye distance
        return False

    # Eyes should be above nose, nose above mouth
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0

    if not (eye_y < nose[1] < mouth_y):
        return False

    # Vertical span should be reasonable
    if mouth_y - eye_y < 10.0:
        return False

    # Mouth width should be reasonable relative to eye distance
    mouth_dist = math.dist(left_mouth, right_mouth)
    if mouth_dist < eye_dist * 0.15:
        return False

    return True


def is_valid_face_detection(
    detection: FaceDetection,
    frame_width: int,
    frame_height: int,
    policy: FaceProcessingPolicy,
) -> bool:
    """Check if a face detection passes all validation criteria."""
    box = detection.box

    # NaN/Infinity check
    for val in [box.x1, box.y1, box.x2, box.y2, detection.score]:
        if not math.isfinite(val):
            return False

    # Score check
    if detection.score < policy.min_detection_score:
        return False

    # Box dimensions
    if box.width <= 0 or box.height <= 0:
        return False

    if box.width < policy.min_face_width:
        return False

    if box.height < policy.min_face_height:
        return False

    if box.width > frame_width * policy.max_face_frame_ratio:
        return False

    if box.height > frame_height * policy.max_face_frame_ratio:
        return False

    # Aspect ratio
    if not (policy.min_face_aspect_ratio <= box.aspect_ratio <= policy.max_face_aspect_ratio):
        return False

    # Box not completely outside frame
    if box.x2 <= 0 or box.y2 <= 0:
        return False

    if box.x1 >= frame_width or box.y1 >= frame_height:
        return False

    # Landmark validation
    if detection.landmarks is not None:
        if not valid_landmark_geometry(detection.landmarks):
            return False

    return True


def filter_detections(
    detections: list[FaceDetection],
    frame_width: int,
    frame_height: int,
    policy: FaceProcessingPolicy,
) -> list[FaceDetection]:
    """Filter detections, keeping only valid ones."""
    return [
        d for d in detections
        if is_valid_face_detection(d, frame_width, frame_height, policy)
    ]


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedFaceCandidate:
    detection: FaceDetection
    score: float
    track_match: bool


def rank_face_candidates(
    detections: list[FaceDetection],
    frame_width: int,
    frame_height: int,
    previous_track: FaceTrack | None = None,
    policy: FaceProcessingPolicy | None = None,
) -> list[RankedFaceCandidate]:
    """Rank face candidates for selection.

    Scores based on:
    - Detection confidence (50%)
    - Temporal IoU with previous track (25%)
    - Face size relative to frame (15%)
    - Proximity to frame center (10%)
    """
    if not detections:
        return []

    if policy is None:
        policy = FaceProcessingPolicy()

    candidates = []
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    for detection in detections:
        box = detection.box

        # Detection score (0-1, already normalized)
        det_score = detection.score

        # Temporal IoU score
        if previous_track is not None:
            temporal_iou = box_iou(box, previous_track.smoothed_box)
        else:
            temporal_iou = 0.0

        # Face size score (larger = better, normalized)
        max_area = frame_width * frame_height * policy.max_face_frame_ratio ** 2
        size_score = min(1.0, box.area / max(1.0, max_area))

        # Center proximity (closer to center = better)
        dist_from_center = math.dist(box.center, (center_x, center_y))
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)
        center_score = 1.0 - (dist_from_center / max(1.0, max_dist))

        # Weighted score
        if previous_track is not None:
            candidate_score = (
                0.50 * det_score
                + 0.25 * temporal_iou
                + 0.15 * size_score
                + 0.10 * center_score
            )
            is_track_match = temporal_iou >= 0.20
        else:
            candidate_score = (
                0.50 * det_score
                + 0.20 * 0.0  # no temporal
                + 0.20 * size_score
                + 0.10 * center_score
            )
            is_track_match = False

        candidates.append(RankedFaceCandidate(
            detection=detection,
            score=candidate_score,
            track_match=is_track_match,
        ))

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Identity similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[name-defined]
    """Calculate cosine similarity between two vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Face processing decision
# ---------------------------------------------------------------------------


def decide_face_processing(
    detection: FaceDetection | None,
    track: FaceTrack | None,
    identity_score: float | None,
    policy: FaceProcessingPolicy,
) -> FaceProcessingDecision:
    """Central decision: should this face be processed?"""
    if detection is None:
        return FaceProcessingDecision.reject(RejectReason.NO_DETECTION)

    if detection.score < policy.min_detection_score:
        return FaceProcessingDecision.reject(RejectReason.LOW_DETECTION_SCORE)

    if track is None or track.state is not TrackState.CONFIRMED:
        return FaceProcessingDecision.reject(RejectReason.TRACK_NOT_CONFIRMED)

    if policy.enable_identity_check:
        if identity_score is None:
            return FaceProcessingDecision.reject(RejectReason.IDENTITY_NOT_AVAILABLE)

        if policy.min_identity_score is not None and identity_score < policy.min_identity_score:
            return FaceProcessingDecision.reject(RejectReason.IDENTITY_MISMATCH)

    return FaceProcessingDecision.accept(
        detection=detection,
        track_id=track.track_id,
        detection_score=detection.score,
        identity_score=identity_score,
    )


# ---------------------------------------------------------------------------
# Box manipulation
# ---------------------------------------------------------------------------


def expand_box(
    box: BoundingBox,
    expansion_factor: float,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    """Expand a bounding box by a factor around its center."""
    cx = box.center[0]
    cy = box.center[1]

    new_width = box.width * expansion_factor
    new_height = box.height * expansion_factor

    expanded = BoundingBox(
        x1=cx - new_width / 2.0,
        y1=cy - new_height / 2.0,
        x2=cx + new_width / 2.0,
        y2=cy + new_height / 2.0,
    )

    return expanded.clamp(frame_width, frame_height)


def smooth_box(
    previous_box: BoundingBox,
    current_box: BoundingBox,
    alpha: float = 0.70,
) -> BoundingBox:
    """Exponential smoothing of bounding box coordinates."""
    return BoundingBox(
        x1=alpha * previous_box.x1 + (1.0 - alpha) * current_box.x1,
        y1=alpha * previous_box.y1 + (1.0 - alpha) * current_box.y1,
        x2=alpha * previous_box.x2 + (1.0 - alpha) * current_box.x2,
        y2=alpha * previous_box.y2 + (1.0 - alpha) * current_box.y2,
    )
