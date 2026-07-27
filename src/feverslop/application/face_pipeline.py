from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from feverslop.domain.face_detection import (
    BoundingBox,
    FaceDetection,
    FaceProcessingDecision,
    FaceProcessingPolicy,
    FaceTrack,
    FrameResult,
    RejectReason,
    TrackState,
    box_iou,
    decide_face_processing,
    expand_box,
    filter_detections,
    rank_face_candidates,
    smooth_box,
)

logger = logging.getLogger(__name__)


@dataclass
class FramePipelineState:
    """Mutable state for processing a single frame."""

    frame_index: int
    frame: np.ndarray
    detections: list[FaceDetection] = field(default_factory=list)
    filtered_detections: list[FaceDetection] = field(default_factory=list)
    candidate: FaceDetection | None = None
    decision: FaceProcessingDecision | None = None
    track: FaceTrack | None = None
    identity_score: float | None = None
    identity_actor_id: str | None = None
    result: FrameResult | None = None

    def __post_init__(self):
        if not self.detections:
            self.detections = []
        if not self.filtered_detections:
            self.filtered_detections = []


class FacePipeline:
    """Application-level orchestrator for face detection, tracking, and processing.

    Coordinates the following ports:
    - FaceDetectorPort: detection and embedding
    - FaceIdentityPort: identity verification
    - FaceMaskPort: mask generation and smoothing
    - FrameSourcePort: frame reading
    - VideoEncoderPort: frame writing
    - DebugArtifactPort: debug output

    Fail-safe behavior:
    - If no plausible face is detected, the frame is returned unchanged
    - All processing failures fall back to the original frame
    - False negatives are acceptable; false positives are not
    """

    def __init__(
        self,
        detector: "FaceDetectorPort",
        identity_port: "FaceIdentityPort",
        mask_port: "FaceMaskPort",
        debug_port: "DebugArtifactPort | None",
        policy: FaceProcessingPolicy | None = None,
    ):
        self.detector = detector
        self.identity_port = identity_port
        self.mask_port = mask_port
        self.debug_port = debug_port
        self.policy = policy or FaceProcessingPolicy()

        # Track management
        self._tracks: dict[int, FaceTrack] = {}
        self._next_track_id: int = 0
        self._previous_mask: np.ndarray | None = None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Process a single frame through the face pipeline.

        Returns FrameResult with the processed (or unchanged) frame.
        """
        frame_height, frame_width = frame.shape[:2]
        state = FramePipelineState(
            frame_index=frame_index,
            frame=frame,
        )

        # Step 1: Detect faces
        state.detections = self._detect_faces(state, frame)

        # Step 2: Filter detections
        state.filtered_detections = self._filter_detections(
            state, frame_width, frame_height
        )

        # Step 3: Rank candidates
        candidate = self._rank_and_select_candidate(state)
        state.candidate = candidate

        # Step 4: Update temporal tracking
        state.track = self._update_track(state, candidate)

        # Step 5: Identity verification
        identity_score, actor_id = self._verify_identity(state)
        state.identity_score = identity_score
        state.identity_actor_id = actor_id

        # Step 6: Make processing decision
        decision = self._make_decision(state)
        state.decision = decision

        # Step 7: Process or reject
        result = self._execute_decision(state)
        state.result = result

        # Step 8: Debug output
        if self.debug_port is not None and self.policy.debug_output:
            self._write_debug(state)

        return result

    def _detect_faces(
        self, state: FramePipelineState, frame: np.ndarray
    ) -> list[FaceDetection]:
        """Step 1: Run face detection."""
        try:
            detections = self.detector.detect_faces(frame)
            logger.debug(
                "Frame %d: detected %d faces",
                state.frame_index,
                len(detections),
            )
            return detections
        except Exception as e:
            logger.error("Frame %d: detection failed: %s", state.frame_index, e)
            return []

    def _filter_detections(
        self,
        state: FramePipelineState,
        frame_width: int,
        frame_height: int,
    ) -> list[FaceDetection]:
        """Step 2: Filter detections by validity."""
        filtered = filter_detections(
            state.detections, frame_width, frame_height, self.policy
        )
        logger.debug(
            "Frame %d: %d detections after filtering (%d rejected)",
            state.frame_index,
            len(filtered),
            len(state.detections) - len(filtered),
        )
        return filtered

    def _rank_and_select_candidate(
        self, state: FramePipelineState
    ) -> FaceDetection | None:
        """Step 3: Rank candidates and select the best one."""
        frame_height, frame_width = state.frame.shape[:2]

        candidates = rank_face_candidates(
            state.filtered_detections,
            frame_width,
            frame_height,
            previous_track=next(iter(self._tracks.values())) if self._tracks else None,
            policy=self.policy,
        )

        if not candidates:
            return None

        best = candidates[0]
        logger.debug(
            "Frame %d: selected candidate score=%.4f, track_match=%s",
            state.frame_index,
            best.score,
            best.track_match,
        )
        return best.detection

    def _update_track(
        self, state: FramePipelineState, candidate: FaceDetection | None
    ) -> FaceTrack | None:
        """Step 4: Update temporal tracking state."""
        if candidate is None:
            # No detection — age existing tracks
            self._age_tracks(state.frame_index)
            return None

        # Try to match with existing track
        track = self._match_track(state.frame_index, candidate.box)

        if track is None:
            # Create new track
            track = self._create_track(state.frame_index, candidate)
        else:
            # Update existing track
            self._update_existing_track(track, state.frame_index, candidate)

        return track

    def _match_track(
        self, frame_index: int, box: BoundingBox
    ) -> FaceTrack | None:
        """Find the best matching track for a detection."""
        best_track: FaceTrack | None = None
        best_iou = 0.0

        for track in self._tracks.values():
            if track.state is TrackState.LOST:
                continue

            iou = box_iou(box, track.smoothed_box)
            if iou >= self.policy.min_track_iou and iou > best_iou:
                best_iou = iou
                best_track = track

        return best_track

    def _create_track(
        self, frame_index: int, detection: FaceDetection
    ) -> FaceTrack:
        """Create a new track for a detection."""
        track_id = self._next_track_id
        self._next_track_id += 1

        track = FaceTrack(
            track_id=track_id,
            state=TrackState.UNCONFIRMED,
            box=detection.box,
            smoothed_box=detection.box,
            detection_score=detection.score,
            identity_score=None,
            confirmed_frames=1,
            missing_frames=0,
            last_frame_index=frame_index,
            current_detection=detection,
        )
        self._tracks[track_id] = track
        logger.debug("Frame %d: created new track %d", frame_index, track_id)
        return track

    def _update_existing_track(
        self,
        track: FaceTrack,
        frame_index: int,
        detection: FaceDetection,
    ) -> None:
        """Update an existing track with a new detection."""
        # Smooth the box
        track.smoothed_box = smooth_box(
            track.smoothed_box,
            detection.box,
            alpha=self.policy.mask_temporal_smoothing,
        )

        # Update state
        track.box = detection.box
        track.detection_score = detection.score
        track.last_frame_index = frame_index
        track.missing_frames = 0
        track.current_detection = detection

        # Confirm track after enough frames
        if track.state is TrackState.UNCONFIRMED:
            track.confirmed_frames += 1
            if track.confirmed_frames >= self.policy.track_confirmation_frames:
                track.state = TrackState.CONFIRMED
                logger.debug(
                    "Frame %d: track %d confirmed",
                    frame_index,
                    track.track_id,
                )

    def _age_tracks(self, frame_index: int) -> None:
        """Age tracks when no detection is found."""
        for track in self._tracks.values():
            if track.state is TrackState.LOST:
                continue

            track.missing_frames += 1
            if track.missing_frames >= self.policy.track_max_missing_frames:
                track.state = TrackState.LOST
                logger.debug(
                    "Frame %d: track %d lost after %d missing frames",
                    frame_index,
                    track.track_id,
                    track.missing_frames,
                )

    def _verify_identity(
        self, state: FramePipelineState
    ) -> tuple[float | None, str | None]:
        """Step 5: Verify identity of the candidate."""
        if not self.policy.enable_identity_check:
            return None, None

        if state.candidate is None or state.candidate.embedding is None:
            return None, None

        try:
            actor_id, similarity = self.identity_port.verify_identity(
                state.candidate.embedding
            )
            logger.debug(
                "Frame %d: identity check -> actor=%s, score=%.4f",
                state.frame_index,
                actor_id,
                similarity,
            )
            return similarity, actor_id
        except Exception as e:
            logger.error("Frame %d: identity verification failed: %s", state.frame_index, e)
            return None, None

    def _make_decision(self, state: FramePipelineState) -> FaceProcessingDecision:
        """Step 6: Make the processing decision."""
        return decide_face_processing(
            detection=state.candidate,
            track=state.track,
            identity_score=state.identity_score,
            policy=self.policy,
        )

    def _execute_decision(self, state: FramePipelineState) -> FrameResult:
        """Step 7: Execute the decision (process or reject)."""
        if state.decision is None or not state.decision.should_process:
            reason = (
                state.decision.reject_reason
                if state.decision is not None and state.decision.reject_reason is not None
                else RejectReason.NO_DETECTION
            )
            logger.info(
                "Frame %d: REJECTED %s (dets=%d, filtered=%d, track=%s, identity=%s)",
                state.frame_index,
                reason,
                len(state.detections),
                len(state.filtered_detections),
                state.track.state if state.track else "None",
                state.identity_score,
            )
            return FrameResult.unchanged(state.frame, reason)

        if state.candidate is None:
            return FrameResult.unchanged(state.frame, RejectReason.NO_DETECTION)

        try:
            processed_frame = self._process_face(state)
            box = state.candidate.box
            expanded_box = expand_box(
                box, self.policy.face_crop_expansion,
                state.frame.shape[1], state.frame.shape[0],
            )

            return FrameResult.processed(
                frame=processed_frame,
                detection_score=state.candidate.score,
                identity_score=state.identity_score,
                track_id=state.track.track_id if state.track is not None else -1,
                box=box,
                expanded_box=expanded_box,
            )
        except Exception as e:
            logger.error("Frame %d: face processing failed: %s", state.frame_index, e)
            return FrameResult.unchanged(state.frame, RejectReason.PROCESSING_FAILED)

    def _process_face(self, state: FramePipelineState) -> np.ndarray:
        """Process the face region and composite back into frame."""
        if state.candidate is None:
            return state.frame

        frame = state.frame.copy()
        frame_height, frame_width = frame.shape[:2]
        box = state.candidate.box

        # Expand box for crop region
        expanded_box = expand_box(
            box, self.policy.face_crop_expansion,
            frame_width, frame_height,
        )

        # Generate mask
        mask = self.mask_port.generate_mask(
            frame,
            expanded_box,
            state.candidate.landmarks,
            feather_radius=self.policy.mask_feather_radius,
        )

        # Validate mask
        if not self.mask_port.validate_mask(mask):
            raise ValueError("Invalid mask generated")

        # Temporal mask smoothing
        if self._previous_mask is not None:
            mask = self.mask_port.smooth_mask_temporal(
                self._previous_mask,
                mask,
                alpha=self.policy.mask_temporal_smoothing,
            )
        self._previous_mask = mask

        # TODO: In the full implementation, crop the face region, run V2V processing,
        # and composite the processed face back using the mask.
        # For now, return the frame unchanged with the mask applied.

        return frame

    def _write_debug(self, state: FramePipelineState) -> None:
        """Step 8: Write debug artifacts with full pipeline context."""
        if self.debug_port is None:
            return

        # Write detection overlay
        if state.detections:
            reason = (
                "ACCEPTED" if state.decision and state.decision.should_process
                else f"REJECTED: {state.decision.reject_reason}" if state.decision
                else "NO_DETECTION"
            )
            # Build extra debug info
            extra = {}
            # Detection details
            for i, det in enumerate(state.detections):
                extra[f"det[{i}]"] = f"score={det.score:.3f} box=({det.box.x1:.0f},{det.box.y1:.0f},{det.box.x2:.0f},{det.box.y2:.0f})"
                if det.landmarks:
                    extra[f"det[{i}]_lm"] = f"points={len(det.landmarks.points)}"
                if det.embedding is not None:
                    extra[f"det[{i}]_emb"] = f"dim={det.embedding.shape}"
                else:
                    extra[f"det[{i}]_emb"] = "None"
            # Filter results
            extra["filtered"] = f"{len(state.filtered_detections)}/{len(state.detections)} passed"
            # Track state
            if state.track:
                extra["track"] = f"id={state.track.track_id} state={state.track.state} confirmed={state.track.confirmed_frames} missing={state.track.missing_frames}"
            else:
                extra["track"] = "None"
            # Identity
            if state.identity_score is not None:
                extra["identity"] = f"score={state.identity_score:.4f} actor={state.identity_actor_id}"
            else:
                extra["identity"] = "None"
            # Decision details
            if state.decision:
                extra["decision"] = f"process={state.decision.should_process} reason={state.decision.reject_reason}"

            self.debug_port.write_detection_overlay(
                frame_index=state.frame_index,
                frame=state.frame,
                detections=state.detections,
                decision_reason=reason,
                extra_info=extra,
            )

    def reset(self) -> None:
        """Reset pipeline state for a new video."""
        self._tracks.clear()
        self._next_track_id = 0
        self._previous_mask = None


# Type imports for Protocol references (only used in type hints)
from feverslop.ports.face_pipeline import (  # noqa: E402
    DebugArtifactPort,
    FaceDetectorPort,
    FaceIdentityPort,
    FaceMaskPort,
)
