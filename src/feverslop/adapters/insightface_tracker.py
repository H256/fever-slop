from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from feverslop.adapters.insightface_extractor import InsightFaceExtractor, _crop_square
from feverslop.domain.face_detection import FaceBox, FaceCropResult, FaceTrackEntry

logger = logging.getLogger(__name__)

MAX_SHORT_GAP = 2
RETRY_MATCH_INTERVAL = 4
MATCH_THRESHOLD = 0.85
FALLBACK_AFTER_FRAMES = 5
FALLBACK_ACTOR_ID = "fallback_single"


class InsightFaceTracker:
    """Tracks actor faces through a video, produces crops and anchors.

    Uses retry-based embedding matching: tries to match detected faces to
    the target actor every frame until a match is found. Once matched,
    locks the actor_id and continues tracking by bbox proximity.

    If no embedding match is found within FALLBACK_AFTER_FRAMES frames,
    falls back to the largest detected face (for AI-generated faces where
    embeddings don't match reference photos).
    """

    def __init__(self, extractor: InsightFaceExtractor | None = None):
        self._extractor = extractor or InsightFaceExtractor()

    def track_video(
        self,
        video_path: Path,
        actor_embeddings: dict[str, np.ndarray],
        crop_size: int = 768,
        anchor_interval: int = 16,
        crop_padding: float = 0.25,
        output_dir: Path | None = None,
        actor_id: str | None = None,
    ) -> FaceCropResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        if output_dir is None:
            output_dir = Path(video_path).parent / "facefix"

        crop_frames_dir = output_dir / "crops"
        anchor_dir = output_dir / "anchors"
        crop_frames_dir.mkdir(parents=True, exist_ok=True)
        anchor_dir.mkdir(parents=True, exist_ok=True)

        _fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        _total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        entries: list[FaceTrackEntry] = []
        anchor_paths: list[Path] = []

        last_box: FaceBox | None = None
        gap_count = 0
        resolved_actor_id: str | None = None
        fallback_mode = False
        frames_with_faces = 0

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            all_faces = self._extractor.detect_all(frame_rgb)

            if all_faces:
                frames_with_faces += 1

            target_face: FaceBox | None = None

            # If actor not yet resolved, try matching each detected face.
            if resolved_actor_id is None and not fallback_mode:
                for face in all_faces:
                    if face.embedding is not None:
                        score = _cosine_match(face.embedding, actor_id, actor_embeddings)
                        if score is not None and score >= MATCH_THRESHOLD:
                            resolved_actor_id = actor_id
                            target_face = replace(face, actor_id=actor_id)
                            logger.info(
                                "FaceFix: resolved actor %s at frame %d (score %.3f)",
                                actor_id, frame_idx, score,
                            )
                            break

                # If we've seen faces but none matched the embedding, switch to size-based fallback.
                if resolved_actor_id is None and frames_with_faces >= FALLBACK_AFTER_FRAMES and not fallback_mode:
                    fallback_mode = True
                    logger.warning(
                        "FaceFix: embedding match failed after %d frames with faces, "
                        "falling back to largest face (actor_id=%s)",
                        frames_with_faces, actor_id or FALLBACK_ACTOR_ID,
                    )

            # Fallback mode: pick largest face.
            fallback_id = resolved_actor_id or FALLBACK_ACTOR_ID
            if fallback_mode and target_face is None and all_faces:
                largest = max(all_faces, key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1))
                if last_box is not None and _bbox_iou(largest, last_box) > 0.3:
                    target_face = replace(largest, actor_id=fallback_id)
                elif last_box is None:
                    target_face = replace(largest, actor_id=fallback_id)

            # If actor already resolved, find nearest face to last position.
            if resolved_actor_id is not None and target_face is None:
                for face in all_faces:
                    if last_box is None or _bbox_iou(face, last_box) > 0.3:
                        target_face = replace(face, actor_id=resolved_actor_id)
                        break

            # Short-gap fallback: reuse last box.
            if target_face is None and last_box is not None and gap_count <= MAX_SHORT_GAP:
                target_face = last_box
                gap_count += 1
            elif target_face is not None:
                last_box = target_face
                gap_count = 0
            else:
                last_box = None
                gap_count = 0

            if target_face is not None:
                box = target_face
                crop = _crop_square(frame_bgr, box.x1, box.y1, box.x2, box.y2, padding=crop_padding)
                crop_resized = cv2.resize(crop, (crop_size, crop_size))

                crop_path = crop_frames_dir / f"crop_{frame_idx:06d}.png"
                cv2.imwrite(str(crop_path), crop_resized)

                entries.append(FaceTrackEntry(
                    frame_index=frame_idx,
                    box=box,
                    crop_path=crop_path,
                ))

                if frame_idx % anchor_interval == 0 or frame_idx == 0:
                    anchor_path = anchor_dir / f"anchor_{frame_idx:06d}.png"
                    cv2.imwrite(str(anchor_path), crop_resized)
                    anchor_paths.append(anchor_path)

            frame_idx += 1

        cap.release()

        if not entries:
            logger.warning("No face tracks found for actor %s in %s", actor_id, video_path)

        return FaceCropResult(
            actor_id=actor_id or FALLBACK_ACTOR_ID,
            entries=entries,
            anchor_paths=sorted(anchor_paths),
            crop_frames_dir=crop_frames_dir,
            crop_mp4_path=None,
        )

    def track_video_with_encoder(
        self,
        video_path: Path,
        actor_embeddings: dict[str, np.ndarray],
        crop_size: int = 768,
        anchor_interval: int = 16,
        crop_padding: float = 0.25,
        output_dir: Path | None = None,
        actor_id: str | None = None,
        fps: float | None = None,
        ffmpeg_path: str = "ffmpeg",
    ) -> FaceCropResult:
        result = self.track_video(
            video_path=video_path,
            actor_embeddings=actor_embeddings,
            crop_size=crop_size,
            anchor_interval=anchor_interval,
            crop_padding=crop_padding,
            output_dir=output_dir,
            actor_id=actor_id,
        )

        if result.entries:
            from feverslop.adapters.face_video_encoder import encode_face_crop_mp4

            if fps is None:
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                cap.release()

            crop_mp4 = (output_dir or Path(video_path).parent / "facefix") / "face_crop.mp4"
            encode_face_crop_mp4(
                frames_folder=result.crop_frames_dir,
                fps=fps,
                output_path=crop_mp4,
                ffmpeg_path=ffmpeg_path,
            )
            result = replace(result, crop_mp4_path=crop_mp4)

        return result


def _cosine_match(
    embedding: np.ndarray,
    target_id: str | None,
    actor_embeddings: dict[str, np.ndarray],
) -> float | None:
    if target_id is None or target_id not in actor_embeddings:
        return None
    ref = actor_embeddings[target_id]
    return float(np.dot(embedding, ref) / (
        np.linalg.norm(embedding) * np.linalg.norm(ref) + 1e-8
    ))


def _bbox_iou(a: FaceBox, b: FaceBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, a.x2 - a.x1) * max(1, a.y2 - a.y1)
    area_b = max(1, b.x2 - b.x1) * max(1, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0
