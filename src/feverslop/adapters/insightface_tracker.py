from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from feverslop.adapters.insightface_extractor import InsightFaceExtractor, _crop_square
from feverslop.domain.face_detection import FaceBox, FaceCropResult, FaceTrackEntry

logger = logging.getLogger(__name__)

MAX_SHORT_GAP = 2


class InsightFaceTracker:
    """Tracks actor faces through a video, produces crops and anchors."""

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

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = self._extractor.detect_and_match(frame_rgb, actor_embeddings)

            target_faces = [f for f in faces if f.actor_id == actor_id or (actor_id is None and f.actor_id)]
            if not target_faces and last_box is not None and gap_count <= MAX_SHORT_GAP:
                target_faces = [last_box]
                gap_count += 1
            elif target_faces:
                last_box = target_faces[0]
                gap_count = 0
            else:
                last_box = None
                gap_count = 0

            if target_faces:
                box = target_faces[0]
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
            actor_id=actor_id or "unknown",
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
            result = result._replace(crop_mp4_path=crop_mp4)

        return result
