from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from feverslop.domain.face_detection import (
    BoundingBox,
    FaceDetection,
    FaceLandmarks,
)
from feverslop.ports.face_pipeline import FaceDetectorPort

logger = logging.getLogger(__name__)


class InsightFaceDetectorAdapter(FaceDetectorPort):
    """InsightFace-based face detector adapter implementing FaceDetectorPort."""

    def __init__(self, model_dir: Path | None = None):
        self._model_dir = model_dir
        self._analyzer: Any | None = None

    @property
    def analyzer(self) -> Any:
        if self._analyzer is None:
            from insightface.app import FaceAnalysis

            model_dir = (
                str(self._model_dir)
                if self._model_dir
                else "~/.insightface/models/buffalo_l"
            )
            self._analyzer = FaceAnalysis(
                name="buffalo_l",
                root=model_dir,
                providers=["CPUExecutionProvider"],
            )
            self._analyzer.prepare(ctx_id=0, det_size=(640, 640))
        return self._analyzer

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect all faces in an RGB frame."""
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = self.analyzer.get(img_bgr)

        detections: list[FaceDetection] = []
        for face in faces:
            bbox = face.bbox.astype(np.float64)
            x1, y1, x2, y2 = bbox

            landmarks: FaceLandmarks | None = None
            if face.landmark is not None:
                lm = face.landmark.flatten().tolist()
                if len(lm) >= 10:
                    landmarks = FaceLandmarks(
                        points=[
                            (lm[0], lm[1]),  # left eye
                            (lm[2], lm[3]),  # right eye
                            (lm[4], lm[5]),  # nose
                            (lm[6], lm[7]),  # left mouth
                            (lm[8], lm[9]),  # right mouth
                        ]
                    )

            embedding = face.embedding.copy() if face.embedding is not None else None

            detections.append(
                FaceDetection(
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=float(face.det_score),
                    landmarks=landmarks,
                    embedding=embedding,
                )
            )

        return detections

    def extract_embedding(
        self, frame: np.ndarray, box: BoundingBox
    ) -> np.ndarray | None:
        """Extract face embedding from a specific region."""
        try:
            x1, y1, x2, y2 = (
                int(box.x1),
                int(box.y1),
                int(box.x2),
                int(box.y2),
            )
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            img_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            faces = self.analyzer.get(img_bgr)

            if faces:
                return faces[0].embedding.copy()
        except Exception as e:
            logger.error("Embedding extraction failed: %s", e)

        return None
