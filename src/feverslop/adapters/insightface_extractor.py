from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from feverslop.domain.face_detection import FaceBox

logger = logging.getLogger(__name__)

_MODEL_NAME = "buffalo_l"
_MODEL_DIR = "~/.insightface/models/buffalo_l"


def _get_model_dir() -> Path:
    return Path(_MODEL_DIR).expanduser()


def _ensure_model_dir() -> Path:
    model_dir = _get_model_dir()
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _load_analyzer() -> FaceAnalysis:
    model_dir = _ensure_model_dir()
    app = FaceAnalysis(name=_MODEL_NAME, root=str(model_dir), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1)
    return app


class InsightFaceExtractor:
    """Extracts face references and embeddings using InsightFace.

    Handles lazy extraction of face_ref.png from actor sheets and
    detection/matching of faces in video frames.
    """

    def __init__(self, analyzer: FaceAnalysis | None = None):
        self._analyzer = analyzer

    @property
    def analyzer(self) -> FaceAnalysis:
        if self._analyzer is None:
            self._analyzer = _load_analyzer()
        return self._analyzer

    def extract_face_from_image(self, image_path: Path) -> Path | None:
        """Extract the largest face from an image and save as face_ref.png."""
        face_ref = image_path.parent / "face_ref.png"
        if face_ref.exists():
            return face_ref

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Cannot read image: %s", image_path)
            return None

        faces = self.analyzer.get(img)
        if not faces:
            logger.warning("No faces found in: %s", image_path)
            return None

        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = largest.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        face_img = _crop_square(img, x1, y1, x2, y2, padding=0.25)
        cv2.imwrite(str(face_ref), face_img)
        logger.info("Extracted face_ref from %s -> %s", image_path, face_ref)
        return face_ref

    def extract_embedding(self, face_ref_path: Path) -> np.ndarray | None:
        """Extract recognition embedding from a face reference image."""
        if not face_ref_path.exists():
            return None

        img = cv2.imread(str(face_ref_path))
        if img is None:
            return None

        faces = self.analyzer.get(img)
        if not faces:
            return None

        return faces[0].embedding.copy()

    def detect_and_match(
        self,
        video_frame_rgb: np.ndarray,
        actor_embeddings: dict[str, np.ndarray],
        threshold: float = 0.85,
    ) -> list[FaceBox]:
        """Detect faces in a frame and match to known actors by embedding."""
        img_bgr = cv2.cvtColor(video_frame_rgb, cv2.COLOR_RGB2BGR)
        faces = self.analyzer.get(img_bgr)
        if not faces:
            return []

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            embedding = face.embedding

            matched_id = _best_match_actor(embedding, actor_embeddings, threshold)

            results.append(FaceBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=float(face.det_score),
                actor_id=matched_id,
                embedding=embedding.copy() if embedding is not None else None,
            ))

        return results


def _best_match_actor(
    embedding: np.ndarray,
    actor_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> str | None:
    if not actor_embeddings:
        return None

    best_score = 0.0
    best_id = None
    for actor_id, ref_emb in actor_embeddings.items():
        score = float(np.dot(embedding, ref_emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(ref_emb) + 1e-8
        ))
        if score > best_score:
            best_score = score
            best_id = actor_id

    if best_score >= threshold:
        return best_id
    return None


def _crop_square(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1)
    pad = int(size * padding)
    size += pad * 2

    nx1 = max(0, int(cx - size / 2))
    ny1 = max(0, int(cy - size / 2))
    nx2 = min(w, nx1 + size)
    ny2 = min(h, ny1 + size)

    pad_left = nx2 - nx1
    nx1 = max(0, nx1 + nx1 - nx2)
    ny1 = max(0, ny1 + ny1 - ny2)

    crop = img[ny1:ny1 + pad_left, nx1:nx1 + pad_left]
    if crop.shape[0] != crop.shape[1]:
        crop = cv2.resize(crop, (pad_left, pad_left))
    return crop
