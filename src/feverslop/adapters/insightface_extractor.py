from __future__ import annotations

import logging
import urllib.request
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from feverslop.domain.face_detection import FaceBox

logger = logging.getLogger(__name__)

_MODEL_NAME = "buffalo_l"
_MODEL_DIR = "~/.insightface/models/buffalo_l"
_ADAFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/adaface_glint360k.onnx"
_ADAFACE_NAME = "adaface_glint360k.onnx"


def _load_analyzer() -> FaceAnalysis:
    model_dir = _ensure_model_dir()
    analyzer = FaceAnalysis(name=_MODEL_NAME, root=str(model_dir), providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    return analyzer


def _get_model_dir() -> Path:
    return Path(_MODEL_DIR).expanduser()


def _ensure_model_dir() -> Path:
    model_dir = _get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _download_adaface(model_dir: Path) -> Path:
    adaface_path = model_dir / _ADAFACE_NAME
    if adaface_path.exists():
        return adaface_path

    logger.info("Downloading AdaFace ONNX model...")
    urllib.request.urlretrieve(_ADAFACE_URL, str(adaface_path))  # noqa: S310
    logger.info("AdaFace downloaded to %s", adaface_path)
    return adaface_path


# Standard 5-point face alignment template (112x112)
_ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)


def _align_face(img: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Align face using 5-point landmarks to standard template."""
    src_pts = landmarks.astype(np.float64)
    dst_pts = _ARCFACE_TEMPLATE * (size / 112.0)

    mat = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if mat is None or mat[0] is None:
        logger.warning("Affine alignment failed, falling back to resize")
        return cv2.resize(img, (size, size))

    result = cv2.warpAffine(
        img,
        mat[0],
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return result


class InsightFaceExtractor:
    """Extracts face references and embeddings using InsightFace + AdaFace.

    Detection + landmarks: SCRFD from buffalo_l
    Recognition: AdaFace with 5-point alignment (112x112)
    """

    def __init__(self, analyzer: FaceAnalysis | None = None):
        self._analyzer = analyzer
        self._adaface = None

    @property
    def analyzer(self) -> FaceAnalysis:
        if self._analyzer is None:
            self._analyzer = _load_analyzer()
        return self._analyzer

    @property
    def adaface(self):
        if self._adaface is None:
            model_dir = _get_model_dir()
            adaface_path = _download_adaface(model_dir)
            self._adaface = get_model(str(adaface_path))
            self._adaface.prepare(ctx_id=-1)
        return self._adaface

    def _extract_adaface_embedding(self, img_bgr: np.ndarray, landmarks) -> np.ndarray | None:
        """Extract AdaFace embedding from 5-point aligned face."""
        if landmarks is None:
            logger.debug("AdaFace: landmarks is None, cannot align")
            return None
        try:
            aligned = _align_face(img_bgr, landmarks, size=112)
        except Exception as exc:
            logger.warning("AdaFace alignment failed: %s", exc, exc_info=True)
            return None

        try:
            feat = self.adaface.get_feat(aligned)
            if feat is not None:
                return feat.flatten()
        except Exception as exc:
            logger.warning("AdaFace embedding failed: %s", exc, exc_info=True)
            return None
        return None

    def extract_face_from_image(self, image_path: Path) -> Path | None:
        """Extract the largest face from an image and save as face_ref.png."""
        face_ref = image_path.parent / "face_ref.png"
        if face_ref.exists():
            return face_ref

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Cannot read image: %s", image_path)
            return None

        logger.debug("FaceAnalysis on %s (shape %s)", image_path, img.shape)
        faces = self.analyzer.get(img)
        logger.debug("Faces detected: %d", len(faces))

        if not faces:
            logger.info("No faces found in: %s", image_path)
            return None

        for i, f in enumerate(faces):
            bbox = f.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            lm_type = type(f.landmark).__name__ if f.landmark is not None else "None"
            lm_shape = f.landmark.shape if f.landmark is not None else "N/A"
            logger.debug(
                "Face %d: bbox=(%.0f,%.0f,%.0f,%.0f) area=%.0f score=%.3f landmark=%s(%s)",
                i, bbox[0], bbox[1], bbox[2], bbox[3], area, f.det_score, lm_type, lm_shape,
            )

        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = largest.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        face_img = _crop_square(img, x1, y1, x2, y2, padding=0.25)
        cv2.imwrite(str(face_ref), face_img)
        logger.info("Extracted face_ref: %s -> %s (crop shape %s)", image_path, face_ref, face_img.shape)
        return face_ref

    def extract_embedding(self, face_ref_path: Path) -> np.ndarray | None:
        """Extract recognition embedding from a face reference image using AdaFace."""
        if not face_ref_path.exists():
            logger.warning("face_ref missing: %s", face_ref_path)
            return None

        img = cv2.imread(str(face_ref_path))
        if img is None:
            logger.warning("Cannot read face_ref: %s", face_ref_path)
            return None

        logger.debug("Embedding extraction on %s (shape %s)", face_ref_path, img.shape)
        faces = self.analyzer.get(img)
        logger.debug("Faces in face_ref: %d", len(faces))

        if not faces:
            logger.info("No faces found in face_ref: %s", face_ref_path)
            return None

        face = faces[0]
        bbox = face.bbox
        lm_type = type(face.landmark).__name__ if face.landmark is not None else "None"
        lm_shape = face.landmark.shape if face.landmark is not None else "N/A"
        logger.debug(
            "face_ref face: bbox=(%.0f,%.0f,%.0f,%.0f) score=%.3f landmark=%s(%s)",
            bbox[0], bbox[1], bbox[2], bbox[3], face.det_score, lm_type, lm_shape,
        )

        embedding = self._extract_adaface_embedding(img, face.landmark)
        if embedding is not None:
            norm = np.linalg.norm(embedding)
            logger.debug("AdaFace embedding OK, shape=%s norm=%.4f", embedding.shape, norm)
            return embedding

        logger.warning("AdaFace embedding failed, falling back to buffalo_l ArcFace")
        return face.embedding.copy()

    def detect_all(
        self,
        video_frame_rgb: np.ndarray,
    ) -> list[FaceBox]:
        """Detect all faces in a frame. Returns embeddings via AdaFace."""
        img_bgr = cv2.cvtColor(video_frame_rgb, cv2.COLOR_RGB2BGR)
        faces = self.analyzer.get(img_bgr)
        if not faces:
            return []

        results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)

            adaface_emb = self._extract_adaface_embedding(img_bgr, face.landmark)
            if adaface_emb is not None:
                emb = adaface_emb
                emb_src = "AdaFace"
            else:
                emb = face.embedding.copy() if face.embedding is not None else None
                emb_src = "ArcFace" if emb is not None else "NONE"

            results.append(FaceBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=float(face.det_score),
                actor_id=None,
                embedding=emb,
            ))

            logger.debug(
                "[DIAG] Frame face %d: bbox=(%d,%d,%d,%d) area=%d det=%.3f emb=%s landmark=%s",
                i, x1, y1, x2, y2, area, face.det_score, emb_src, face.landmark is not None
            )

        return results

    def detect_and_match(
        self,
        video_frame_rgb: np.ndarray,
        actor_embeddings: dict[str, np.ndarray],
        threshold: float = 0.85,
    ) -> list[FaceBox]:
        """Detect faces in a frame and match to known actors by embedding."""
        results = self.detect_all(video_frame_rgb)
        if not results:
            return []

        if not actor_embeddings:
            return results

        matched_boxes = []
        unmatched_boxes = []
        for box in results:
            if box.embedding is not None:
                matched_id = _best_match_actor(box.embedding, actor_embeddings, threshold)
                matched_boxes.append(replace(box, actor_id=matched_id))
            else:
                unmatched_boxes.append(box)

        matched_count = sum(1 for b in matched_boxes if b.actor_id is not None)

        if matched_count == 0:
            all_boxes = matched_boxes + unmatched_boxes
            all_boxes.sort(key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1), reverse=True)
            actor_ids = list(actor_embeddings.keys())
            reassigned = []
            for i, box in enumerate(all_boxes):
                if i < len(actor_ids):
                    reassigned.append(replace(box, actor_id=actor_ids[i]))
                else:
                    reassigned.append(box)
            logger.warning(
                "[DIAG] No embedding matches, fallback: %d face(s) -> %d actor(s) by size",
                len(all_boxes), len(actor_ids)
            )
            return reassigned

        return matched_boxes + unmatched_boxes


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

        logger.debug(
            "[DIAG] Cosine vs %s: %.4f (threshold %.2f)",
            actor_id, score, threshold
        )

    if best_score >= threshold:
        return best_id
    return None


def _crop_square(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1)
    size += int(size * padding * 2)

    nx1 = int(cx - size / 2)
    ny1 = int(cy - size / 2)
    nx2 = nx1 + size
    ny2 = ny1 + size

    ox = max(0, -nx1)
    oy = max(0, -ny1)
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    crop = img[ny1:ny2, nx1:nx2]
    ch, cw = crop.shape[:2]

    if ch < size or cw < size:
        pad_h = max(0, size - ch)
        pad_w = max(0, size - cw)
        crop = np.pad(
            crop,
            ((oy, pad_h - oy), (ox, pad_w - ox), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return cv2.resize(crop, (size, size))
