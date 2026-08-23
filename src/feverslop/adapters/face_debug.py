"""FaceDebugAdapter — writes debug artifacts from face pipeline.

Implements DebugArtifactPort protocol: writes detection overlays, crops,
masks, and composites for pipeline debugging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from feverslop.domain.face_detection import FaceDetection

logger = logging.getLogger(__name__)


class FaceDebugAdapter:
    """Writes face pipeline debug artifacts to disk.

    All artifacts are written as PNGs under ``output_dir/label/`` with
    zero-padded frame indexes in the filename for lexicographic sorting.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- DebugArtifactPort --

    def write_debug_image(
        self,
        frame_index: int,
        image: np.ndarray,
        label: str = "debug",
    ) -> Path:
        """Write a raw debug image."""
        path = self.output_dir / f"{label}_frame{frame_index:06d}.png"
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image)
        return path

    def write_detection_overlay(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: list[FaceDetection],
        decision_reason: str,
        extra_info: str | None = None,
    ) -> Path:
        """Write a frame with bounding boxes drawn for each detection."""
        overlay = frame.copy()

        for det in detections:
            x1, y1 = int(det.box.x1), int(det.box.y1)
            x2, y2 = int(det.box.x2), int(det.box.y2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put decision label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        label = decision_reason
        if extra_info:
            label = f"{label} | {extra_info}"
        (w, h), _ = cv2.getTextSize(
            label, font, font_scale, thickness,
        )
        # Position in top-right
        x = overlay.shape[1] - w - 10
        y = 20 + h
        cv2.putText(
            overlay, label, (x, y), font, font_scale, (0, 0, 255), thickness,
        )

        path = self.output_dir / f"detection_frame{frame_index:06d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return path

    def write_crop(
        self,
        frame_index: int,
        crop: np.ndarray,
        label: str = "crop",
    ) -> Path:
        """Write a face crop image."""
        path = self.output_dir / f"{label}_frame{frame_index:06d}.png"
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), crop)
        return path

    def write_mask(
        self,
        frame_index: int,
        mask: np.ndarray,
        label: str = "mask",
    ) -> Path:
        """Write a mask image (grayscale)."""
        path = self.output_dir / f"{label}_frame{frame_index:06d}.png"
        cv2.imwrite(str(path), mask)
        return path

    def write_composite(
        self,
        frame_index: int,
        original: np.ndarray,
        processed: np.ndarray,
        mask: np.ndarray,
    ) -> Path:
        """Write side-by-side composite: original | processed | mask."""
        h, w = original.shape[:2]
        composite = np.zeros((h, w * 3 + 4, 3), dtype=np.uint8)
        composite[:, :w] = original
        composite[:, w + 1:w * 2 + 1] = processed

        if mask.ndim == 2:
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_color = mask[:, :, :3]
        composite[:, w * 2 + 2:w * 3 + 2] = cv2.cvtColor(
            mask_color, cv2.COLOR_RGB2BGR,
        )

        path = self.output_dir / f"composite_frame{frame_index:06d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        return path
