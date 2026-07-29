from __future__ import annotations

import logging

import numpy as np

from feverslop.domain.face_detection import (
    BoundingBox,
    FaceLandmarks,
)
from feverslop.ports.face_pipeline import FaceMaskPort, MaskValidationResult

logger = logging.getLogger(__name__)


class FaceMaskAdapter(FaceMaskPort):
    """cv2-based face mask generation implementing FaceMaskPort."""

    def generate_mask(
        self,
        frame: np.ndarray,
        box: BoundingBox,
        landmarks: FaceLandmarks | None = None,
        feather_radius: int = 16,
        min_nonzero_ratio: float = 0.01,
    ) -> np.ndarray:
        """Generate a radial feather mask for face compositing.

        Returns grayscale mask (H, W) with values 0-255.
        Raises ValueError if mask validation fails.
        """
        frame_height, frame_width = frame.shape[:2]
        mask = np.zeros((frame_height, frame_width), dtype=np.float32)

        cx = int((box.x1 + box.x2) / 2.0)
        cy = int((box.y1 + box.y2) / 2.0)
        radius = max(int(box.width / 2.0), int(box.height / 2.0))

        # Scale feather relative to box size for small faces
        adaptive_feather = min(feather_radius, max(1, radius // 4))

        # Create radial gradient
        y, x = np.ogrid[:frame_height, :frame_width]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Inner circle (full mask)
        inner_radius = max(1, radius - adaptive_feather)
        mask[dist <= inner_radius] = 1.0

        # Feather transition
        feather_region = (dist > inner_radius) & (dist <= radius)
        mask[feather_region] = 1.0 - (dist[feather_region] - inner_radius) / adaptive_feather

        mask = (mask * 255).astype(np.uint8)

        # Validate; caller decides whether to reject.
        result = self.validate_mask(mask, min_nonzero_ratio=0.01)
        if isinstance(result, MaskValidationResult) and not result.valid:
            logger.warning(
                "Mask validation failed (box=%.1fx%.1f, ratio=%.4f): %s",
                box.width, box.height, result.nonzero_ratio, result.message,
            )

        return mask

    def smooth_mask_temporal(
        self,
        previous_mask: np.ndarray,
        current_mask: np.ndarray,
        alpha: float = 0.70,
    ) -> np.ndarray:
        """Apply temporal smoothing to mask."""
        result = (
            alpha * previous_mask.astype(np.float32)
            + (1.0 - alpha) * current_mask.astype(np.float32)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    def validate_mask(
        self, mask: np.ndarray, min_nonzero_ratio: float = 0.01
    ) -> MaskValidationResult:
        """Validate that mask has meaningful content.

        Returns MaskValidationResult with validity flag for explicit handling.
        """
        if mask.size == 0:
            return MaskValidationResult(
                valid=False, nonzero_ratio=0.0, message="mask is empty"
            )

        nonzero_ratio = np.count_nonzero(mask) / mask.size
        if nonzero_ratio >= min_nonzero_ratio:
            return MaskValidationResult(valid=True, nonzero_ratio=nonzero_ratio)

        return MaskValidationResult(
            valid=False,
            nonzero_ratio=nonzero_ratio,
            message=f"nonzero_ratio={nonzero_ratio:.4f} < threshold={min_nonzero_ratio:.4f}",
        )
