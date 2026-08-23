from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from feverslop.domain.face_composite import CompositeResult
from feverslop.domain.face_detection import FaceRepairData

logger = logging.getLogger(__name__)


def radial_feather_mask(size: int, feather_pixels: int) -> np.ndarray:
    """Create a radial feather mask of given size."""
    y, x = np.ogrid[:size, :size]
    cy, cx = size / 2, size / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = size / 2
    mask = np.clip(1.0 - dist / max_r, 0, 1)
    feather_radius = feather_pixels / max_r
    mask = np.clip(mask / feather_radius, 0, 1)
    return mask


def voronoi_partition(masks: list[np.ndarray], centers: list[tuple[int, int]], size: tuple[int, int]) -> list[np.ndarray]:
    """Partition overlapping feather masks using Voronoi tie-break.

    For each pixel in the overlap region, assign to the nearest face center.
    """
    h, w = size
    if len(masks) <= 1:
        return masks

    dist_map = np.full((h, w), np.inf, dtype=np.float32)
    owner = np.full((h, w), -1, dtype=np.int32)

    for idx, (cy, cx) in enumerate(centers):
        y, x = np.ogrid[:h, :w]
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = d < dist_map
        dist_map[mask] = d[mask]
        owner[mask] = idx

    partitioned = []
    for idx, mask in enumerate(masks):
        part = np.where(owner == idx, mask, 0.0)
        partitioned.append(part)

    return partitioned


def color_match(source: np.ndarray, target: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Match color statistics of source to target region."""
    src_f = source.astype(np.float64)
    tgt_f = target.astype(np.float64)

    src_mean = src_f.mean(axis=(0, 1))
    src_std = src_f.std(axis=(0, 1))
    tgt_mean = tgt_f.mean(axis=(0, 1))
    tgt_std = tgt_f.std(axis=(0, 1))

    epsilon = 1e-5
    matched = (src_f - src_mean) / (src_std + epsilon) * (tgt_std + epsilon) + tgt_mean
    result = np.clip(matched, 0, 255).astype(np.uint8)

    blended = cv2.addWeighted(source, 1.0 - strength, result, strength, 0)
    return blended


class FaceCompositor:
    """Composites repaired face frames back into original video frames.

    Uses radial feather masks with Voronoi partitioning for overlap handling
    and optional color matching.
    """

    def __init__(
        self,
        feather_pixels: int = 48,
        color_match_strength: float = 0.65,
        diagnostic: bool = False,
    ):
        self.feather_pixels = feather_pixels
        self.color_match_strength = color_match_strength
        self.diagnostic = diagnostic

    def composite(
        self,
        face_repairs: list[FaceRepairData],
        original_frames: np.ndarray,
        output_dir: Path | None = None,
    ) -> CompositeResult:
        if not face_repairs:
            return CompositeResult(
                composited_frames=original_frames.copy(),
                diagnostic_mask_path=None,
            )

        n_frames = len(original_frames)
        h, w = original_frames.shape[1:3]
        result = original_frames.copy()

        feather_base = radial_feather_mask(768, self.feather_pixels)

        diagnostic_mask = np.zeros((h, w), dtype=np.float32)

        for frame_idx in range(n_frames):
            frame_masks = []
            frame_centers = []

            for repair in face_repairs:
                entry = _find_entry_for_frame(repair.track_entries, frame_idx)
                if entry is None:
                    continue

                box = entry.box
                repaired_path = repair.repaired_frames_dir / f"repaired_{frame_idx:06d}.png"
                if not repaired_path.exists():
                    repaired_path = repair.repaired_frames_dir / f"repaired_{entry.frame_index:06d}.png"
                if not repaired_path.exists():
                    continue

                repaired = cv2.imread(str(repaired_path))
                if repaired is None:
                    continue

                repaired_resized = cv2.resize(repaired, (box.x2 - box.x1, box.y2 - box.y1))

                crop_h = box.y2 - box.y1
                crop_w = box.x2 - box.x1
                feather_scaled = cv2.resize(feather_base, (crop_w, crop_h))
                feather_clipped = feather_scaled[:repaired_resized.shape[0], :repaired_resized.shape[1]]

                if self.color_match_strength > 0:
                    y2_clip = min(box.y2, h)
                    x2_clip = min(box.x2, w)
                    target_region = original_frames[frame_idx, box.y1:y2_clip, box.x1:x2_clip]
                    if target_region.shape[:2] == repaired_resized.shape[:2]:
                        repaired_resized = color_match(
                            repaired_resized, target_region, self.color_match_strength,
                        )

                region_h = repaired_resized.shape[0]
                region_w = repaired_resized.shape[1]
                y_end = min(box.y1 + region_h, h)
                x_end = min(box.x1 + region_w, w)
                effective_h = y_end - box.y1
                effective_w = x_end - box.x1
                if effective_h <= 0 or effective_w <= 0:
                    continue

                mask_clipped = feather_clipped[:effective_h, :effective_w]
                repaired_clipped = repaired_resized[:effective_h, :effective_w]

                frame_region = result[frame_idx, box.y1:y_end, box.x1:x_end]
                blended = np.zeros_like(frame_region)
                for c in range(3):
                    blended[:, :, c] = np.clip(
                        mask_clipped * repaired_clipped[:, :, c] + (1 - mask_clipped) * frame_region[:, :, c],
                        0, 255,
                    ).astype(np.uint8)
                result[frame_idx, box.y1:y_end, box.x1:x_end] = blended

                mask_for_partition = np.zeros((h, w), dtype=np.float32)
                mask_for_partition[box.y1:y_end, box.x1:x_end] = mask_clipped
                frame_masks.append(mask_for_partition)
                frame_centers.append((box.y1 + effective_h // 2, box.x1 + effective_w // 2))

            if len(frame_masks) > 1:
                partitioned = voronoi_partition(frame_masks, frame_centers, (h, w))
                result_frame = original_frames[frame_idx].copy()
                for p_idx, repair in enumerate(face_repairs):
                    if p_idx >= len(partitioned):
                        break
                    entry = _find_entry_for_frame(repair.track_entries, frame_idx)
                    if entry is None:
                        continue
                    partition_mask = partitioned[p_idx]
                    if partition_mask.max() > 0:
                        box = entry.box
                    else:
                        continue

                for p_idx in range(len(frame_masks)):
                    entry = _find_entry_for_frame(face_repairs[p_idx].track_entries, frame_idx)
                    if entry is None:
                        continue
                    box = entry.box
                    repaired_path = face_repairs[p_idx].repaired_frames_dir / f"repaired_{frame_idx:06d}.png"
                    if not repaired_path.exists():
                        continue
                    repaired = cv2.imread(str(repaired_path))
                    if repaired is None:
                        continue
                    repaired_resized = cv2.resize(repaired, (box.x2 - box.x1, box.y2 - box.y1))
                    region_h = repaired_resized.shape[0]
                    region_w = repaired_resized.shape[1]
                    y_end = min(box.y1 + region_h, h)
                    x_end = min(box.x1 + region_w, w)
                    pm = partitioned[p_idx][box.y1:y_end, box.x1:x_end]
                    for c in range(3):
                        result_frame[box.y1:y_end, box.x1:x_end, c] = np.clip(
                            pm * repaired_resized[:, :, c] + (1 - pm) * result_frame[box.y1:y_end, box.x1:x_end, c],
                            0, 255,
                        ).astype(np.uint8)
                    diagnostic_mask = np.maximum(diagnostic_mask, pm)
                result[frame_idx] = result_frame
            elif frame_masks:
                diagnostic_mask = np.maximum(diagnostic_mask, frame_masks[0])

        diagnostic_mask_path = None
        if self.diagnostic and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            diagnostic_mask_path = output_dir / "diagnostic_mask.png"
            mask_uint8 = (diagnostic_mask * 255).astype(np.uint8)
            cv2.imwrite(str(diagnostic_mask_path), mask_uint8)

        return CompositeResult(
            composited_frames=result,
            diagnostic_mask_path=diagnostic_mask_path,
        )


def _find_entry_for_frame(entries: list, frame_idx: int):
    if not entries:
        return None
    closest = entries[0]
    min_diff = abs(closest.frame_index - frame_idx)
    for entry in entries[1:]:
        diff = abs(entry.frame_index - frame_idx)
        if diff < min_diff:
            min_diff = diff
            closest = entry
    if min_diff <= 2:
        return closest
    return None
