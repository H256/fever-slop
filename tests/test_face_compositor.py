import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from feverslop.adapters.face_compositor import (
    FaceCompositor,
    _find_entry_for_frame,
    color_match,
    radial_feather_mask,
    voronoi_partition,
)
from feverslop.domain.face_detection import (
    FaceBox,
    FaceRepairData,
    FaceTrackEntry,
)


class TestRadialFeatherMask(unittest.TestCase):
    def test_shape(self):
        mask = radial_feather_mask(100, 20)
        self.assertEqual(mask.shape, (100, 100))

    def test_center_bright(self):
        mask = radial_feather_mask(100, 20)
        center = mask[50, 50]
        edge = mask[0, 0]
        self.assertGreater(center, edge)

    def test_range(self):
        mask = radial_feather_mask(256, 50)
        self.assertLessEqual(mask.max(), 1.0)
        self.assertGreaterEqual(mask.min(), 0.0)


class TestVoronoiPartition(unittest.TestCase):
    def test_single_mask(self):
        mask = np.ones((100, 100), dtype=np.float32)
        result = voronoi_partition([mask], [(50, 50)], (100, 100))
        self.assertEqual(len(result), 1)
        np.testing.assert_array_almost_equal(result[0], mask)

    def test_no_overlap(self):
        mask1 = np.zeros((100, 100), dtype=np.float32)
        mask1[:50, :50] = 1.0
        mask2 = np.zeros((100, 100), dtype=np.float32)
        mask2[50:, 50:] = 1.0
        result = voronoi_partition([mask1, mask2], [(25, 25), (75, 75)], (100, 100))
        self.assertEqual(len(result), 2)
        combined = result[0] + result[1]
        self.assertLessEqual(combined.max(), 1.0)

    def test_overlap_partitioned(self):
        mask1 = np.ones((100, 100), dtype=np.float32) * 0.5
        mask2 = np.ones((100, 100), dtype=np.float32) * 0.5
        result = voronoi_partition([mask1, mask2], [(25, 25), (75, 75)], (100, 100))
        self.assertEqual(len(result), 2)
        combined = result[0] + result[1]
        np.testing.assert_array_less(combined, 0.51)


class TestColorMatch(unittest.TestCase):
    def test_same_image(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = color_match(img, img, strength=0.5)
        np.testing.assert_array_almost_equal(result, img, decimal=0)

    def test_full_strength(self):
        source = np.full((50, 50, 3), 100, dtype=np.uint8)
        target = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = color_match(source, target, strength=1.0)
        mean = result.mean()
        self.assertGreater(mean, 100)
        self.assertLess(mean, 201)

    def test_no_strength(self):
        source = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        target = np.zeros((50, 50, 3), dtype=np.uint8)
        result = color_match(source, target, strength=0.0)
        np.testing.assert_array_equal(result, source)


class TestFindEntryForFrame(unittest.TestCase):
    def test_exact_match(self):
        entries = [FaceTrackEntry(frame_index=10, box=FaceBox(0, 0, 10, 10, 0.9))]
        result = _find_entry_for_frame(entries, 10)
        self.assertIsNotNone(result)

    def test_close_match(self):
        entries = [FaceTrackEntry(frame_index=8, box=FaceBox(0, 0, 10, 10, 0.9))]
        result = _find_entry_for_frame(entries, 10)
        self.assertIsNotNone(result)

    def test_too_far(self):
        entries = [FaceTrackEntry(frame_index=0, box=FaceBox(0, 0, 10, 10, 0.9))]
        result = _find_entry_for_frame(entries, 100)
        self.assertIsNone(result)

    def test_empty(self):
        result = _find_entry_for_frame([], 10)
        self.assertIsNone(result)

    def test_closest(self):
        entries = [
            FaceTrackEntry(frame_index=0, box=FaceBox(0, 0, 10, 10, 0.9)),
            FaceTrackEntry(frame_index=5, box=FaceBox(0, 0, 10, 10, 0.9)),
        ]
        result = _find_entry_for_frame(entries, 7)
        self.assertEqual(result.frame_index, 5)


class TestFaceCompositorEmpty(unittest.TestCase):
    def test_no_repairs(self):
        compositor = FaceCompositor()
        frames = np.zeros((10, 480, 640, 3), dtype=np.uint8)
        result = compositor.composite([], frames)
        np.testing.assert_array_equal(result.composited_frames, frames)
        self.assertIsNone(result.diagnostic_mask_path)


def _solid_png(path: Path, rgb: tuple[int, int, int]) -> None:
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:, :] = rgb
    assert cv2.imwrite(str(path), img)


def _make_repair(
    tmp: Path,
    actor_id: str,
    entries: list[FaceTrackEntry],
    write_frame: int | None,
    rgb: tuple[int, int, int],
) -> FaceRepairData:
    d = tmp / actor_id
    d.mkdir(parents=True, exist_ok=True)
    if write_frame is not None:
        _solid_png(d / f"repaired_{write_frame:06d}.png", rgb)
    return FaceRepairData(
        actor_id=actor_id,
        repaired_frames_dir=d,
        track_entries=entries,
        crop_size=48,
    )


class TestFaceCompositorPartitionAlignment(unittest.TestCase):
    """Regression for issue 1277 (ADAPT-002): the Voronoi partition mask must be
    applied to the same face whose repaired image is composited. A skipped face
    (no track entry / no repaired file) must not shift the mask-to-face mapping."""

    def test_skipped_face_does_not_swap_identity(self):
        # 3 faces A, B, C. A is skipped at frame 10 (no track entry within 2
        # frames, no repaired file). B and C are present with distinct colors.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            red = (255, 0, 0)
            blue = (0, 0, 255)

            a = _make_repair(
                tmp, "A",
                [FaceTrackEntry(frame_index=50, box=FaceBox(150, 50, 210, 110, 0.9))],
                None, red,
            )
            b = _make_repair(
                tmp, "B",
                [FaceTrackEntry(frame_index=1, box=FaceBox(50, 50, 110, 110, 0.9))],
                1, red,
            )
            c = _make_repair(
                tmp, "C",
                [FaceTrackEntry(frame_index=1, box=FaceBox(290, 50, 350, 110, 0.9))],
                1, blue,
            )

            frames = np.zeros((2, 200, 400, 3), dtype=np.uint8)
            compositor = FaceCompositor(color_match_strength=0.0)
            result = compositor.composite([a, b, c], frames)
            out = result.composited_frames[1]

            # B's region center must hold B's own color (red), not C's.
            b_center = out[80, 80]
            # C's region center must hold C's own color (blue), not B's.
            c_center = out[80, 320]

            self.assertGreater(b_center[0], 200, f"B region should be red, got {b_center}")
            self.assertLess(b_center[1], 60, f"B region should be red, got {b_center}")
            self.assertGreater(c_center[2], 200, f"C region should be blue, got {c_center}")
            self.assertLess(c_center[0], 60, f"C region should be blue, got {c_center}")

    def test_partition_uses_frame_index_fallback(self):
        # C's repaired file exists only at entry.frame_index (9), not frame_idx
        # (10). The partition path must fall back and still composite C's face.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            b = _make_repair(
                tmp, "B",
                [FaceTrackEntry(frame_index=1, box=FaceBox(50, 50, 110, 110, 0.9))],
                1, (255, 0, 0),
            )
            c = _make_repair(
                tmp, "C",
                [FaceTrackEntry(frame_index=0, box=FaceBox(290, 50, 350, 110, 0.9))],
                0, (0, 0, 255),
            )

            frames = np.zeros((2, 200, 400, 3), dtype=np.uint8)
            compositor = FaceCompositor(color_match_strength=0.0)
            result = compositor.composite([b, c], frames)
            out = result.composited_frames[1]

            c_center = out[80, 320]
            self.assertGreater(c_center[2], 200, f"C region should be blue, got {c_center}")


if __name__ == "__main__":
    unittest.main()
