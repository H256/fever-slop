import unittest

import numpy as np

from feverslop.adapters.face_compositor import (
    FaceCompositor,
    _find_entry_for_frame,
    color_match,
    radial_feather_mask,
    voronoi_partition,
)
from feverslop.domain.face_detection import FaceBox, FaceTrackEntry


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


if __name__ == "__main__":
    unittest.main()
