import unittest
from pathlib import Path

import numpy as np

from feverslop.domain.face_composite import CompositeResult


class TestCompositeResult(unittest.TestCase):
    def test_construction(self):
        frames = np.zeros((10, 720, 1280, 3), dtype=np.uint8)
        result = CompositeResult(
            composited_frames=frames,
            diagnostic_mask_path=Path("/tmp/mask.png"),
        )
        self.assertEqual(result.composited_frames.shape, (10, 720, 1280, 3))
        self.assertEqual(result.diagnostic_mask_path, Path("/tmp/mask.png"))

    def test_no_diagnostic(self):
        frames = np.zeros((5, 480, 640, 3), dtype=np.uint8)
        result = CompositeResult(composited_frames=frames, diagnostic_mask_path=None)
        self.assertIsNone(result.diagnostic_mask_path)

    def test_frozen(self):
        result = CompositeResult(composited_frames=np.zeros((1, 10, 10, 3), dtype=np.uint8))
        with self.assertRaises(Exception):
            result.diagnostic_mask_path = Path("/tmp/x.png")


if __name__ == "__main__":
    unittest.main()
