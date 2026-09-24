import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import MagicMock, patch

import numpy as np

from feverslop.adapters.insightface_extractor import (
    InsightFaceExtractor,
    _crop_square,
    _download_adaface,
)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class TestCropSquare(unittest.TestCase):
    def test_basic_square(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        crop = _crop_square(img, 100, 100, 200, 200, padding=0.0)
        self.assertEqual(crop.shape[0], crop.shape[1])
        self.assertGreaterEqual(crop.shape[0], 100)

    def test_with_padding(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        crop = _crop_square(img, 100, 100, 200, 200, padding=0.25)
        self.assertEqual(crop.shape[0], crop.shape[1])
        expected = int(100 * 1.5)
        self.assertEqual(crop.shape[0], expected)

    def test_rectangular_bbox(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        crop = _crop_square(img, 50, 100, 150, 300, padding=0.0)
        self.assertEqual(crop.shape[0], crop.shape[1])

    def test_near_boundary(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _crop_square(img, 80, 80, 100, 100, padding=0.0)
        self.assertEqual(crop.shape[0], crop.shape[1])


class TestInsightFaceExtractor(unittest.TestCase):
    def test_detect_all_returns_empty_no_faces(self):
        extractor = InsightFaceExtractor()
        extractor._analyzer = MagicMock()
        extractor._analyzer.get.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.detect_all(frame)
        self.assertEqual(result, [])

    def test_analyzer_lazy_init(self):
        extractor = InsightFaceExtractor()
        self.assertIsNone(extractor._analyzer)


class TestDownloadAdaface(unittest.TestCase):
    def test_existing_file_skips_download(self):
        model_dir = Path(mkdtemp(prefix="adaface-test-"))
        try:
            final = model_dir / "adaface_glint360k.onnx"
            final.write_bytes(b"complete-model")
            with patch(
                "feverslop.adapters.insightface_extractor.urllib.request.urlretrieve"
            ) as mock_retrieve:
                result = _download_adaface(model_dir)
            self.assertEqual(result, final)
            self.assertEqual(result.read_bytes(), b"complete-model")
            mock_retrieve.assert_not_called()
        finally:
            _rmtree(model_dir)

    def test_successful_download_renames_into_place(self):
        model_dir = Path(mkdtemp(prefix="adaface-test-"))
        try:
            def fake_retrieve(url, dest):
                Path(dest).write_bytes(b"complete-model")

            with patch(
                "feverslop.adapters.insightface_extractor.urllib.request.urlretrieve",
                side_effect=fake_retrieve,
            ):
                result = _download_adaface(model_dir)
            self.assertEqual(result, model_dir / "adaface_glint360k.onnx")
            self.assertEqual(result.read_bytes(), b"complete-model")
            # No stray temp files remain.
            leftovers = [p for p in model_dir.iterdir() if p.name.endswith(".part")]
            self.assertEqual(leftovers, [])
        finally:
            _rmtree(model_dir)

    def test_interrupted_download_leaves_no_partial_file(self):
        model_dir = Path(mkdtemp(prefix="adaface-test-"))
        try:
            def fake_retrieve(url, dest):
                Path(dest).write_bytes(b"partial")
                raise OSError("connection reset")

            with patch(
                "feverslop.adapters.insightface_extractor.urllib.request.urlretrieve",
                side_effect=fake_retrieve,
            ):
                with self.assertRaises(OSError):
                    _download_adaface(model_dir)
            # The final model must NOT exist (so the next run retries) and no
            # temp file is left behind.
            self.assertFalse((model_dir / "adaface_glint360k.onnx").exists())
            leftovers = [p for p in model_dir.iterdir() if p.name.endswith(".part")]
            self.assertEqual(leftovers, [])
        finally:
            _rmtree(model_dir)


if __name__ == "__main__":
    unittest.main()
