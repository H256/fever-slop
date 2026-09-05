import unittest
from unittest.mock import MagicMock

import numpy as np

from feverslop.adapters.insightface_extractor import (
    InsightFaceExtractor,
    _best_match_actor,
    _crop_square,
)


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


class TestBestMatchActor(unittest.TestCase):
    def test_zero_vectors_do_not_match(self):
        zero = np.zeros(2)
        nonzero = np.array([1.0, 0.0])
        for query, reference in ((zero, nonzero), (nonzero, zero), (zero, zero)):
            with self.subTest(query=query, reference=reference):
                self.assertIsNone(_best_match_actor(query, {"hero": reference}, 0.70))

    def test_exact_threshold_uses_unbiased_cosine_score(self):
        reference = np.array([1.0, 0.0])
        query = np.array([0.8, 0.6])
        self.assertEqual("hero", _best_match_actor(query, {"hero": reference}, 0.8))
        self.assertIsNone(_best_match_actor(query, {"hero": reference}, 0.81))

    def test_no_embeddings(self):
        emb = np.array([1.0, 0.0, 0.0])
        result = _best_match_actor(emb, {}, 0.5)
        self.assertIsNone(result)

    def test_match_above_threshold(self):
        ref = np.array([1.0, 0.0, 0.0])
        query = np.array([0.9, 0.1, 0.0])
        result = _best_match_actor(query, {"hero": ref}, 0.5)
        self.assertEqual(result, "hero")

    def test_no_match_below_threshold(self):
        ref = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, 1.0, 0.0])
        result = _best_match_actor(query, {"hero": ref}, 0.5)
        self.assertIsNone(result)

    def test_best_of_multiple(self):
        ref_a = np.array([1.0, 0.0, 0.0])
        ref_b = np.array([0.0, 1.0, 0.0])
        query = np.array([0.8, 0.2, 0.0])
        result = _best_match_actor(query, {"a": ref_a, "b": ref_b}, 0.5)
        self.assertEqual(result, "a")

    def test_cosine_similarity(self):
        ref = np.array([1.0, 1.0])
        query = np.array([1.0, 1.0])
        result = _best_match_actor(query, {"x": ref}, 0.0)
        self.assertEqual(result, "x")


class TestInsightFaceExtractor(unittest.TestCase):
    def test_detect_all_returns_empty_no_faces(self):
        extractor = InsightFaceExtractor()
        extractor._analyzer = MagicMock()
        extractor._analyzer.get.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.detect_all(frame)
        self.assertEqual(result, [])

    def test_detect_and_match_returns_empty_no_faces(self):
        extractor = InsightFaceExtractor()
        extractor._analyzer = MagicMock()
        extractor._analyzer.get.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.detect_and_match(frame, {}, threshold=0.5)
        self.assertEqual(result, [])

    def test_analyzer_lazy_init(self):
        extractor = InsightFaceExtractor()
        self.assertIsNone(extractor._analyzer)


if __name__ == "__main__":
    unittest.main()
