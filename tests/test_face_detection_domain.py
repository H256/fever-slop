import unittest
from pathlib import Path

import numpy as np

from feverslop.domain.face_detection import (
    FaceBox,
    FaceCropResult,
    FaceEmbedding,
    FaceRepairData,
    FaceTrackEntry,
)


class TestFaceEmbedding(unittest.TestCase):
    def test_construction(self):
        emb = np.array([0.1, 0.2, 0.3])
        fe = FaceEmbedding(actor_id="actor1", embedding=emb)
        self.assertEqual(fe.actor_id, "actor1")
        np.testing.assert_array_almost_equal(fe.embedding, emb)

    def test_frozen(self):
        fe = FaceEmbedding(actor_id="actor1", embedding=np.zeros(512))
        with self.assertRaises(Exception):
            fe.actor_id = "actor2"


class TestFaceBox(unittest.TestCase):
    def test_construction(self):
        fb = FaceBox(x1=10, y1=20, x2=100, y2=110, confidence=0.95)
        self.assertEqual(fb.x1, 10)
        self.assertEqual(fb.y1, 20)
        self.assertEqual(fb.x2, 100)
        self.assertEqual(fb.y2, 110)
        self.assertEqual(fb.confidence, 0.95)
        self.assertIsNone(fb.actor_id)
        self.assertIsNone(fb.embedding)

    def test_with_actor(self):
        emb = np.random.rand(512)
        fb = FaceBox(x1=0, y1=0, x2=100, y2=100, confidence=0.8, actor_id="hero", embedding=emb)
        self.assertEqual(fb.actor_id, "hero")
        np.testing.assert_array_almost_equal(fb.embedding, emb)


class TestFaceTrackEntry(unittest.TestCase):
    def test_construction(self):
        box = FaceBox(x1=10, y1=10, x2=110, y2=110, confidence=0.9)
        entry = FaceTrackEntry(frame_index=5, box=box, crop_path=Path("/tmp/crop.png"))
        self.assertEqual(entry.frame_index, 5)
        self.assertEqual(entry.crop_path, Path("/tmp/crop.png"))


class TestFaceCropResult(unittest.TestCase):
    def test_construction(self):
        result = FaceCropResult(
            actor_id="hero",
            entries=[],
            anchor_paths=[Path("/tmp/anchor_0.png")],
            crop_frames_dir=Path("/tmp/crops"),
            crop_mp4_path=Path("/tmp/face_crop.mp4"),
        )
        self.assertEqual(result.actor_id, "hero")
        self.assertEqual(len(result.anchor_paths), 1)


class TestFaceRepairData(unittest.TestCase):
    def test_construction(self):
        data = FaceRepairData(
            actor_id="hero",
            repaired_frames_dir=Path("/tmp/repaired"),
            track_entries=[],
            crop_size=768,
        )
        self.assertEqual(data.crop_size, 768)


if __name__ == "__main__":
    unittest.main()
