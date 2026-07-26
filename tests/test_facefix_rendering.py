import unittest
from pathlib import Path

from feverslop.domain.facefix_rendering import FaceFixConfig, FaceFixSceneRequest


class TestFaceFixConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = FaceFixConfig()
        self.assertEqual(cfg.keyframe_indices, "0,16,32,48")
        self.assertEqual(cfg.guiding_strength, 0.2)
        self.assertEqual(cfg.cond_image_strength, 0.5)
        self.assertEqual(cfg.temporal_tile_size, 56)
        self.assertEqual(cfg.temporal_overlap, 24)
        self.assertEqual(cfg.temporal_overlap_cond_strength, 0.5)
        self.assertEqual(cfg.adain_factor, 0.0)
        self.assertTrue(cfg.postprocess)

    def test_custom(self):
        cfg = FaceFixConfig(
            keyframe_indices="0,8,16",
            guiding_strength=0.3,
            cond_image_strength=0.7,
        )
        self.assertEqual(cfg.keyframe_indices, "0,8,16")
        self.assertEqual(cfg.guiding_strength, 0.3)
        self.assertEqual(cfg.cond_image_strength, 0.7)


class TestFaceFixSceneRequest(unittest.TestCase):
    def test_construction(self):
        req = FaceFixSceneRequest(
            scene_number=3,
            source_video=Path("/tmp/scene_0003.mp4"),
            reference_images=[Path("/tmp/face_01.png")],
            output_dir=Path("/tmp/output"),
        )
        self.assertEqual(req.scene_number, 3)
        self.assertEqual(req.source_video, Path("/tmp/scene_0003.mp4"))
        self.assertEqual(len(req.reference_images), 1)
        self.assertEqual(req.output_dir, Path("/tmp/output"))

    def test_no_references(self):
        req = FaceFixSceneRequest(
            scene_number=1,
            source_video=Path("/tmp/scene_0001.mp4"),
        )
        self.assertEqual(req.reference_images, ())
        self.assertEqual(req.output_dir, Path("."))


if __name__ == "__main__":
    unittest.main()
