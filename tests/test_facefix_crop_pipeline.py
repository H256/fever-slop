import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestFaceFixCropPipeline(unittest.TestCase):
    @patch("feverslop.composition.facefix_pipeline._run_crop_facefix")
    @patch("feverslop.composition.facefix_pipeline._run_legacy_facefix")
    def test_use_crop_pipeline_true(self, mock_legacy, mock_crop):
        from feverslop.composition.facefix_pipeline import run_facefix, FaceFixCompositionOptions

        mock_crop.return_value = [Path("/tmp/result.mp4")]
        options = FaceFixCompositionOptions(
            scenes_dir="/tmp/scenes",
            project_dir="/tmp/project",
            use_crop_pipeline=True,
        )
        run_facefix(options)
        mock_crop.assert_called_once()
        mock_legacy.assert_not_called()

    @patch("feverslop.composition.facefix_pipeline._run_crop_facefix")
    @patch("feverslop.composition.facefix_pipeline._run_legacy_facefix")
    def test_use_crop_pipeline_false(self, mock_legacy, mock_crop):
        from feverslop.composition.facefix_pipeline import run_facefix, FaceFixCompositionOptions

        mock_legacy.return_value = [Path("/tmp/result.mp4")]
        options = FaceFixCompositionOptions(
            scenes_dir="/tmp/scenes",
            use_crop_pipeline=False,
        )
        run_facefix(options)
        mock_legacy.assert_called_once()
        mock_crop.assert_not_called()

    def test_default_use_crop_pipeline(self):
        from feverslop.composition.facefix_pipeline import FaceFixCompositionOptions
        options = FaceFixCompositionOptions()
        self.assertTrue(options.use_crop_pipeline)


class TestLoadVideoFrames(unittest.TestCase):
    @patch("feverslop.composition.facefix_pipeline.cv2")
    def test_load_success(self, mock_cv2):
        from feverslop.composition.facefix_pipeline import _load_video_frames

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        result = _load_video_frames(Path("/tmp/video.mp4"))
        self.assertEqual(result.shape, (2, 480, 640, 3))

    @patch("feverslop.composition.facefix_pipeline.cv2")
    def test_load_fail(self, mock_cv2):
        from feverslop.composition.facefix_pipeline import _load_video_frames

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        result = _load_video_frames(Path("/tmp/missing.mp4"))
        self.assertIsNone(result)


class TestSceneArtifactLayoutExtensions(unittest.TestCase):
    def test_actor_methods(self):
        from feverslop.scene_artifacts import SceneArtifactLayout
        layout = SceneArtifactLayout(Path("/tmp/project"))

        self.assertEqual(layout.actors_dir_path(), Path("/tmp/project/output/references/actors"))

        scene_dir = layout.scene_facefix_dir(1, "hero")
        self.assertEqual(scene_dir, Path("/tmp/project/output/render/scenes/scene_0001/facefix/hero"))

        crop_mp4 = layout.scene_face_crop_mp4(1, "hero")
        self.assertEqual(crop_mp4, scene_dir / "face_crop.mp4")

        anchors = layout.scene_face_anchors_dir(1, "hero")
        self.assertEqual(anchors, scene_dir / "anchors")

        crops = layout.scene_face_crops_dir(1, "hero")
        self.assertEqual(crops, scene_dir / "crops")

        repaired = layout.scene_face_repaired_dir(1, "hero")
        self.assertEqual(repaired, scene_dir / "repaired")

    def test_actor_ids_empty(self):
        from feverslop.scene_artifacts import SceneArtifactLayout
        layout = SceneArtifactLayout(Path("/tmp/nonexistent_project"))
        self.assertEqual(layout.actor_ids(), [])


class TestFaceVideoEncoder(unittest.TestCase):
    @patch("feverslop.adapters.face_video_encoder.subprocess")
    def test_encode_success(self, mock_subprocess):
        from feverslop.adapters.face_video_encoder import encode_face_crop_mp4

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr="")
        frames_dir = Path("/tmp/crops")
        output = Path("/tmp/face_crop.mp4")

        result = encode_face_crop_mp4(frames_dir, 24.0, output)
        self.assertEqual(result, output)
        mock_subprocess.run.assert_called_once()


class TestFaceFixCropBackend(unittest.TestCase):
    def test_build_workflow_patcher(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        mock_client = MagicMock()
        mock_client.upload_file_via_image_endpoint.return_value = {"filename": "test.mp4", "subfolder": "test"}
        mock_client.upload_image.return_value = {"filename": "a.png", "subfolder": "test"}

        backend = ComfyUIFaceFixCropBackend(
            client=mock_client,
            workflow_path=Path("workflows/video_ltxv_facefix_crop.json"),
        )

        with patch.object(backend, "load_workflow") as mock_load:
            mock_load.return_value = {
                "1": {"_meta": {"title": "#LOAD_VIDEO"}, "class_type": "LoadVideo", "inputs": {"video": "", "videopath": ""}},
                "2": {"_meta": {"title": "#IMAGE_1"}, "class_type": "LoadImage", "inputs": {"image": ""}},
                "3": {"_meta": {"title": "#IMAGE_2"}, "class_type": "LoadImage", "inputs": {"image": ""}},
                "4": {"_meta": {"title": "#SAVE_VIDEO"}, "class_type": "SaveAnimatedWEBP", "inputs": {"filename_prefix": ""}},
            }

            workflow = backend.build_workflow(
                scene_number=1,
                face_crop_mp4=Path("/tmp/face_crop.mp4"),
                anchors_dir=Path("/tmp/anchors"),
                actor_id="hero",
            )
            self.assertIsInstance(workflow, dict)


if __name__ == "__main__":
    unittest.main()
