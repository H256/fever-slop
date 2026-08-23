import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from feverslop.adapters.insightface_tracker import InsightFaceTracker
from feverslop.domain.face_detection import FaceBox, FaceCropResult


class TestInsightFaceTracker(unittest.TestCase):
    def test_track_video_no_frames(self):
        extractor = MagicMock()
        tracker = InsightFaceTracker(extractor)

        with patch("feverslop.adapters.insightface_tracker.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            with self.assertRaises(ValueError):
                tracker.track_video(Path("/tmp/video.mp4"), {})

    def test_track_video_empty_result(self):
        extractor = MagicMock()
        extractor.detect_all.return_value = []
        tracker = InsightFaceTracker(extractor)

        with patch("feverslop.adapters.insightface_tracker.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: 24.0 if x == 5 else 0 if x == 7 else None
            mock_cap.read.side_effect = [(False, None)]
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.CAP_PROP_FPS = 5
            mock_cv2.CAP_PROP_FRAME_COUNT = 7

            result = tracker.track_video(
                Path("/tmp/video.mp4"),
                {},
                crop_size=256,
                anchor_interval=16,
                output_dir=Path("/tmp/facefix"),
            )

            self.assertIsInstance(result, FaceCropResult)
            self.assertEqual(len(result.entries), 0)

    def test_track_video_releases_capture_when_processing_fails(self):
        extractor = MagicMock()
        extractor.detect_all.side_effect = RuntimeError("detector failed")
        tracker = InsightFaceTracker(extractor)

        with patch("feverslop.adapters.insightface_tracker.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 24.0
            mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FPS = 5
            mock_cv2.CAP_PROP_FRAME_COUNT = 7

            with self.assertRaisesRegex(RuntimeError, "detector failed"):
                tracker.track_video(Path("/tmp/video.mp4"), {})

            mock_cap.release.assert_called_once_with()


class TestInsightFaceTrackerAnchorSelection(unittest.TestCase):
    def test_anchor_interval(self):
        ref_emb = np.ones(512)
        face_emb = np.ones(512)
        extractor = MagicMock()
        extractor.detect_all.return_value = [
            FaceBox(x1=10, y1=10, x2=100, y2=100, confidence=0.9, embedding=face_emb),
        ]
        tracker = InsightFaceTracker(extractor)

        with patch("feverslop.adapters.insightface_tracker.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: 24.0 if x == 5 else 33.0 if x == 7 else None
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cap.read.side_effect = [(True, frame_data)] * 33 + [(False, None)]
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            mock_cv2.imwrite.return_value = True
            mock_cv2.CAP_PROP_FPS = 5
            mock_cv2.CAP_PROP_FRAME_COUNT = 7

            result = tracker.track_video(
                Path("/tmp/video.mp4"),
                {"hero": ref_emb},
                crop_size=64,
                anchor_interval=16,
                output_dir=Path("/tmp/facefix"),
                actor_id="hero",
            )

            self.assertGreater(len(result.anchor_paths), 0)


if __name__ == "__main__":
    unittest.main()
