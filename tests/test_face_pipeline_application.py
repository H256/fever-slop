"""Tests for face pipeline application layer with fake ports."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from feverslop.domain.face_detection import (
    BoundingBox,
    FaceDetection,
    FaceLandmarks,
    FaceProcessingPolicy,
    RejectReason,
)
from feverslop.application.face_pipeline import FacePipeline


class FakeDetectorPort:
    def __init__(self, detections=None):
        self.detections = detections or []
        self.call_count = 0

    def detect_faces(self, frame):
        self.call_count += 1
        return self.detections

    def extract_embedding(self, frame, box):
        return np.random.rand(512)


class FakeIdentityPort:
    def __init__(self, actor_id="actor1", similarity=0.85):
        self.actor_id = actor_id
        self.similarity = similarity
        self.call_count = 0

    def verify_identity(self, detection_embedding):
        self.call_count += 1
        return self.actor_id, self.similarity

    def register_reference(self, embedding, actor_id):
        pass

    def get_actor_embedding(self, actor_id):
        return None

    def get_all_embeddings(self):
        return []


class FakeMaskPort:
    def __init__(self):
        self.call_count = 0

    def generate_mask(self, frame, box, landmarks=None, feather_radius=16):
        self.call_count += 1
        h, w = frame.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255

    def smooth_mask_temporal(self, previous_mask, current_mask, alpha=0.70):
        return current_mask

    def validate_mask(self, mask, min_nonzero_ratio=0.01):
        return True


class FakeDebugPort:
    def __init__(self):
        self.images = []

    def write_debug_image(self, frame_index, image, label):
        self.images.append((frame_index, label, image))
        return MagicMock()

    def write_detection_overlay(self, frame_index, frame, detections, decision_reason, extra_info=None):
        self.images.append((frame_index, "detection", frame))
        return MagicMock()

    def write_crop(self, frame_index, crop, label="crop"):
        return MagicMock()

    def write_mask(self, frame_index, mask, label="mask"):
        return MagicMock()

    def write_composite(self, frame_index, original, processed, mask):
        return MagicMock()


class TestFacePipeline(unittest.TestCase):
    def _make_detection(self, score=0.9):
        return FaceDetection(
            box=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0),
            score=score,
            landmarks=FaceLandmarks(points=[
                (60.0, 60.0), (140.0, 60.0), (100.0, 100.0),
                (70.0, 140.0), (130.0, 140.0),
            ]),
        )

    def test_no_detection_returns_unchanged(self):
        detector = FakeDetectorPort(detections=[])
        pipeline = FacePipeline(
            detector=detector,
            identity_port=FakeIdentityPort(),
            mask_port=FakeMaskPort(),
            debug_port=None,
            policy=FaceProcessingPolicy(debug_output=False),
        )
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, 0)

        self.assertFalse(result.processed)
        self.assertEqual(result.reject_reason, RejectReason.NO_DETECTION)
        self.assertTrue(np.array_equal(result.frame, frame))

    def test_unconfirmed_track_rejects(self):
        det = self._make_detection()
        detector = FakeDetectorPort(detections=[det])
        pipeline = FacePipeline(
            detector=detector,
            identity_port=FakeIdentityPort(),
            mask_port=FakeMaskPort(),
            debug_port=None,
            policy=FaceProcessingPolicy(
                track_confirmation_frames=3, debug_output=False
            ),
        )
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Frame 0: track unconfirmed
        result = pipeline.process_frame(frame, 0)
        self.assertFalse(result.processed)
        self.assertEqual(result.reject_reason, RejectReason.TRACK_NOT_CONFIRMED)

        # Frame 1: still unconfirmed
        result = pipeline.process_frame(frame, 1)
        self.assertFalse(result.processed)

        # Frame 2: should be confirmed
        result = pipeline.process_frame(frame, 2)
        self.assertTrue(result.processed)

    def test_low_score_rejects(self):
        det = self._make_detection(score=0.1)
        detector = FakeDetectorPort(detections=[det])
        pipeline = FacePipeline(
            detector=detector,
            identity_port=FakeIdentityPort(),
            mask_port=FakeMaskPort(),
            debug_port=None,
            policy=FaceProcessingPolicy(debug_output=False),
        )
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, 0)

        self.assertFalse(result.processed)
        self.assertEqual(result.reject_reason, RejectReason.NO_DETECTION)

    def test_identity_mismatch_rejects(self):
        det = self._make_detection()
        det_with_emb = FaceDetection(
            box=det.box, score=det.score, landmarks=det.landmarks,
            embedding=np.random.rand(512),
        )
        detector = FakeDetectorPort(detections=[det_with_emb])
        identity = FakeIdentityPort(similarity=0.3)
        pipeline = FacePipeline(
            detector=detector,
            identity_port=identity,
            mask_port=FakeMaskPort(),
            debug_port=None,
            policy=FaceProcessingPolicy(
                enable_identity_check=True,
                min_identity_score=0.70,
                track_confirmation_frames=2,
                debug_output=False,
            ),
        )
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # First frame: track unconfirmed
        pipeline.process_frame(frame, 0)
        # Second frame: confirmed but identity mismatch
        result = pipeline.process_frame(frame, 1)
        self.assertFalse(result.processed)
        self.assertEqual(result.reject_reason, RejectReason.IDENTITY_MISMATCH)

    def test_debug_output(self):
        det = self._make_detection()
        detector = FakeDetectorPort(detections=[det])
        debug_port = FakeDebugPort()
        pipeline = FacePipeline(
            detector=detector,
            identity_port=FakeIdentityPort(),
            mask_port=FakeMaskPort(),
            debug_port=debug_port,
            policy=FaceProcessingPolicy(debug_output=True),
        )
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame, 0)

        self.assertTrue(len(debug_port.images) > 0)

    def test_reset(self):
        det = self._make_detection()
        detector = FakeDetectorPort(detections=[det])
        pipeline = FacePipeline(
            detector=detector,
            identity_port=FakeIdentityPort(),
            mask_port=FakeMaskPort(),
            debug_port=None,
            policy=FaceProcessingPolicy(debug_output=False),
        )
        # Process some frames
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for i in range(5):
            pipeline.process_frame(frame, i)

        # Reset
        pipeline.reset()
        self.assertEqual(pipeline._next_track_id, 0)
        self.assertEqual(len(pipeline._tracks), 0)


if __name__ == "__main__":
    unittest.main()
