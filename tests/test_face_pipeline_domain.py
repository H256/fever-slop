"""Tests for face pipeline domain models and functions."""

import unittest
import numpy as np

from feverslop.domain.face_detection import (
    BoundingBox,
    FaceDetection,
    FaceLandmarks,
    FaceProcessingDecision,
    FaceProcessingPolicy,
    FaceTrack,
    FrameResult,
    RejectReason,
    TrackState,
    box_iou,
    cosine_similarity,
    decide_face_processing,
    expand_box,
    filter_detections,
    is_valid_face_detection,
    normalized_center_distance,
    rank_face_candidates,
    smooth_box,
    valid_landmark_geometry,
)


class TestBoundingBox(unittest.TestCase):
    def test_basic_properties(self):
        box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        self.assertAlmostEqual(box.width, 100.0)
        self.assertAlmostEqual(box.height, 100.0)
        self.assertAlmostEqual(box.aspect_ratio, 1.0)
        self.assertAlmostEqual(box.area, 10000.0)
        self.assertEqual(box.center, (60.0, 70.0))

    def test_to_tuple(self):
        box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        self.assertEqual(box.to_tuple(), (10.0, 20.0, 110.0, 120.0))

    def test_clamp_in_frame(self):
        box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        clamped = box.clamp(320, 240)
        self.assertEqual(clamped.x1, 10.0)
        self.assertEqual(clamped.y1, 20.0)

    def test_clamp_out_of_frame(self):
        box = BoundingBox(x1=-50.0, y1=-30.0, x2=50.0, y2=30.0)
        clamped = box.clamp(320, 240)
        self.assertEqual(clamped.x1, 0.0)
        self.assertEqual(clamped.y1, 0.0)

    def test_aspect_ratio_zero_height(self):
        box = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=0.0)
        self.assertEqual(box.aspect_ratio, 0.0)

    def test_area_negative(self):
        box = BoundingBox(x1=100.0, y1=100.0, x2=0.0, y2=0.0)
        self.assertEqual(box.area, 0.0)


class TestFaceLandmarks(unittest.TestCase):
    def test_valid_5_points(self):
        lm = FaceLandmarks(points=[(0, 0), (10, 0), (5, 10), (2, 20), (8, 20)])
        self.assertEqual(len(lm.points), 5)

    def test_invalid_point_count(self):
        with self.assertRaises(AssertionError):
            FaceLandmarks(points=[(0, 0), (10, 0)])


class TestValidLandmarkGeometry(unittest.TestCase):
    def test_valid_geometry(self):
        lm = FaceLandmarks(points=[
            (50.0, 50.0),   # left eye
            (150.0, 50.0),  # right eye
            (100.0, 100.0), # nose
            (70.0, 150.0),  # left mouth
            (130.0, 150.0), # right mouth
        ])
        self.assertTrue(valid_landmark_geometry(lm))

    def test_collapsed_eyes(self):
        lm = FaceLandmarks(points=[
            (100.0, 50.0),
            (102.0, 50.0),  # too close
            (100.0, 100.0),
            (90.0, 150.0),
            (110.0, 150.0),
        ])
        self.assertFalse(valid_landmark_geometry(lm))

    def test_inverted_vertical(self):
        lm = FaceLandmarks(points=[
            (50.0, 150.0),  # eyes below nose
            (150.0, 150.0),
            (100.0, 50.0),  # nose above eyes
            (70.0, 100.0),
            (130.0, 100.0),
        ])
        self.assertFalse(valid_landmark_geometry(lm))

    def test_nan_point(self):
        lm = FaceLandmarks(points=[
            (float('nan'), 50.0),
            (150.0, 50.0),
            (100.0, 100.0),
            (70.0, 150.0),
            (130.0, 150.0),
        ])
        self.assertFalse(valid_landmark_geometry(lm))


class TestBoxIoU(unittest.TestCase):
    def test_identical_boxes(self):
        box = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_no_overlap(self):
        a = BoundingBox(x1=0.0, y1=0.0, x2=50.0, y2=50.0)
        b = BoundingBox(x1=60.0, y1=60.0, x2=110.0, y2=110.0)
        self.assertAlmostEqual(box_iou(a, b), 0.0)

    def test_partial_overlap(self):
        a = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        b = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0)
        self.assertGreater(box_iou(a, b), 0.0)
        self.assertLess(box_iou(a, b), 1.0)

    def test_zero_area(self):
        a = BoundingBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0)
        b = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        self.assertAlmostEqual(box_iou(a, b), 0.0)


class TestNormalizedCenterDistance(unittest.TestCase):
    def test_identical(self):
        box = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        self.assertAlmostEqual(
            normalized_center_distance(box, box, 320, 240), 0.0
        )

    def test_different(self):
        a = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        b = BoundingBox(x1=200.0, y1=100.0, x2=300.0, y2=200.0)
        dist = normalized_center_distance(a, b, 320, 240)
        self.assertGreater(dist, 0.0)

    def test_zero_frame(self):
        box = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        self.assertAlmostEqual(
            normalized_center_distance(box, box, 0, 0), 0.0
        )


class TestIsValidFaceDetection(unittest.TestCase):
    def _make_detection(self, score=0.9, x1=50, y1=50, x2=150, y2=150):
        return FaceDetection(
            box=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            score=score,
            landmarks=FaceLandmarks(points=[
                (60.0, 60.0), (140.0, 60.0), (100.0, 100.0),
                (70.0, 140.0), (130.0, 140.0),
            ]),
        )

    def test_valid_detection(self):
        det = self._make_detection()
        self.assertTrue(is_valid_face_detection(det, 320, 240, FaceProcessingPolicy()))

    def test_low_score(self):
        det = self._make_detection(score=0.1)
        self.assertFalse(is_valid_face_detection(det, 320, 240, FaceProcessingPolicy()))

    def test_face_too_small(self):
        det = self._make_detection(x1=50, y1=50, x2=60, y2=60)
        self.assertFalse(is_valid_face_detection(det, 320, 240, FaceProcessingPolicy()))

    def test_invalid_landmarks(self):
        det = self._make_detection()
        bad_lm = FaceLandmarks(points=[
            (100.0, 50.0), (102.0, 50.0), (100.0, 100.0),
            (90.0, 150.0), (110.0, 150.0),
        ])
        det = FaceDetection(box=det.box, score=det.score, landmarks=bad_lm)
        self.assertFalse(is_valid_face_detection(det, 320, 240, FaceProcessingPolicy()))

    def test_nan_coordinates(self):
        det = FaceDetection(
            box=BoundingBox(x1=float('nan'), y1=50.0, x2=150.0, y2=150.0),
            score=0.9,
        )
        self.assertFalse(is_valid_face_detection(det, 320, 240, FaceProcessingPolicy()))

    def test_outside_frame(self):
        det = FaceDetection(
            box=BoundingBox(x1=400.0, y1=300.0, x2=500.0, y2=400.0),
            score=0.9,
        )
        self.assertFalse(is_valid_face_detection(det, 320, 240, FaceProcessingPolicy()))


class TestFilterDetections(unittest.TestCase):
    def _make_valid(self):
        return FaceDetection(
            box=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0),
            score=0.9,
            landmarks=FaceLandmarks(points=[
                (60.0, 60.0), (140.0, 60.0), (100.0, 100.0),
                (70.0, 140.0), (130.0, 140.0),
            ]),
        )

    def _make_invalid(self):
        return FaceDetection(
            box=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
            score=0.1,
        )

    def test_filters_correctly(self):
        valid = self._make_valid()
        invalid = self._make_invalid()
        result = filter_detections([valid, invalid], 320, 240, FaceProcessingPolicy())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], valid)

    def test_empty_input(self):
        result = filter_detections([], 320, 240, FaceProcessingPolicy())
        self.assertEqual(result, [])


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0)


class TestDecideFaceProcessing(unittest.TestCase):
    def _make_detection(self):
        return FaceDetection(
            box=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0),
            score=0.9,
            landmarks=FaceLandmarks(points=[
                (60.0, 60.0), (140.0, 60.0), (100.0, 100.0),
                (70.0, 140.0), (130.0, 140.0),
            ]),
        )

    def _make_track(self):
        det = self._make_detection()
        return FaceTrack(
            track_id=1, state=TrackState.CONFIRMED,
            box=det.box, smoothed_box=det.box,
            detection_score=det.score, identity_score=None,
            confirmed_frames=3, missing_frames=0, last_frame_index=0,
            current_detection=det,
        )

    def test_no_detection(self):
        decision = decide_face_processing(None, self._make_track(), None, FaceProcessingPolicy())
        self.assertFalse(decision.should_process)
        self.assertEqual(decision.reject_reason, RejectReason.NO_DETECTION)

    def test_low_score(self):
        det = FaceDetection(box=BoundingBox(x1=50, y1=50, x2=150, y2=150), score=0.1)
        decision = decide_face_processing(det, self._make_track(), None, FaceProcessingPolicy())
        self.assertFalse(decision.should_process)
        self.assertEqual(decision.reject_reason, RejectReason.LOW_DETECTION_SCORE)

    def test_unconfirmed_track(self):
        track = FaceTrack(
            track_id=1, state=TrackState.UNCONFIRMED,
            box=BoundingBox(x1=50, y1=50, x2=150, y2=150),
            smoothed_box=BoundingBox(x1=50, y1=50, x2=150, y2=150),
            detection_score=0.9, identity_score=None,
            confirmed_frames=1, missing_frames=0, last_frame_index=0,
        )
        decision = decide_face_processing(self._make_detection(), track, None, FaceProcessingPolicy())
        self.assertFalse(decision.should_process)
        self.assertEqual(decision.reject_reason, RejectReason.TRACK_NOT_CONFIRMED)

    def test_accepted(self):
        decision = decide_face_processing(
            self._make_detection(), self._make_track(), None, FaceProcessingPolicy()
        )
        self.assertTrue(decision.should_process)
        self.assertIsNone(decision.reject_reason)

    def test_identity_mismatch(self):
        policy = FaceProcessingPolicy(
            enable_identity_check=True, min_identity_score=0.80
        )
        decision = decide_face_processing(
            self._make_detection(), self._make_track(), 0.5, policy
        )
        self.assertFalse(decision.should_process)
        self.assertEqual(decision.reject_reason, RejectReason.IDENTITY_MISMATCH)


class TestExpandBox(unittest.TestCase):
    def test_expand(self):
        box = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0)
        expanded = expand_box(box, 1.5, 320, 240)
        self.assertGreater(expanded.width, box.width)
        self.assertGreater(expanded.height, box.height)

    def test_expand_clamped(self):
        box = BoundingBox(x1=300.0, y1=200.0, x2=350.0, y2=250.0)
        expanded = expand_box(box, 2.0, 320, 240)
        self.assertLessEqual(expanded.x2, 320.0)
        self.assertLessEqual(expanded.y2, 240.0)


class TestSmoothBox(unittest.TestCase):
    def test_same_box(self):
        box = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0)
        smoothed = smooth_box(box, box, alpha=0.7)
        self.assertEqual(smoothed.x1, box.x1)

    def test_smoothing(self):
        prev = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        curr = BoundingBox(x1=10.0, y1=10.0, x2=110.0, y2=110.0)
        smoothed = smooth_box(prev, curr, alpha=0.7)
        self.assertAlmostEqual(smoothed.x1, 3.0)  # 0.7*0 + 0.3*10


class TestFaceProcessingDecision(unittest.TestCase):
    def test_accept(self):
        det = FaceDetection(
            box=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0),
            score=0.9,
        )
        decision = FaceProcessingDecision.accept(det, 1, 0.9, 0.85)
        self.assertTrue(decision.should_process)
        self.assertEqual(decision.track_id, 1)
        self.assertEqual(decision.detection_score, 0.9)
        self.assertEqual(decision.identity_score, 0.85)
        self.assertIsNone(decision.reject_reason)

    def test_reject(self):
        decision = FaceProcessingDecision.reject(RejectReason.NO_DETECTION)
        self.assertFalse(decision.should_process)
        self.assertEqual(decision.reject_reason, RejectReason.NO_DETECTION)


class TestFrameResult(unittest.TestCase):
    def test_unchanged(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = FrameResult.unchanged(frame, RejectReason.NO_DETECTION)
        self.assertFalse(result.processed)
        self.assertEqual(result.reject_reason, RejectReason.NO_DETECTION)
        self.assertIsNone(result.track_id)

    def test_processed(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        box = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0)
        expanded = BoundingBox(x1=0.0, y1=0.0, x2=200.0, y2=200.0)
        result = FrameResult.processed(frame, 0.9, 0.85, 1, box, expanded)
        self.assertTrue(result.processed)
        self.assertEqual(result.track_id, 1)
        self.assertIsNone(result.reject_reason)


class TestRankFaceCandidates(unittest.TestCase):
    def _make_detection(self, score=0.9, x1=50, y1=50, x2=150, y2=150):
        return FaceDetection(
            box=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            score=score,
            landmarks=FaceLandmarks(points=[
                (60.0, 60.0), (140.0, 60.0), (100.0, 100.0),
                (70.0, 140.0), (130.0, 140.0),
            ]),
        )

    def test_empty(self):
        result = rank_face_candidates([], 320, 240)
        self.assertEqual(result, [])

    def test_single(self):
        det = self._make_detection()
        result = rank_face_candidates([det], 320, 240)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].detection, det)

    def test_ranking(self):
        high = self._make_detection(score=0.99)
        low = self._make_detection(score=0.5, x1=10, y1=10, x2=30, y2=30)
        result = rank_face_candidates([low, high], 320, 240)
        self.assertEqual(result[0].detection, high)


if __name__ == "__main__":
    unittest.main()
