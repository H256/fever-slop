import unittest

from feverslop.domain.minimax_h3_frames import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_DURATION_SECONDS,
    MINIMAX_H3_MIN_DURATION_SECONDS,
    MIN_FRAMES,
    _duration_from_frames,
    _frames_from_duration,
    _next_valid_17n5,
    _validate_duration,
    _validate_frames,
)
from feverslop.errors import FeverSlopValidationError


class NextValid17n5Tests(unittest.TestCase):
    """Tests for _next_valid_17n5."""

    def test_identity_for_valid_values(self):
        for val in (5, 22, 39, 107):
            with self.subTest(val=val):
                self.assertEqual(val, _next_valid_17n5(val))

    def test_rounds_up(self):
        self.assertEqual(5, _next_valid_17n5(2))
        self.assertEqual(5, _next_valid_17n5(3))
        self.assertEqual(22, _next_valid_17n5(6))

    def test_clamps_zero_and_negative(self):
        self.assertEqual(5, _next_valid_17n5(0))
        self.assertEqual(5, _next_valid_17n5(-5))
        self.assertEqual(5, _next_valid_17n5(-100))


class FramesFromDurationTests(unittest.TestCase):
    """Tests for _frames_from_duration."""

    def test_boundary_durations(self):
        self.assertEqual(107, _frames_from_duration(4.0))
        self.assertEqual(362, _frames_from_duration(15.0))

    def test_very_short_duration(self):
        self.assertEqual(5, _frames_from_duration(0.1))

    def test_mid_range(self):
        self.assertEqual(243, _frames_from_duration(10.0))

    def test_five_seconds(self):
        self.assertEqual(124, _frames_from_duration(5.0))


class DurationFromFramesTests(unittest.TestCase):
    """Tests for _duration_from_frames."""

    def test_inverse_correctness(self):
        self.assertAlmostEqual(4.458, _duration_from_frames(107), places=3)
        self.assertAlmostEqual(15.083, _duration_from_frames(362), places=3)

    def test_round_trip_fidelity(self):
        for sec in (4.0, 5.0, 10.0, 15.0):
            with self.subTest(sec=sec):
                frames = _frames_from_duration(sec)
                recovered = _duration_from_frames(frames)
                expected_frames = round(recovered * MINIMAX_H3_FPS)
                self.assertEqual(frames, expected_frames)


class ValidateDurationTests(unittest.TestCase):
    """Tests for _validate_duration."""

    def test_passes_valid_boundaries(self):
        self.assertIsNone(_validate_duration(4.0))
        self.assertIsNone(_validate_duration(10.0))
        self.assertIsNone(_validate_duration(15.0))

    def test_raises_below_minimum(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_duration(3.9)

    def test_raises_above_maximum(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_duration(15.1)


class ValidateFramesTests(unittest.TestCase):
    """Tests for _validate_frames."""

    def test_passes_valid_frame_counts(self):
        self.assertIsNone(_validate_frames(5))
        self.assertIsNone(_validate_frames(22))
        self.assertIsNone(_validate_frames(56))
        self.assertIsNone(_validate_frames(107))

    def test_raises_zero(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(0)

    def test_raises_invalid(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(1)

    def test_raises_non_aligned(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(20)

    def test_raises_17n1_values(self):
        """Values that satisfied 17N+1 but not 17N+5 should fail."""
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(17)
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(103)


class ValidFrameSequenceTests(unittest.TestCase):
    """Known valid 17N+5 frame counts."""

    def test_sequence(self):
        expected = [5, 22, 39, 56, 73, 90, 107, 124]
        for val in expected:
            with self.subTest(val=val):
                self.assertIsNone(_validate_frames(val))


class ModuleConstantsTests(unittest.TestCase):
    """Tests for module-level constants."""

    def test_fps(self):
        self.assertEqual(24, MINIMAX_H3_FPS)

    def test_min_duration(self):
        self.assertEqual(4.0, MINIMAX_H3_MIN_DURATION_SECONDS)

    def test_max_duration(self):
        self.assertEqual(15.0, MINIMAX_H3_MAX_DURATION_SECONDS)

    def test_min_frames(self):
        self.assertEqual(5, MIN_FRAMES)


if __name__ == "__main__":
    unittest.main()
