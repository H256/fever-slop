import unittest

from feverslop.domain.minimax_h3_frames import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_DURATION_SECONDS,
    MINIMAX_H3_MIN_DURATION_SECONDS,
    _duration_from_frames,
    _frames_from_duration,
    _next_valid_17n1,
    _validate_duration,
    _validate_frames,
)
from feverslop.errors import FeverSlopValidationError


class NextValid17n1Tests(unittest.TestCase):
    """Tests for _next_valid_17n1."""

    def test_identity_for_valid_values(self):
        for val in (1, 18, 35, 103):
            with self.subTest(val=val):
                self.assertEqual(val, _next_valid_17n1(val))

    def test_rounds_up(self):
        self.assertEqual(18, _next_valid_17n1(2))
        self.assertEqual(18, _next_valid_17n1(17))
        self.assertEqual(35, _next_valid_17n1(19))

    def test_clamps_zero_and_negative(self):
        self.assertEqual(1, _next_valid_17n1(0))
        self.assertEqual(1, _next_valid_17n1(-5))
        self.assertEqual(1, _next_valid_17n1(-100))


class FramesFromDurationTests(unittest.TestCase):
    """Tests for _frames_from_duration."""

    def test_boundary_durations(self):
        self.assertEqual(103, _frames_from_duration(4.0))
        self.assertEqual(375, _frames_from_duration(15.0))

    def test_very_short_duration(self):
        self.assertEqual(18, _frames_from_duration(0.1))

    def test_mid_range(self):
        self.assertEqual(256, _frames_from_duration(10.0))

    def test_fractional_seconds(self):
        self.assertEqual(120, _frames_from_duration(4.55))
        self.assertEqual(35, _frames_from_duration(1.25))


class DurationFromFramesTests(unittest.TestCase):
    """Tests for _duration_from_frames."""

    def test_inverse_correctness(self):
        self.assertAlmostEqual(4.292, _duration_from_frames(103), places=3)
        self.assertAlmostEqual(15.625, _duration_from_frames(375), places=3)

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
        self.assertIsNone(_validate_frames(1))
        self.assertIsNone(_validate_frames(18))
        self.assertIsNone(_validate_frames(103))

    def test_raises_zero(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(0)

    def test_raises_invalid_17n1(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(17)

    def test_raises_non_aligned(self):
        with self.assertRaises(FeverSlopValidationError):
            _validate_frames(20)


class ModuleConstantsTests(unittest.TestCase):
    """Tests for module-level constants."""

    def test_fps(self):
        self.assertEqual(24, MINIMAX_H3_FPS)

    def test_min_duration(self):
        self.assertEqual(4.0, MINIMAX_H3_MIN_DURATION_SECONDS)

    def test_max_duration(self):
        self.assertEqual(15.0, MINIMAX_H3_MAX_DURATION_SECONDS)


if __name__ == "__main__":
    unittest.main()
