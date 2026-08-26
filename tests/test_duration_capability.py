import unittest

from feverslop.domain.duration_capability import DurationCapability


class DurationCapabilityTests(unittest.TestCase):
    def test_aligns_duration_to_valid_frame_count(self):
        capability = DurationCapability.create(
            fps=24, min_seconds=2, max_seconds=12, preferred_seconds=8, frame_alignment=17, frame_offset=5,
        )
        self.assertEqual(192, capability.frames_for(8.2))
        self.assertTrue(capability.validate(8.2))

    def test_rejects_out_of_range_duration(self):
        capability = DurationCapability.create(fps=24, min_seconds=2, max_seconds=12, preferred_seconds=8)
        with self.assertRaises(ValueError):
            capability.frames_for(1)
        with self.assertRaises(ValueError):
            capability.frames_for(13)
