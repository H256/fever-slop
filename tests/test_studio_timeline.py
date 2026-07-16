import unittest


class StudioTimelineTests(unittest.TestCase):
    def test_normalize_trim_clamps_and_preserves_minimum_duration(self):
        from feverslop.studio.desktop.timeline import normalize_trim

        self.assertEqual(normalize_trim(-1.0, 12.0, duration=10.0), (0.0, 10.0))
        self.assertEqual(normalize_trim(5.0, 4.0, duration=10.0), (5.0, 5.04))

    def test_format_timestamp_uses_editor_clock(self):
        from feverslop.studio.desktop.timeline import format_timestamp

        self.assertEqual(format_timestamp(65.125), "01:05.125")
