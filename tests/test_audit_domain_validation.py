import math
import unittest


class AuditDomainValidationTests(unittest.TestCase):
    def test_timeline_segment_rejects_invalid_bounds_and_non_finite_values(self):
        from feverslop.domain.timeline import TimelineSegment

        with self.assertRaises(ValueError):
            TimelineSegment(start=5.0, end=2.0, kind="vocals")
        with self.assertRaises(ValueError):
            TimelineSegment(start=math.nan, end=2.0, kind="vocals")

    def test_srt_timestamp_accepts_semicolon_millisecond_separator(self):
        from feverslop.domain.srt import parse_srt_timestamp

        self.assertAlmostEqual(61.5, parse_srt_timestamp("00:01:01;500"))


if __name__ == "__main__":
    unittest.main()
