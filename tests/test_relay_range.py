from __future__ import annotations

import unittest

from feverslop.domain.relay_range import RelayRange


class RelayRangeConstructionTests(unittest.TestCase):
    def test_length_is_end_minus_start(self):
        self.assertEqual(12, RelayRange(4, 16).length)

    def test_single_frame_length(self):
        self.assertEqual(1, RelayRange(0, 1).length)

    def test_stores_raw_values_verbatim(self):
        # The value does not bound-check; clamp() owns normalization, so raw
        # (even negative) dict input can be held before it is clamped.
        self.assertEqual(-5, RelayRange(-5, 30).start)
        self.assertEqual(30, RelayRange(-5, 30).end_exclusive)

    def test_negative_length_when_unclamped(self):
        self.assertEqual(-3, RelayRange(10, 7).length)


class RelayRangeClampTests(unittest.TestCase):
    def test_exact_fit_within_bounds(self):
        self.assertEqual(RelayRange(0, 48), RelayRange(0, 48).clamp(48))

    def test_end_clamped_to_frame_count(self):
        self.assertEqual(RelayRange(0, 48), RelayRange(0, 100).clamp(48))

    def test_start_clamped_to_zero(self):
        self.assertEqual(RelayRange(0, 30), RelayRange(-5, 30).clamp(48))

    def test_start_clamped_to_frame_count_minus_one(self):
        # start pushed to the last valid index, end forced one past it.
        self.assertEqual(RelayRange(47, 48), RelayRange(100, 200).clamp(48))

    def test_end_pulled_below_start_yields_minimal_range(self):
        # end < start: start wins, end becomes start + 1.
        self.assertEqual(RelayRange(10, 11), RelayRange(10, 2).clamp(48))

    def test_zero_frame_count_still_yields_one_frame(self):
        # Mirrors the original _clamp_relay_segment: even frame_count=0 yields
        # a single frame (start clamped to 0, end forced to start + 1).
        self.assertEqual(RelayRange(0, 1), RelayRange(0, 10).clamp(0))

    def test_clamped_result_keeps_length_positive(self):
        result = RelayRange(40, 500).clamp(48)
        self.assertIsNotNone(result)
        self.assertEqual(48, result.end_exclusive)
        self.assertEqual(8, result.length)


class RelayRangeBoundaryDictTests(unittest.TestCase):
    def test_from_dict_reads_raw_keys(self):
        data = {"frame_start": 3, "frame_end": 20, "state": "singing"}
        self.assertEqual(RelayRange(3, 20), RelayRange.from_dict(data))

    def test_to_dict_emits_raw_keys(self):
        self.assertEqual({"frame_start": 3, "frame_end": 20}, RelayRange(3, 20).to_dict())

    def test_roundtrip_preserves_values(self):
        original = RelayRange(12, 44)
        self.assertEqual(original, RelayRange.from_dict(original.to_dict()))

    def test_custom_keys(self):
        data = {"fs": 1, "fe": 9}
        self.assertEqual(RelayRange(1, 9), RelayRange.from_dict(data, start_key="fs", end_key="fe"))
        self.assertEqual({"fs": 1, "fe": 9}, RelayRange(1, 9).to_dict(start_key="fs", end_key="fe"))

    def test_equality_and_hash(self):
        self.assertEqual(RelayRange(0, 48), RelayRange(0, 48))
        self.assertEqual(len({RelayRange(0, 48), RelayRange(0, 48)}), 1)


if __name__ == "__main__":
    unittest.main()
