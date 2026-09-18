from __future__ import annotations

import unittest

from feverslop.tools.render_plan_normalizer import (
    frame_count_from_duration,
    normalize_render_plan,
    scene_duration_from_frame_count,
)

FPS = 24


def _scene(number: int, start: float, end: float, frame_count: int, relays: list[dict]) -> dict:
    return {
        "scene": number,
        "abs_start_seconds": start,
        "abs_end_seconds": end,
        "duration_seconds": round(end - start, 6),
        "frame_count": frame_count,
        "fps": FPS,
        "metadata": {"type": "singing"},
        "ltx": {"base_prompt": "base", "prompt_relay": relays},
        "z_image": {},
    }


class NormalizerFrameMathTests(unittest.TestCase):
    def test_frame_count_rounds_up_plus_one(self):
        self.assertEqual(25, frame_count_from_duration(1.0, FPS))

    def test_scene_duration_uses_frames_minus_one(self):
        self.assertEqual(1.0, scene_duration_from_frame_count(25, FPS))


class NormalizerValidationTests(unittest.TestCase):
    def test_empty_plan_returns_empty(self):
        self.assertEqual([], normalize_render_plan([], 0.5, 10.0))

    def test_min_duration_must_be_positive(self):
        with self.assertRaises(ValueError):
            normalize_render_plan([{"fps": FPS}], 0.0, 10.0)

    def test_max_duration_must_be_at_least_min(self):
        with self.assertRaises(ValueError):
            normalize_render_plan([{"fps": FPS}], 5.0, 1.0)


class NormalizerRelayClampTests(unittest.TestCase):
    def test_merged_relays_offset_and_clamp_to_timeline(self):
        # min_duration=1.0 -> min_frames=25, so scene 1 (13 frames) does NOT
        # reach the threshold alone and accumulates with scene 2 into one
        # group. _merge_group offsets each scene's relays by cursor_frames
        # and clamps them to the merged timeline (frame_count - 1 = 36).
        plan = [
            _scene(1, 0.0, 0.5, 13, [
                {"frame_start": 5, "frame_end": 40, "prompt": "a"},
            ]),
            _scene(2, 0.5, 1.5, 25, [
                {"frame_start": 30, "frame_end": 90, "prompt": "b"},
            ]),
        ]
        result = normalize_render_plan(plan, 1.0, 10.0, renumber=False)
        self.assertEqual(1, len(result))
        merged = result[0]
        self.assertEqual(37, merged["frame_count"])
        relays = merged["ltx"]["prompt_relay"]
        self.assertEqual(2, len(relays))
        # Scene 1 relay: no offset (cursor=0), end clamped to 36.
        self.assertEqual(5, relays[0]["frame_start"])
        self.assertEqual(36, relays[0]["frame_end"])
        # Scene 2 relay: offset by cursor_frames=12 -> start 42 clamped to 36,
        # end clamped to 36 with floor start+1 -> 37.
        self.assertEqual(36, relays[1]["frame_start"])
        self.assertEqual(37, relays[1]["frame_end"])

    def test_single_scene_relay_within_bounds_is_preserved(self):
        plan = [
            _scene(1, 0.0, 1.0, 25, [
                {"frame_start": 5, "frame_end": 20, "prompt": "a"},
            ]),
        ]
        result = normalize_render_plan(plan, 0.5, 10.0, renumber=False)
        self.assertEqual(1, len(result))
        relays = result[0]["ltx"]["prompt_relay"]
        self.assertEqual(5, relays[0]["frame_start"])
        self.assertEqual(20, relays[0]["frame_end"])


if __name__ == "__main__":
    unittest.main()
