import unittest

from feverslop.domain.continuation_segments import split_semantic_action
from feverslop.domain.continuity import BoundaryFrameManifest
from feverslop.domain.duration_capability import DurationCapability


class ContinuationSegmentTests(unittest.TestCase):
    def test_splits_long_action_with_stable_ids_and_exact_coverage(self):
        segments = split_semantic_action(
            action_id="ritual",
            start_seconds=10.0,
            duration_seconds=25.0,
            max_duration_seconds=12.0,
            fps=24,
        )

        self.assertEqual(["ritual-0001", "ritual-0002", "ritual-0003"], [item.segment_id for item in segments])
        self.assertEqual([False, True, True], [item.starts_with_anchor for item in segments])
        self.assertEqual(10.0, segments[0].start_seconds)
        self.assertEqual(35.0, segments[-1].end_seconds)
        self.assertAlmostEqual(25.0, sum(item.duration_seconds for item in segments), places=5)
        self.assertTrue(all(item.duration_seconds <= 12.0 for item in segments))

    def test_rebalances_short_tail_and_is_deterministic(self):
        kwargs = {
            "action_id": "walk",
            "start_seconds": 0.0,
            "duration_seconds": 13.0,
            "max_duration_seconds": 12.0,
            "fps": 24,
            "min_duration_seconds": 4.0,
        }

        first = split_semantic_action(**kwargs)
        second = split_semantic_action(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual([6.5, 6.5], [item.duration_seconds for item in first])

    def test_uses_profile_fps_limits_and_alignment(self):
        capability = DurationCapability.create(
            fps=24, min_seconds=2, max_seconds=12, preferred_seconds=8,
            frame_alignment=17, frame_offset=5,
        )
        segments = split_semantic_action(
            action_id="ritual", start_seconds=10, duration_seconds=25,
            max_duration_seconds=99, fps=50, capability=capability,
        )

        self.assertEqual(24, round(sum(item.duration_seconds for item in segments) / 25 * 24))
        self.assertEqual(10, segments[0].start_seconds)
        self.assertEqual(35, segments[-1].end_seconds)
        self.assertTrue(all(item.duration_seconds <= 12 for item in segments))
        self.assertEqual(5, round(segments[0].duration_seconds * 24) % 17)
        self.assertTrue(all(round(item.duration_seconds * 24) % 17 == 0 for item in segments[1:]))

    def test_rejects_impossible_or_invalid_inputs(self):
        with self.assertRaises(ValueError):
            split_semantic_action(
                action_id="x", start_seconds=0, duration_seconds=1, max_duration_seconds=0, fps=24
            )
        with self.assertRaises(ValueError):
            split_semantic_action(
                action_id="x", start_seconds=0, duration_seconds=3, max_duration_seconds=2, fps=24, min_duration_seconds=2
            )

    def test_boundary_manifest_roundtrip_and_staleness_check(self):
        manifest = BoundaryFrameManifest.create(
            source_clip_path="output/render/scenes/scene_0001/final.mp4",
            source_clip_sha256="a" * 64,
            frame_index=287,
            extractor_revision="last-frame-v2",
            frame_path="output/render/scenes/scene_0001/lastframe.png",
            frame_sha256="b" * 64,
        )

        self.assertEqual(manifest, BoundaryFrameManifest.from_dict(manifest.to_dict()))
        self.assertTrue(manifest.matches(source_clip_sha256="a" * 64, frame_sha256="b" * 64))
        self.assertFalse(manifest.matches(source_clip_sha256="c" * 64, frame_sha256="b" * 64))


if __name__ == "__main__":
    unittest.main()
