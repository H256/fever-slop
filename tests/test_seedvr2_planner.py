import unittest

from feverslop.domain.seedvr2 import plan_seedvr2_passes, resolve_target_resolution


class SeedVR2PlannerTests(unittest.TestCase):
    def test_target_width_derives_height_preserving_aspect_ratio(self):
        self.assertEqual((3840, 2160), resolve_target_resolution(640, 360, target_width=3840))

    def test_target_height_derives_width_preserving_aspect_ratio(self):
        self.assertEqual((3840, 2160), resolve_target_resolution(640, 360, target_height=2160))

    def test_missing_target_uses_default_scale(self):
        self.assertEqual((1280, 720), resolve_target_resolution(640, 360))

    def test_auto_planner_limits_each_pass_and_reaches_target(self):
        passes = plan_seedvr2_passes(640, 360, target_width=3840, max_pass_scale=2.0, max_ai_passes=3)

        self.assertEqual([(1280, 720), (2560, 1440), (3840, 2160)], [p.output_size for p in passes])
        self.assertTrue(all(p.scale <= 2.0 + 1e-9 for p in passes))

    def test_planner_rejects_unreachable_target(self):
        with self.assertRaisesRegex(ValueError, "max_ai_passes"):
            plan_seedvr2_passes(640, 360, target_width=3840, max_pass_scale=2.0, max_ai_passes=2)
