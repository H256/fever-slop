import unittest

from render_ltx import build_arg_parser, resolve_rolling_frames


class RenderLTXCliTests(unittest.TestCase):
    def _parse(self, extra_args=None):
        extra_args = extra_args or []
        return build_arg_parser().parse_args([
            "--render-plan",
            "render_plan.json",
            "--workflow",
            "workflow.json",
            "--audio",
            "song.mp3",
            "--storyboard-dir",
            "storyboard",
            "--output-dir",
            "ltx",
            *extra_args,
        ])

    def test_original_rolling_frame_profile_is_default(self):
        args = self._parse()

        self.assertEqual((50, 25, True), resolve_rolling_frames(args))

    def test_safe_rolling_frame_profile_uses_low_vram_values(self):
        args = self._parse(["--rolling-frame-profile", "safe"])

        self.assertEqual((6, 0, False), resolve_rolling_frames(args))

    def test_explicit_rolling_frame_flags_override_profile(self):
        args = self._parse([
            "--rolling-frame-profile",
            "original",
            "--preroll-frames",
            "12",
            "--tail-loss-frames",
            "3",
        ])

        self.assertEqual((12, 3, True), resolve_rolling_frames(args))

    def test_raw_args_do_not_hide_that_rolling_values_are_profile_resolved(self):
        args = build_arg_parser().parse_args([
            "--render-plan",
            "render_plan.json",
            "--workflow",
            "workflow.json",
            "--audio",
            "song.mp3",
            "--storyboard-dir",
            "storyboard",
            "--output-dir",
            "ltx",
        ])

        self.assertIsNone(args.preroll_frames)
        self.assertIsNone(args.tail_loss_frames)


if __name__ == "__main__":
    unittest.main()
