import unittest
from pathlib import Path
import tempfile

from render_ltx import build_arg_parser, resolve_rolling_frames, rewrite_concat_list


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

    def test_single_prompt_render_mode_is_default(self):
        args = self._parse()

        self.assertEqual("single_prompt", args.render_mode)

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

    def test_lora_1_cli_args_are_available(self):
        args = self._parse([
            "--lora-1-enabled",
            "--lora-1-name",
            "characters/test.safetensors",
            "--lora-1-strength-model",
            "0.85",
            "--lora-1-strength-clip",
            "0.65",
        ])

        self.assertTrue(args.lora_1_enabled)
        self.assertEqual("characters/test.safetensors", args.lora_1_name)
        self.assertEqual(0.85, args.lora_1_strength_model)
        self.assertEqual(0.65, args.lora_1_strength_clip)

    def test_rewrite_concat_list_includes_all_rendered_scene_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            output_dir = temp / "ltx"
            output_dir.mkdir()
            rendered = [
                output_dir / "final" / "scene_0001.mp4",
                output_dir / "final" / "scene_0002.mp4",
            ]

            concat = rewrite_concat_list(rendered, output_dir)

            text = concat.read_text(encoding="utf-8")
            self.assertIn("scene_0001.mp4", text)
            self.assertIn("scene_0002.mp4", text)


if __name__ == "__main__":
    unittest.main()
