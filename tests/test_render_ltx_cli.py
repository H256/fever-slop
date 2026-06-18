import unittest
from pathlib import Path
import tempfile
import json

from render_ltx import (
    build_arg_parser,
    resolve_project_config_defaults,
    resolve_rolling_frames,
    rewrite_concat_list,
)


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

    def test_project_config_values_are_used_when_cli_omits_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_config = temp / "config.json"
            project_config.write_text(
                json.dumps(
                    {
                        "project_name": "scene_limits",
                        "input_audio": "input/song.mp3",
                        "scene_generation": {
                            "min_duration": 3.0,
                            "max_duration": 12.0,
                        },
                        "lora_1": {
                            "enabled": True,
                            "name": "characters/test.safetensors",
                            "strength_model": 0.8,
                            "strength_clip": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = build_arg_parser().parse_args([
                "--project-config",
                str(project_config),
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

            resolved = resolve_project_config_defaults(args)

            self.assertEqual(str(project_config), args.project_config)
            self.assertEqual(3.0, resolved["min_duration"])
            self.assertEqual(12.0, resolved["max_duration"])
            self.assertTrue(resolved["lora_1_enabled"])
            self.assertEqual("characters/test.safetensors", resolved["lora_1_name"])
            self.assertEqual(0.8, resolved["lora_1_strength_model"])
            self.assertEqual(0.6, resolved["lora_1_strength_clip"])

    def test_explicit_cli_values_override_project_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_config = temp / "config.json"
            project_config.write_text(
                json.dumps(
                    {
                        "project_name": "scene_limits",
                        "input_audio": "input/song.mp3",
                        "scene_generation": {
                            "min_duration": 3.0,
                            "max_duration": 12.0,
                        },
                        "lora_1": {
                            "enabled": False,
                            "name": "",
                            "strength_model": 0.8,
                            "strength_clip": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = build_arg_parser().parse_args([
                "--project-config",
                str(project_config),
                "--min-duration",
                "4.0",
                "--max-duration",
                "9.0",
                "--lora-1-enabled",
                "--lora-1-name",
                "characters/override.safetensors",
                "--lora-1-strength-model",
                "0.9",
                "--lora-1-strength-clip",
                "0.7",
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

            resolved = resolve_project_config_defaults(args)

            self.assertEqual(4.0, resolved["min_duration"])
            self.assertEqual(9.0, resolved["max_duration"])
            self.assertTrue(resolved["lora_1_enabled"])
            self.assertEqual("characters/override.safetensors", resolved["lora_1_name"])
            self.assertEqual(0.9, resolved["lora_1_strength_model"])
            self.assertEqual(0.7, resolved["lora_1_strength_clip"])

    def test_project_config_is_discovered_from_render_plan_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_dir = temp / "my_project"
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)

            project_config = project_dir / "config.json"
            project_config.write_text(
                json.dumps(
                    {
                        "project_name": "scene_limits",
                        "input_audio": "input/song.mp3",
                        "scene_generation": {
                            "min_duration": 3.0,
                            "max_duration": 12.0,
                        },
                        "lora_1": {
                            "enabled": True,
                            "name": "characters/test.safetensors",
                            "strength_model": 0.8,
                            "strength_clip": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            render_plan = render_dir / "render_plan_song.json"
            render_plan.write_text("[]", encoding="utf-8")

            args = build_arg_parser().parse_args([
                "--render-plan",
                str(render_plan),
                "--workflow",
                "workflow.json",
                "--audio",
                "song.mp3",
                "--storyboard-dir",
                "storyboard",
                "--output-dir",
                "ltx",
            ])

            resolved = resolve_project_config_defaults(args)

            self.assertEqual(3.0, resolved["min_duration"])
            self.assertEqual(12.0, resolved["max_duration"])
            self.assertTrue(resolved["lora_1_enabled"])
            self.assertEqual("characters/test.safetensors", resolved["lora_1_name"])

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
