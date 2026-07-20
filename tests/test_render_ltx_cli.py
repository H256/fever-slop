import unittest
from pathlib import Path
import tempfile
import json

from render_ltx import (
    build_arg_parser,
    final_concat_paths,
    resolve_project_config_defaults,
    resolve_rolling_frames,
    rewrite_concat_list,
    sanitize_file_stem,
)
from feverslop.composition.render_video import namespace_to_options
from feverslop.errors import FeverSlopValidationError


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

    def test_rolling_frame_profiles_delegate_to_render_video_composition(self):
        import render_ltx
        import feverslop.composition.render_video as render_video_composition
        import feverslop.domain.ltx_rendering as ltx_rendering

        self.assertIs(render_ltx.ROLLING_FRAME_PROFILES, render_video_composition.ROLLING_FRAME_PROFILES)
        self.assertIs(render_video_composition.ROLLING_FRAME_PROFILES, ltx_rendering.ROLLING_FRAME_PROFILES)

    def test_invalid_direct_rolling_frame_profile_raises_contextual_validation_error(self):
        args = self._parse()
        args.rolling_frame_profile = "unknown"

        with self.assertRaisesRegex(
            FeverSlopValidationError,
            "rolling_frame_profile.*unknown.*off.*original.*safe",
        ):
            resolve_rolling_frames(args)

    def test_single_prompt_render_mode_is_default(self):
        args = self._parse()

        self.assertEqual("single_prompt", args.render_mode)
        self.assertEqual("ltx_i2v", args.video_pipeline)

    def test_video_pipeline_ltx_msr_is_forwarded_to_composition_options(self):
        args = self._parse(["--video-pipeline", "ltx_msr"])

        self.assertEqual("ltx_msr", args.video_pipeline)
        self.assertEqual("ltx_msr", namespace_to_options(args).video_pipeline)

    def test_debug_flag_enables_ffmpeg_debug_output(self):
        args = self._parse(["--debug"])

        self.assertTrue(args.debug)
        self.assertTrue(namespace_to_options(args).ffmpeg_debug)

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

    def test_lora_1_strength_cli_values_are_marked_explicit_without_enabling_name_patch(self):
        args = self._parse([
            "--lora-1-strength-model",
            "0",
            "--lora-1-strength-clip",
            "0",
        ])

        resolved = resolve_project_config_defaults(args)

        self.assertFalse(resolved["lora_1_enabled"])
        self.assertTrue(resolved["lora_1_strengths_explicit"])
        self.assertEqual(0.0, resolved["lora_1_strength_model"])
        self.assertEqual(0.0, resolved["lora_1_strength_clip"])

    def test_lora_1_enabled_does_not_require_name_when_only_strengths_are_overridden(self):
        args = self._parse([
            "--lora-1-enabled",
            "--lora-1-strength-model",
            "0.5",
        ])

        resolved = resolve_project_config_defaults(args)

        self.assertTrue(resolved["loras"][0].enabled)
        self.assertEqual("", resolved["loras"][0].name)
        self.assertFalse(resolved["loras"][0].name_explicit)
        self.assertTrue(resolved["loras"][0].strength_model_explicit)
        self.assertFalse(resolved["loras"][0].strength_clip_explicit)

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
            self.assertFalse(resolved["lora_split_enabled"])
            self.assertEqual(1, len(resolved["loras"]))
            self.assertEqual(1, resolved["loras"][0].index)
            self.assertEqual("characters/test.safetensors", resolved["loras"][0].name)

    def test_project_config_loras_array_and_split_flag_are_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_config = temp / "config.json"
            project_config.write_text(
                json.dumps(
                    {
                        "project_name": "multi_lora",
                        "input_audio": "input/song.mp3",
                        "lora_split_enabled": True,
                        "loras": [
                            {
                                "enabled": True,
                                "name": "characters/first.safetensors",
                                "strength_model": 0.8,
                                "strength_clip": 0.6,
                            },
                            {
                                "enabled": False,
                                "name": "characters/second.safetensors",
                                "strength_model": 0.4,
                                "strength_clip": 0.3,
                            },
                            {
                                "enabled": True,
                                "name": "characters/third.safetensors",
                                "strength_model": 0.9,
                                "strength_clip": 0.7,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            args = self._parse(["--project-config", str(project_config)])

            resolved = resolve_project_config_defaults(args)

            self.assertTrue(resolved["lora_split_enabled"])
            self.assertEqual([1, 2, 3], [lora.index for lora in resolved["loras"]])
            self.assertEqual(
                ["characters/first.safetensors", "characters/second.safetensors", "characters/third.safetensors"],
                [lora.name for lora in resolved["loras"]],
            )
            self.assertEqual([True, False, True], [lora.enabled for lora in resolved["loras"]])

    def test_lora_split_cli_overrides_project_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_config = temp / "config.json"
            project_config.write_text(
                json.dumps(
                    {
                        "project_name": "multi_lora",
                        "input_audio": "input/song.mp3",
                        "lora_split_enabled": True,
                    }
                ),
                encoding="utf-8",
            )

            args = self._parse(["--project-config", str(project_config), "--no-lora-split-enabled"])

            resolved = resolve_project_config_defaults(args)

            self.assertFalse(resolved["lora_split_enabled"])

    def test_lora_1_cli_overrides_first_resolved_lora(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_config = temp / "config.json"
            project_config.write_text(
                json.dumps(
                    {
                        "project_name": "multi_lora",
                        "input_audio": "input/song.mp3",
                        "loras": [
                            {
                                "enabled": True,
                                "name": "characters/config.safetensors",
                                "strength_model": 0.8,
                                "strength_clip": 0.6,
                            },
                            {
                                "enabled": True,
                                "name": "characters/second.safetensors",
                                "strength_model": 0.5,
                                "strength_clip": 0.5,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            args = self._parse([
                "--project-config",
                str(project_config),
                "--lora-1-enabled",
                "--lora-1-name",
                "characters/override.safetensors",
                "--lora-1-strength-model",
                "0.9",
                "--lora-1-strength-clip",
                "0.7",
            ])

            resolved = resolve_project_config_defaults(args)

            self.assertEqual(2, len(resolved["loras"]))
            self.assertEqual("characters/override.safetensors", resolved["loras"][0].name)
            self.assertEqual(0.9, resolved["loras"][0].strength_model)
            self.assertEqual("characters/second.safetensors", resolved["loras"][1].name)

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

    def test_sanitize_file_stem_keeps_safe_project_name_characters(self):
        self.assertEqual("La_Entity_01", sanitize_file_stem(" La Entity 01! ", "fallback"))
        self.assertEqual("fallback", sanitize_file_stem("!!!", "fallback"))

    def test_final_concat_paths_use_sanitized_project_name_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "ltx"

            video_only, final_concat = final_concat_paths(
                output_dir=output_dir,
                project_name="La Entity!",
            )

            self.assertEqual(output_dir / "La_Entity_video_only.mp4", video_only)
            self.assertEqual(output_dir / "La_Entity.mp4", final_concat)


if __name__ == "__main__":
    unittest.main()
