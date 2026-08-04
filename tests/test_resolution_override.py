"""Tests for --resolution CLI override (Issue #211)."""

import json
import tempfile
import unittest
from pathlib import Path


class TestResolutionTuple(unittest.TestCase):
    """Test ResolutionTuple parsing."""

    def test_parse_valid_wxh(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        res = ResolutionTuple.parse("1280x720")
        self.assertEqual(res.width, 1280)
        self.assertEqual(res.height, 720)

    def test_parse_valid_common(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        res = ResolutionTuple.parse("1920x1080")
        self.assertEqual(res.width, 1920)
        self.assertEqual(res.height, 1080)

    def test_parse_fails_no_separator(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        with self.assertRaises(ValueError):
            ResolutionTuple.parse("1280")

    def test_parse_fails_empty(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        with self.assertRaises(ValueError):
            ResolutionTuple.parse("")

    def test_parse_fails_non_integer(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        with self.assertRaises(ValueError):
            ResolutionTuple.parse("abcxdef")

    def test_parse_fails_malformed(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        with self.assertRaises(ValueError):
            ResolutionTuple.parse("x720")
        with self.assertRaises(ValueError):
            ResolutionTuple.parse("1280x")


class TestProjectConfigResolutionOverride(unittest.TestCase):
    """Test ProjectConfig.apply_resolution_override()."""

    def _make_config(self, **video_overrides):
        """Create a minimal config.json in tempdir and load it."""
        config_data = {
            "input_audio": "test_audio.mp3",
            "project_name": "test_project",
            "video": {
                "fps": 24,
                "width": 1280,
                "height": 704,
                **video_overrides,
            },
        }
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        # Also create dummy audio file
        audio_path = tmpdir / "test_audio.mp3"
        audio_path.touch()
        from feverslop.config.project_config import ProjectConfig
        return ProjectConfig.load(str(config_path)), tmpdir

    def test_apply_resolution_override_changes_video(self):
        config, _ = self._make_config()
        patched = config.apply_resolution_override(width=1920, height=1080)
        vs = patched.to_video_settings()
        self.assertEqual(vs.width, 1920)
        self.assertEqual(vs.height, 1080)

    def test_apply_resolution_override_preserves_fps(self):
        config, _ = self._make_config(fps=30)
        patched = config.apply_resolution_override(width=1920, height=1080)
        vs = patched.to_video_settings()
        self.assertEqual(vs.fps, 30)
        self.assertEqual(vs.width, 1920)
        self.assertEqual(vs.height, 1080)

    def test_apply_resolution_override_returns_same_when_no_override(self):
        config, _ = self._make_config()
        same = config.apply_resolution_override()
        self.assertIs(same, config)

    def test_apply_resolution_override_only_width(self):
        config, _ = self._make_config()
        patched = config.apply_resolution_override(width=1920)
        vs = patched.to_video_settings()
        self.assertEqual(vs.width, 1920)
        self.assertEqual(vs.height, 704)

    def test_apply_resolution_override_only_height(self):
        config, _ = self._make_config()
        patched = config.apply_resolution_override(height=1080)
        vs = patched.to_video_settings()
        self.assertEqual(vs.width, 1280)
        self.assertEqual(vs.height, 1080)

    def test_apply_resolution_override_is_immutable(self):
        config, _ = self._make_config()
        patched = config.apply_resolution_override(width=1920, height=1080)
        # Original config unchanged
        self.assertEqual(config.to_video_settings().width, 1280)
        self.assertEqual(config.to_video_settings().height, 704)
        # Patched has new values
        self.assertEqual(patched.to_video_settings().width, 1920)
        self.assertEqual(patched.to_video_settings().height, 1080)


class TestResolutionCliParsing(unittest.TestCase):
    """Test --resolution CLI flag parsing."""

    def test_resolution_flag_parsed_into_args(self):
        from feverslop.composition.arg_parser import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["./projects/test", "--resolution", "1920x540"])
        self.assertEqual(args.resolution.width, 1920)
        self.assertEqual(args.resolution.height, 540)

    def test_resolution_flag_missing_is_none(self):
        from feverslop.composition.arg_parser import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["./projects/test"])
        self.assertIsNone(args.resolution)

    def test_resolution_flag_invalid_rejected(self):
        from feverslop.composition.arg_parser import build_arg_parser
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["./projects/test", "--resolution", "bad"])


if __name__ == "__main__":
    unittest.main()
