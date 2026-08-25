"""Tests for --resolution CLI override (Issue #211) and --set-resolution (Issue #220)."""

import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

    def test_parse_megapixels(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        res = ResolutionTuple.parse("0.98")
        self.assertIsNone(res.width)
        self.assertIsNone(res.height)
        self.assertEqual(res.megapixels, 0.98)

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
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
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


class TestSetResolutionOnDisk(unittest.TestCase):
    """Test ProjectConfig.set_resolution_on_disk() for Issue #220."""

    def test_persists_new_resolution_without_disturbing_other_fields(self):
        from feverslop.config.project_config import ProjectConfig

        config_data = {
            "input_audio": "test_audio.mp3",
            "project_name": "test_project",
            "video": {
                "fps": 24,
                "width": 1280,
                "height": 704,
            },
            "custom_field": "preserve_me",
        }
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        (tmpdir / "test_audio.mp3").touch()

        ProjectConfig.set_resolution_on_disk(config_path, width=1920, height=1080)

        raw = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["video"]["width"], 1920)
        self.assertEqual(raw["video"]["height"], 1080)
        self.assertEqual(raw["video"]["fps"], 24)
        self.assertEqual(raw["custom_field"], "preserve_me")

    def test_creates_video_section_if_missing(self):
        from feverslop.config.project_config import ProjectConfig

        config_data = {
            "input_audio": "test_audio.mp3",
            "project_name": "test_project",
        }
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        (tmpdir / "test_audio.mp3").touch()

        ProjectConfig.set_resolution_on_disk(config_path, width=1920, height=1080)

        raw = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["video"]["width"], 1920)
        self.assertEqual(raw["video"]["height"], 1080)

    def test_reload_config_has_new_resolution(self):
        from feverslop.config.project_config import ProjectConfig

        config_data = {
            "input_audio": "test_audio.mp3",
            "project_name": "test_project",
            "video": {
                "fps": 30,
                "width": 1280,
                "height": 704,
            },
        }
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        (tmpdir / "test_audio.mp3").touch()

        ProjectConfig.set_resolution_on_disk(config_path, width=3840, height=2160)

        reloaded = ProjectConfig.load(config_path)
        self.assertEqual(reloaded.to_video_settings().width, 3840)
        self.assertEqual(reloaded.to_video_settings().height, 2160)
        self.assertEqual(reloaded.to_video_settings().fps, 30)


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


class TestSetResolutionCliParsing(unittest.TestCase):
    """Test --set-resolution CLI flag parsing for Issue #220."""

    def test_set_resolution_flag_parsed(self):
        from feverslop.composition.arg_parser import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["./projects/test", "--set-resolution", "1920x1080"])
        self.assertEqual(args.set_resolution.width, 1920)
        self.assertEqual(args.set_resolution.height, 1080)
        # --resolution should still be None
        self.assertIsNone(args.resolution)

    def test_set_resolution_megapixels_flag_parsed(self):
        from feverslop.composition.arg_parser import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["./projects/test", "--set-resolution", "0.98"])
        self.assertEqual(args.set_resolution.megapixels, 0.98)

    def test_set_resolution_flag_missing_is_none(self):
        from feverslop.composition.arg_parser import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["./projects/test"])
        self.assertIsNone(args.set_resolution)

    def test_set_resolution_invalid_rejected(self):
        from feverslop.composition.arg_parser import build_arg_parser
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["./projects/test", "--set-resolution", "bad"])


class TestSetResolutionPipelineStages(unittest.TestCase):
    """Test that --set-resolution resolves to SET_RESOLUTION stage for Issue #220."""

    def test_set_resolution_returns_set_resolution_stage(self):
        from feverslop.composition.arg_parser import PipelineStage, build_arg_parser
        from feverslop.composition.stage_runners import resolve_pipeline_stages

        parser = build_arg_parser()
        args = parser.parse_args(["./projects/test", "--set-resolution", "1920x1080"])
        stages = resolve_pipeline_stages(args)
        self.assertEqual(stages, [PipelineStage.SET_RESOLUTION])

    def test_set_resolution_ignores_skip_flags(self):
        """--set-resolution should ignore all skip flags."""
        from feverslop.composition.arg_parser import PipelineStage, build_arg_parser
        from feverslop.composition.stage_runners import resolve_pipeline_stages

        parser = build_arg_parser()
        args = parser.parse_args([
            "./projects/test",
            "--set-resolution", "1920x1080",
            "--skip-tests",
            "--skip-main-pipeline",
            "--skip-ltx",
        ])
        stages = resolve_pipeline_stages(args)
        self.assertEqual(stages, [PipelineStage.SET_RESOLUTION])

    def test_set_resolution_stage_is_in_choices(self):
        from feverslop.composition.arg_parser import PipelineStage
        self.assertTrue(hasattr(PipelineStage, "SET_RESOLUTION"))


class TestSetResolutionStageLabel(unittest.TestCase):
    """Test that SET_RESOLUTION has a readable label."""

    def test_stage_label_exists(self):
        from feverslop.composition.arg_parser import PipelineStage
        from feverslop.composition.stage_runners import STAGE_LABELS, STAGE_RUNNERS
        self.assertIn(PipelineStage.SET_RESOLUTION, STAGE_LABELS)
        self.assertIn(PipelineStage.SET_RESOLUTION, STAGE_RUNNERS)


class TestSetResolutionSafeInvalidation(unittest.TestCase):
    def test_set_resolution_megapixels_updates_config_and_base(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        from feverslop.composition.stage_runners import _run_set_resolution_stage
        from feverslop.scene_artifacts import SceneArtifactLayout

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps({"input_audio": "song.mp3", "video": {"width": 1280, "height": 704}}), encoding="utf-8")
        (tmpdir / "song.mp3").write_bytes(b"audio")
        layout = SceneArtifactLayout(tmpdir)
        layout.plans_dir.mkdir(parents=True)
        layout.base_plan.write_text(json.dumps([{"scene": 1, "width": 1280, "height": 704}]), encoding="utf-8")
        state = SimpleNamespace(
            args=Namespace(set_resolution=ResolutionTuple.parse("0.98")),
            context=SimpleNamespace(project_config_path=config_path, project_config_dir=tmpdir, artifact_layout=layout),
            plan_for_next_step=layout.base_plan,
        )

        _run_set_resolution_stage(state)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = json.loads(layout.base_plan.read_text(encoding="utf-8"))
        self.assertEqual(0.98, config["video"]["megapixels"])
        self.assertEqual(0.98, base[0]["megapixels"])

    def test_set_resolution_updates_config_and_base_but_leaves_derived_artifacts_stale(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        from feverslop.composition.stage_runners import _run_set_resolution_stage
        from feverslop.scene_artifacts import SceneArtifactLayout

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps({
            "input_audio": "song.mp3",
            "video_pipeline": "ltx_msr",
            "video": {"width": 1280, "height": 704},
        }), encoding="utf-8")
        (tmpdir / "song.mp3").write_bytes(b"audio")
        layout = SceneArtifactLayout(tmpdir)
        layout.plans_dir.mkdir(parents=True)
        scene = {"scene": 1, "width": 1280, "height": 704}
        layout.base_plan.write_text(json.dumps([scene]), encoding="utf-8")
        layout.references_plan.write_text(json.dumps([scene]), encoding="utf-8")
        derived_before = layout.references_plan.read_bytes()
        state = SimpleNamespace(
            args=Namespace(
                set_resolution=ResolutionTuple(1024, 576),
                video_pipeline="ltx_msr",
            ),
            context=SimpleNamespace(
                project_config_path=config_path,
                project_config_dir=tmpdir,
                artifact_layout=layout,
            ),
            plan_for_next_step=layout.base_plan,
        )

        with patch("feverslop.composition.stage_runners._run_ltx_prepare_workflows_stage") as prepare:
            _run_set_resolution_stage(state)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = json.loads(layout.base_plan.read_text(encoding="utf-8"))
        self.assertEqual((1024, 576), (config["video"]["width"], config["video"]["height"]))
        self.assertEqual((1024, 576), (base[0]["width"], base[0]["height"]))
        self.assertEqual(derived_before, layout.references_plan.read_bytes())
        prepare.assert_not_called()

    def test_set_resolution_still_works_before_canonical_plan_exists(self):
        from feverslop.adapters.pipeline_runner_options import ResolutionTuple
        from feverslop.composition.stage_runners import _run_set_resolution_stage
        from feverslop.scene_artifacts import SceneArtifactLayout

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps({
            "input_audio": "song.mp3",
            "video": {"width": 1280, "height": 704},
        }), encoding="utf-8")
        layout = SceneArtifactLayout(tmpdir)
        state = SimpleNamespace(
            args=Namespace(set_resolution=ResolutionTuple(1024, 576)),
            context=SimpleNamespace(
                project_config_path=config_path,
                project_config_dir=tmpdir,
                artifact_layout=layout,
            ),
            plan_for_next_step=layout.base_plan,
        )

        _run_set_resolution_stage(state)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual((1024, 576), (config["video"]["width"], config["video"]["height"]))
        self.assertFalse(layout.base_plan.exists())


if __name__ == "__main__":
    unittest.main()
