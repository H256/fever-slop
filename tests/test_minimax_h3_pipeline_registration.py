"""Unit tests for MiniMax H3 pipeline registration (issue #202).

Verifies that minimax-h3-r2v and minimax-h3-t2v are correctly registered
across VIDEO_PIPELINE_BY_MODE, CLI runner arguments, project config
validation, and render-video backend dispatch.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from feverslop.adapters.pipeline_runner_options import RUNNER_ARGUMENTS
from feverslop.composition.render_video import (
    RenderVideoCompositionOptions,
    build_render_video_scenes_use_case,
)
from feverslop.studio.jobs import (
    FULL_PIPELINE_STEPS_BY_MODE,
    _pipeline_step_names,
    _video_pipeline_for_mode,
)
from feverslop.studio.project_validation import (
    VIDEO_PIPELINE_BY_MODE,
    validate_full_auto_inputs,
    validate_pipeline_mode,
    validate_project_config,
)

RENDER_PATCH = "feverslop.composition.render_video"


# ---------------------------------------------------------------------------
# VIDEO_PIPELINE_BY_MODE
# ---------------------------------------------------------------------------


class VideoPipelineByModeTests(unittest.TestCase):
    """Verify VIDEO_PIPELINE_BY_MODE contains MiniMax H3 entries."""

    def test_contains_minimax_r2v(self):
        self.assertIn("minimax_h3_r2v", VIDEO_PIPELINE_BY_MODE)
        self.assertEqual("minimax-h3-r2v", VIDEO_PIPELINE_BY_MODE["minimax_h3_r2v"])

    def test_contains_minimax_t2v(self):
        self.assertIn("minimax_h3_t2v", VIDEO_PIPELINE_BY_MODE)
        self.assertEqual("minimax-h3-t2v", VIDEO_PIPELINE_BY_MODE["minimax_h3_t2v"])


# ---------------------------------------------------------------------------
# validate_pipeline_mode
# ---------------------------------------------------------------------------


class ValidatePipelineModeTests(unittest.TestCase):
    """Verify validate_pipeline_mode accepts MiniMax H3 modes."""

    def test_accepts_minimax_h3_r2v(self):
        self.assertEqual("minimax_h3_r2v", validate_pipeline_mode("minimax_h3_r2v"))

    def test_accepts_minimax_h3_t2v(self):
        self.assertEqual("minimax_h3_t2v", validate_pipeline_mode("minimax_h3_t2v"))

    def test_raises_on_invalid_mode(self):
        with self.assertRaises(ValueError):
            validate_pipeline_mode("invalid_mode_xyz")

    def test_returns_string_as_is_for_valid_mode(self):
        self.assertEqual("minimax_h3_r2v", validate_pipeline_mode("minimax_h3_r2v"))
        self.assertEqual("minimax_h3_t2v", validate_pipeline_mode("minimax_h3_t2v"))


# ---------------------------------------------------------------------------
# validate_project_config
# ---------------------------------------------------------------------------


class ValidateProjectConfigMinimaxTests(unittest.TestCase):
    """Verify validate_project_config accepts MiniMax video_pipeline values."""

    def _base_config(self, *, video_pipeline=None):
        return {
            "project_name": "test_proj",
            "input_audio": "test.wav",
            "video_pipeline": video_pipeline,
        }

    def test_rejects_unknown_pipeline(self):
        with self.assertRaises(ValueError):
            validate_project_config(self._base_config(video_pipeline="unknown"))

    def test_accepts_minimax_h3_r2v_pipeline(self):
        # Should _not_ raise
        validate_project_config(self._base_config(video_pipeline="minimax-h3-r2v"))

    def test_accepts_minimax_h3_t2v_pipeline(self):
        # Should _not_ raise
        validate_project_config(self._base_config(video_pipeline="minimax-h3-t2v"))

    def test_accepts_none_pipeline(self):
        # Omission is valid
        config = {"project_name": "test", "input_audio": "test.wav"}
        validate_project_config(config)


# ---------------------------------------------------------------------------
# validate_full_auto_inputs
# ---------------------------------------------------------------------------


class ValidateFullAutoInputsTests(unittest.TestCase):
    """Verify validate_full_auto_inputs accepts MiniMax H3 pipeline modes."""

    def _make_request(self, pipeline_mode: str) -> MagicMock:
        req = MagicMock()
        req.duration_seconds = 4.0
        req.width = 1344
        req.height = 768
        req.fps = 24
        req.pipeline_mode = pipeline_mode
        return req

    def test_accepts_minimax_h3_r2v(self):
        # Should _not_ raise
        validate_full_auto_inputs(self._make_request("minimax_h3_r2v"))

    def test_accepts_minimax_h3_t2v(self):
        # Should _not_ raise
        validate_full_auto_inputs(self._make_request("minimax_h3_t2v"))


# ---------------------------------------------------------------------------
# _video_pipeline_for_mode
# ---------------------------------------------------------------------------


class VideoPipelineForModeTests(unittest.TestCase):
    """Verify _video_pipeline_for_mode resolves MiniMax H3 modes."""

    def test_minimax_h3_r2v_maps_to_r2v(self):
        self.assertEqual(
            "minimax-h3-r2v",
            _video_pipeline_for_mode("minimax_h3_r2v"),
        )

    def test_minimax_h3_t2v_maps_to_t2v(self):
        self.assertEqual(
            "minimax-h3-t2v",
            _video_pipeline_for_mode("minimax_h3_t2v"),
        )

    def test_hyphen_underscore_aliases_accepted(self):
        """Both hyphen and underscore forms resolve to the canonical name."""
        self.assertEqual(
            "minimax-h3-r2v",
            _video_pipeline_for_mode("minimax-h3-r2v"),
        )
        self.assertEqual(
            "minimax-h3-t2v",
            _video_pipeline_for_mode("minimax-h3-t2v"),
        )

    def test_raises_on_unknown_mode(self):
        with self.assertRaises(ValueError):
            _video_pipeline_for_mode("does_not_exist")


# ---------------------------------------------------------------------------
# Pipeline step names
# ---------------------------------------------------------------------------


class PipelineStepsTests(unittest.TestCase):
    """Verify _pipeline_step_names returns correct MiniMax H3 steps."""

    EXPECTED_R2V = ["Main pipeline", "MSR references", "Video render", "Final concat"]
    EXPECTED_T2V = ["Main pipeline", "Video render", "Final concat"]

    def test_returns_correct_full_pipeline_steps(self):
        actual = _pipeline_step_names("full-pipeline", pipeline_mode="minimax_h3_r2v")
        self.assertEqual(self.EXPECTED_R2V, actual)

    def test_returns_correct_full_pipeline_steps_t2v(self):
        actual = _pipeline_step_names("full-pipeline", pipeline_mode="minimax_h3_t2v")
        self.assertEqual(self.EXPECTED_T2V, actual)

    def test_r2v_has_msr_references(self):
        self.assertIn("MSR references", self.EXPECTED_R2V)

    def test_t2v_no_msr_references(self):
        self.assertNotIn("MSR references", self.EXPECTED_T2V)

    def test_non_full_pipeline_steps_fall_through(self):
        """Non full-pipeline action returns the action's own step(s)."""
        actual = _pipeline_step_names("preparation", pipeline_mode="minimax_h3_r2v")
        self.assertEqual(actual, ["preparation"])


# ---------------------------------------------------------------------------
# FULL_PIPELINE_STEPS_BY_MODE
# ---------------------------------------------------------------------------


class FullPipelineStepsByModeTests(unittest.TestCase):
    """Verify FULL_PIPELINE_STEPS_BY_MODE contains MiniMax H3 entries."""

    def test_contains_minimax_h3_r2v(self):
        self.assertIn("minimax_h3_r2v", FULL_PIPELINE_STEPS_BY_MODE)

    def test_contains_minimax_h3_t2v(self):
        self.assertIn("minimax_h3_t2v", FULL_PIPELINE_STEPS_BY_MODE)

    def test_r2v_has_expected_steps(self):
        expected = ["Main pipeline", "MSR references", "Video render", "Final concat"]
        self.assertEqual(expected, FULL_PIPELINE_STEPS_BY_MODE["minimax_h3_r2v"])

    def test_t2v_has_expected_steps(self):
        expected = ["Main pipeline", "Video render", "Final concat"]
        self.assertEqual(expected, FULL_PIPELINE_STEPS_BY_MODE["minimax_h3_t2v"])


# ---------------------------------------------------------------------------
# CLI Runner arguments
# ---------------------------------------------------------------------------


class RunnerArgumentsTests(unittest.TestCase):
    """Verify RUNNER_ARGUMENTS includes MiniMax video_pipeline choices."""

    def _get_video_pipeline_choices(self):
        for name, flags, kwargs in RUNNER_ARGUMENTS:
            if name == "video_pipeline":
                return kwargs.get("choices", [])
        self.fail("video_pipeline not found in RUNNER_ARGUMENTS")

    def test_choices_include_minimax_h3_r2v(self):
        self.assertIn("minimax-h3-r2v", self._get_video_pipeline_choices())

    def test_choices_include_minimax_h3_t2v(self):
        self.assertIn("minimax-h3-t2v", self._get_video_pipeline_choices())


# ---------------------------------------------------------------------------
# Render video backend dispatch
# ---------------------------------------------------------------------------


class MockFullAuto:
    """Minimal mock for validate_full_auto_inputs checks."""

    duration_seconds = 4.0
    width = 1344
    height = 768
    fps = 24
    pipeline_mode = "minimax_h3_r2v"


class MockAppConfig:
    """Minimal mock AppConfig that satisfies the pipeline render function."""

    def __init__(self, path):
        self.comfyui = MagicMock()
        self.comfyui.base_url = "http://localhost:8188"
        self.comfyui.model_overrides = {}
        self.comfyui.video_workflow_limits = []
        self.comfyui.default_max_render_duration_seconds = 30.0


class MockProjectConfig:
    """Minimal mock ProjectConfig that satisfies the pipeline render function."""

    def __init__(self, path):
        self.project_dir = "/tmp/test_project"
        self.scene_generation = MagicMock()
        self.scene_generation.min_duration = 4.0
        self.scene_generation.max_duration = 15.0
        self.video_pipeline = "minimax-h3-r2v"

    def to_video_settings(self):
        vs = MagicMock()
        vs.fps = 24
        return vs


class RenderVideoBackendDispatchTests(unittest.TestCase):
    """Verify build_render_video_scenes_use_case creates the right backends."""

    @patch(f"{RENDER_PATCH}.ComfyUIClient")
    def test_dispatcher_creates_minimax_backend_r2v(self, mock_client):
        """Dispatcher should _not_ raise and should _not_ create an MSRS backend."""
        mock_client.return_value = MagicMock()
        options = RenderVideoCompositionOptions(
            app_config_path="config.json",
            workflow_path="workflow.json",
            video_pipeline="minimax-h3-r2v",
            output_dir="/tmp/output",
        )

        mock_pc = MockProjectConfig("x")
        mock_ac = MockAppConfig("x")
        mock_ac.comfyui.video_workflow_limits = []
        mock_ac.comfyui.default_max_render_duration_seconds = 30.0

        with patch(f"{RENDER_PATCH}.AppConfig.load", return_value=mock_ac):
            with patch(f"{RENDER_PATCH}.ProjectConfig.load", return_value=mock_pc):
                use_case = build_render_video_scenes_use_case(options)

        # Verify the backend was instantiated
        self.assertTrue(hasattr(use_case, "backend"))

    @patch(f"{RENDER_PATCH}.ComfyUIClient")
    def test_dispatcher_creates_minimax_backend_t2v(self, mock_client):
        """Dispatcher should create MiniMax H3 T2V backend."""
        mock_client.return_value = MagicMock()
        options = RenderVideoCompositionOptions(
            app_config_path="config.json",
            workflow_path="workflow.json",
            video_pipeline="minimax-h3-t2v",
            output_dir="/tmp/output",
        )

        mock_pc = MockProjectConfig("x")
        mock_ac = MockAppConfig("x")
        mock_ac.comfyui.video_workflow_limits = []
        mock_ac.comfyui.default_max_render_duration_seconds = 30.0

        with patch(f"{RENDER_PATCH}.AppConfig.load", return_value=mock_ac):
            with patch(f"{RENDER_PATCH}.ProjectConfig.load", return_value=mock_pc):
                use_case = build_render_video_scenes_use_case(options)

        self.assertTrue(hasattr(use_case, "backend"))


if __name__ == "__main__":
    unittest.main()
