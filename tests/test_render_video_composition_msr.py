import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
from feverslop.errors import FeverSlopValidationError


class RenderVideoCompositionMSRTests(unittest.TestCase):
    def test_standalone_single_prompt_uses_separate_single_workflow_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {
                "video_workflow_limits": [
                    {"workflow": "relay.json", "max_render_duration_seconds": 24},
                    {"workflow": "single.json", "max_render_duration_seconds": 18},
                ],
            }}), encoding="utf-8")
            relay_workflow = temp / "relay.json"
            single_workflow = temp / "single.json"
            relay_workflow.write_text("{}", encoding="utf-8")
            single_workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient"):
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        workflow_path=relay_workflow,
                        single_prompt_workflow_path=single_workflow,
                        output_dir=temp / "out",
                        render_mode="single_prompt",
                        rolling_frame_profile="off",
                    )
                )

            self.assertEqual(18.0, use_case.backend.max_render_duration_seconds)
            self.assertEqual("single.json", use_case.backend.render_budget_workflow_path)

    def test_standalone_render_uses_scene_fps_to_enforce_default_duration_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {
                "default_max_render_duration_seconds": 2,
            }}), encoding="utf-8")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient") as client_type:
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        workflow_path=workflow,
                        output_dir=temp / "out",
                        render_mode="single_prompt",
                        rolling_frame_profile="off",
                    )
                )

            backend = use_case.backend
            scene = {
                "scene": 1, "fps": 24, "frame_count": 50,
                "abs_start_seconds": 0,
                "ltx": {"original_style_i2v_prompt": "prompt"},
            }
            with self.assertRaisesRegex(FeverSlopValidationError, "limited to 49 frames"):
                backend.render_scene_video(
                    scene, "audio.mp3", "start.png", backend._rolling_spec(scene)
                )

            self.assertIsNone(backend.max_render_frames)
            self.assertEqual(2.0, backend.max_render_duration_seconds)
            client_type.return_value.queue_prompt.assert_not_called()

    def test_auto_mode_uses_strictest_workflow_limit_and_reports_limiting_workflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {
                "video_workflow_limits": [
                    {"workflow": "relay.json", "max_render_duration_seconds": 24},
                    {"workflow": "single.json", "max_render_duration_seconds": 18},
                ],
            }}), encoding="utf-8")
            project_config = temp / "config.json"
            project_config.write_text(json.dumps({
                "input_audio": "song.mp3",
                "video": {"fps": 24},
                "scene_generation": {"min_duration": 2, "max_duration": 30},
            }), encoding="utf-8")
            relay_workflow = temp / "relay.json"
            single_workflow = temp / "single.json"
            relay_workflow.write_text("{}", encoding="utf-8")
            single_workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient") as client_type:
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        project_config_path=project_config,
                        workflow_path=relay_workflow,
                        single_prompt_workflow_path=single_workflow,
                        output_dir=temp / "out",
                        render_mode="auto",
                        rolling_frame_profile="original",
                    )
                )

            backend = use_case.backend
            scene = {
                "scene": 1,
                "fps": 24,
                "frame_count": 409,
                "abs_start_seconds": 0,
                "ltx": {
                    "render_mode_hint": "single_prompt",
                    "original_style_i2v_prompt": "prompt",
                },
            }
            with self.assertRaisesRegex(FeverSlopValidationError, r"single\.json is limited to 433 frames"):
                backend.render_scene_video(
                    scene,
                    "audio.mp3",
                    "start.png",
                    backend._rolling_spec(scene),
                )

            self.assertEqual(433, backend.max_render_frames)
            client_type.return_value.queue_prompt.assert_not_called()

    def test_workflow_limit_is_resolved_with_project_fps_and_rolling_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {
                "default_max_render_duration_seconds": 20,
                "video_workflow_limits": [{
                    "workflow": "configured/workflow.json",
                    "max_render_duration_seconds": 12.5,
                }],
            }}), encoding="utf-8")
            project_config = temp / "config.json"
            project_config.write_text(json.dumps({
                "input_audio": "song.mp3",
                "video": {"fps": 25},
                "scene_generation": {"min_duration": 2, "max_duration": 30},
            }), encoding="utf-8")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient"):
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        project_config_path=project_config,
                        workflow_path=workflow,
                        output_dir=temp / "out",
                        video_pipeline="ltx_msr",
                        rolling_frame_profile="original",
                        max_duration=99,
                    )
                )

            self.assertEqual(313, use_case.backend.max_render_frames)
            self.assertEqual(12.5, use_case.backend.max_render_duration_seconds)

    def test_unmatched_workflow_uses_default_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {
                "default_max_render_duration_seconds": 10,
                "video_workflow_limits": [{
                    "workflow": "other.json",
                    "max_render_duration_seconds": 5,
                }],
            }}), encoding="utf-8")
            project_config = temp / "config.json"
            project_config.write_text(json.dumps({
                "input_audio": "song.mp3",
                "video": {"fps": 24},
            }), encoding="utf-8")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient"):
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        project_config_path=project_config,
                        workflow_path=workflow,
                        output_dir=temp / "out",
                        video_pipeline="ltx_msr",
                    )
                )

            self.assertEqual(241, use_case.backend.max_render_frames)
            self.assertEqual(10.0, use_case.backend.max_render_duration_seconds)

    def test_video_pipeline_ltx_msr_builds_msr_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {"base_url": "http://127.0.0.1:8188"}}), encoding="utf-8")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient"):
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        workflow_path=workflow,
                        output_dir=temp / "out",
                        video_pipeline="ltx_msr",
                    )
                )

            self.assertIsInstance(use_case.backend, ComfyUIMSRVideoRenderBackend)

    def test_video_pipeline_ltx_msr_forwards_randomize_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {"base_url": "http://127.0.0.1:8188"}}), encoding="utf-8")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient"):
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        workflow_path=workflow,
                        output_dir=temp / "out",
                        video_pipeline="ltx_msr",
                        randomize_seed=True,
                    )
                )

            self.assertTrue(use_case.backend.randomize_seed)
