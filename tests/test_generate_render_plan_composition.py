import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
from feverslop.composition.generate_render_plan import build_generate_render_plan_execution_request


class GenerateRenderPlanCompositionTests(unittest.TestCase):
    def test_execution_request_ignores_empty_workflows_when_explicit_override_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_config_path = temp / "config.json"
            project_config_path.write_text(
                json.dumps(
                    {
                        "project_name": "demo",
                        "input_audio": "song.wav",
                        "video": {"fps": 24},
                        "scene_generation": {"min_duration": 2.0, "max_duration": 30.0},
                    }
                ),
                encoding="utf-8",
            )
            app_config_path = temp / "app_config.json"
            app_config_path.write_text(
                json.dumps(
                    {
                        "comfyui": {
                            "default_max_render_duration_seconds": 12.0,
                            "video_workflow_limits": [
                                {
                                    "workflow": "optimized.json",
                                    "max_render_duration_seconds": 24.0,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            execution_request = build_generate_render_plan_execution_request(
                GenerateRenderPlanRequest(
                    project_config_path=project_config_path,
                    app_config_path=app_config_path,
                    video_workflow_paths=(Path(""), Path("."), "   ", Path("optimized.json")),
                )
            )

            policy = execution_request.scene_duration_policy
            self.assertEqual(24.0, policy.max_render_duration_seconds)
            self.assertEqual("optimized.json", policy.limiting_workflow)

    def test_execution_request_resolves_workflow_capacity_without_mutating_project_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_config_path = temp / "config.json"
            project_config_path.write_text(
                json.dumps(
                    {
                        "project_name": "demo",
                        "input_audio": "song.wav",
                        "video": {"fps": 24},
                        "scene_generation": {"min_duration": 2.0, "max_duration": 30.0},
                    }
                ),
                encoding="utf-8",
            )
            app_config_path = temp / "app_config.json"
            app_config_path.write_text(
                json.dumps(
                    {"comfyui": {"default_max_render_duration_seconds": 18.0}}
                ),
                encoding="utf-8",
            )

            execution_request = build_generate_render_plan_execution_request(
                GenerateRenderPlanRequest(
                    project_config_path=project_config_path,
                    app_config_path=app_config_path,
                    video_workflow_paths=(Path("workflows/video.json"),),
                    rolling_frame_profile="original",
                )
            )

            policy = execution_request.scene_duration_policy
            self.assertIsNotNone(policy)
            self.assertEqual(2.0, policy.effective_min_seconds)
            self.assertEqual(14.916, policy.effective_max_seconds)
            self.assertEqual("video.json", policy.limiting_workflow)
            self.assertEqual(30.0, execution_request.config.scene_generation.max_duration)
            self.assertEqual(
                30.0,
                json.loads(project_config_path.read_text(encoding="utf-8"))["scene_generation"][
                    "max_duration"
                ],
            )

    def test_scene_generation_seed_minus_one_randomizes_concrete_seed(self):
        from feverslop.composition.generate_render_plan import _build_scene_generator

        generator = _build_scene_generator(
            SimpleNamespace(
                min_duration=2.0,
                max_duration=10.0,
                bias=0.7,
                duration_preset="impact_weighted",
                seed=-1,
            )
        )

        self.assertIsInstance(generator.seed, int)
        self.assertNotEqual(-1, generator.seed)


if __name__ == "__main__":
    unittest.main()
