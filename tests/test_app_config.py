import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
import os


class AppConfigTests(unittest.TestCase):
    def test_loads_video_workflow_duration_limits(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "default_max_render_duration_seconds": 18,
                    "video_workflow_limits": [
                      {
                        "workflow": "workflows/video_Ingredients_v4.json",
                        "max_render_duration_seconds": 12.5
                      },
                      {
                        "workflow": "workflows/video_other.json",
                        "max_render_duration_seconds": 24
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            config = AppConfig.load(config_path)

        self.assertEqual(18.0, config.comfyui.default_max_render_duration_seconds)
        self.assertIsInstance(config.comfyui.video_workflow_limits, tuple)
        self.assertEqual(
            [
                ("workflows/video_Ingredients_v4.json", 12.5),
                ("workflows/video_other.json", 24.0),
            ],
            [
                (limit.workflow, limit.max_render_duration_seconds)
                for limit in config.comfyui.video_workflow_limits
            ],
        )
        limit = config.comfyui.video_workflow_limits[0]
        with self.assertRaises(FrozenInstanceError):
            limit.max_render_duration_seconds = 20

    def test_missing_video_workflow_duration_limits_preserves_existing_defaults(self):
        from feverslop.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertIsNone(config.comfyui.default_max_render_duration_seconds)
        self.assertEqual((), config.comfyui.video_workflow_limits)

    def test_loads_comfyui_prompt_timeout_seconds(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "prompt_timeout_seconds": 7200
                  }
                }
                """,
                encoding="utf-8",
            )

            config = AppConfig.load(config_path)

        self.assertEqual(7200, config.comfyui.prompt_timeout_seconds)

    def test_missing_comfyui_prompt_timeout_defaults_to_30_minutes(self):
        from feverslop.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertEqual(1800, config.comfyui.prompt_timeout_seconds)

    def test_load_accepts_windows_relative_app_config_path(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "llm": {
                    "model": "configured-model"
                  }
                }
                """,
                encoding="utf-8",
            )
            with working_directory(config_path.parent):
                config = AppConfig.load(".\\app_config.json")

        self.assertEqual("configured-model", config.llm.model)

    def test_storyboard_prompt_transforms_default_to_empty_list(self):
        from feverslop.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertEqual([], config.storyboard_prompt_transforms)

    def test_loads_storyboard_prompt_transform_config(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "storyboard_prompt_transforms": [
                    {
                      "workflow": "workflows/image_t2i_startframe_ideogram_v1.json",
                      "kind": "template",
                      "template": "docs/ideogram4_prompt_template.md",
                      "positive_prompt_input": "text",
                      "debug_dir": "ideogram4_prompt_debug"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            config = AppConfig.load(config_path)

        [transform] = config.storyboard_prompt_transforms
        self.assertEqual("workflows/image_t2i_startframe_ideogram_v1.json", transform.workflow)
        self.assertEqual("template", transform.kind)
        self.assertEqual("docs/ideogram4_prompt_template.md", transform.template)
        self.assertEqual("text", transform.positive_prompt_input)
        self.assertEqual("ideogram4_prompt_debug", transform.debug_dir)


@contextmanager
def working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


if __name__ == "__main__":
    unittest.main()
