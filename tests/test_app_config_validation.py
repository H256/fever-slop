import tempfile
import unittest
from pathlib import Path


class AppConfigValidationTests(unittest.TestCase):
    def test_rejects_nonboolean_dspy_cache(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {"dspy_cache": "true"}}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "llm.dspy_cache must be a boolean"):
                AppConfig.load(config_path)

    def test_rejects_nonfinite_default_max_render_duration(self):
        from feverslop.config.app_config import AppConfig

        for value in ("NaN", "Infinity"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "app_config.json"
                config_path.write_text(
                    f'{{"comfyui": {{"default_max_render_duration_seconds": {value}}}}}',
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "default_max_render_duration_seconds"):
                    AppConfig.load(config_path)

    def test_rejects_nonfinite_video_workflow_limit_duration(self):
        from feverslop.config.app_config import AppConfig

        for value in ("NaN", "Infinity"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "app_config.json"
                config_path.write_text(
                    f"""
                    {{
                      "comfyui": {{
                        "video_workflow_limits": [
                          {{"workflow": "video.json", "max_render_duration_seconds": {value}}}
                        ]
                      }}
                    }}
                    """,
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "max_render_duration_seconds"):
                    AppConfig.load(config_path)

    def test_rejects_nonstring_video_workflow_limit_name(self):
        from feverslop.config.app_config import AppConfig

        for value in ("null", "123", "{}"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "app_config.json"
                config_path.write_text(
                    f"""
                    {{
                      "comfyui": {{
                        "video_workflow_limits": [
                          {{"workflow": {value}, "max_render_duration_seconds": 18}}
                        ]
                      }}
                    }}
                    """,
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "workflow must be a string"):
                    AppConfig.load(config_path)

    def test_rejects_nonpositive_default_max_render_duration(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"comfyui": {"default_max_render_duration_seconds": 0}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "default_max_render_duration_seconds"):
                AppConfig.load(config_path)

    def test_rejects_blank_video_workflow_limit_name(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "video_workflow_limits": [
                      {"workflow": "   ", "max_render_duration_seconds": 18}
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "workflow"):
                AppConfig.load(config_path)

    def test_rejects_nonpositive_video_workflow_limit_duration(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "video_workflow_limits": [
                      {"workflow": "workflows/video.json", "max_render_duration_seconds": -1}
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "max_render_duration_seconds"):
                AppConfig.load(config_path)

    def test_rejects_duplicate_video_workflow_basenames_case_insensitively(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "video_workflow_limits": [
                      {"workflow": "workflows/Ingredients_v4.json", "max_render_duration_seconds": 18},
                      {"workflow": "alternate/ingredients_V4.JSON", "max_render_duration_seconds": 12}
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate video workflow limit"):
                AppConfig.load(config_path)

    def test_all_required_keys_present_loads_without_error(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"base_url": "http://example.com"}, "comfyui": {"base_url": "http://localhost:8188"}}',
                encoding="utf-8",
            )

            config = AppConfig.load(config_path, required_keys=["llm", "comfyui"])

        self.assertEqual("http://example.com", config.llm.base_url)

    def test_missing_config_file_raises_with_required_keys(self):
        from feverslop.config.app_config import AppConfig

        config_path = Path("does-not-exist.json")

        with self.assertRaises(ValueError) as ctx:
            AppConfig.load(config_path, required_keys=["llm", "comfyui"])

        self.assertIn("llm", str(ctx.exception))
        self.assertIn("comfyui", str(ctx.exception))

    def test_empty_config_raises_for_required_keys(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{}', encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                AppConfig.load(config_path, required_keys=["llm", "comfyui"])

            self.assertIn("llm", str(ctx.exception))
            self.assertIn("comfyui", str(ctx.exception))


    def test_storyboard_prompt_transform_rejects_missing_workflow(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "storyboard_prompt_transforms": [{}]
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "workflow"):
                AppConfig.load(config_path)

    def test_storyboard_prompt_transform_requires_workflow_as_string(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "storyboard_prompt_transforms": [
                    {"workflow": null}
                  ]
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "workflow"):
                AppConfig.load(config_path)

    def test_comfyui_model_override_rejects_missing_fields(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "model_overrides": [{}]
                  }
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "workflow"):
                AppConfig.load(config_path)

    def test_comfyui_model_override_requires_all_six_fields(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "model_overrides": [
                      {
                        "workflow": "test.json"
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "node_id"):
                AppConfig.load(config_path)

    def test_comfyui_model_override_requires_string_fields(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                """
                {
                  "comfyui": {
                    "model_overrides": [
                      {
                        "workflow": null
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "workflow"):
                AppConfig.load(config_path)


class AppConfigBackwardCompatTests(unittest.TestCase):
    def test_no_required_keys_allows_missing_file(self):
        from feverslop.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertEqual("http://localhost:8080/v1", config.llm.base_url)


class LLMConfigValidationTests(unittest.TestCase):
    """Test LLMConfig temperature and max_tokens validation."""

    def test_rejects_negative_temperature(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"temperature": -0.5}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "llm.temperature must be >= 0"):
                AppConfig.load(config_path)

    def test_rejects_zero_max_tokens(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"max_tokens": 0}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "llm.max_tokens must be > 0"):
                AppConfig.load(config_path)

    def test_rejects_negative_max_tokens(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"max_tokens": -1}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "llm.max_tokens must be > 0"):
                AppConfig.load(config_path)

    def test_rejects_zero_llm_max_concurrent_requests(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"max_concurrent_requests": 0}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "llm.max_concurrent_requests must be > 0"):
                AppConfig.load(config_path)


class RecursiveRequiredKeysTests(unittest.TestCase):
    """Test recursive required_keys validation with dot-notation paths."""

    def test_required_keys_catches_nested_null(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"base_url": null}, "comfyui": {}}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                AppConfig.load(config_path, required_keys=["llm.base_url"])
            self.assertIn("base_url", str(ctx.exception))

    def test_required_keys_accepts_nested_present(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"base_url": "http://example.com"}, "comfyui": {}}',
                encoding="utf-8",
            )
            config = AppConfig.load(config_path, required_keys=["llm.base_url"])
            self.assertEqual("http://example.com", config.llm.base_url)

    def test_required_keys_toplevel_still_works(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"base_url": "http://example.com"}}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                AppConfig.load(config_path, required_keys=["llm", "comfyui"])
            self.assertIn("comfyui", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
