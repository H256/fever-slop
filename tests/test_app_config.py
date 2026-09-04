import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch


class AppConfigTests(unittest.TestCase):
    def test_h3_judge_can_be_configured_as_advisory(self):
        from feverslop.config.app_config import AppConfig

        self.assertFalse(AppConfig.load(Path("does-not-exist.json")).llm.prompt_judge_blocking)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"prompt_judge_blocking": false}}',
                encoding="utf-8",
            )
            configured = AppConfig.load(config_path)

        self.assertFalse(configured.llm.prompt_judge_blocking)

    def test_h3_judge_output_budget_is_configurable_with_safe_default(self):
        from feverslop.config.app_config import AppConfig

        default = AppConfig.load(Path("does-not-exist.json"))
        self.assertEqual(8192, default.llm.prompt_judge_max_tokens)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"prompt_judge_max_tokens": 12288}}',
                encoding="utf-8",
            )
            configured = AppConfig.load(config_path)

        self.assertEqual(12288, configured.llm.prompt_judge_max_tokens)

    def test_execution_config_does_not_change_existing_positional_arguments(self):
        from feverslop.config.app_config import AppConfig, ComfyUIConfig, LLMConfig

        library = Path("legacy-library")
        config = AppConfig(LLMConfig(), ComfyUIConfig(), library)

        self.assertEqual(library, config.global_library_path)

    def test_vram_handoff_defaults_to_continuous(self):
        from feverslop.config.app_config import AppConfig, VramHandoffMode

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text("{}", encoding="utf-8")

            config = AppConfig.load(config_path)

        self.assertIs(VramHandoffMode.CONTINUOUS, config.execution.vram_handoff)

    def test_loads_manual_vram_handoff(self):
        from feverslop.config.app_config import AppConfig, VramHandoffMode

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"execution": {"vram_handoff": "manual"}}',
                encoding="utf-8",
            )

            config = AppConfig.load(config_path)

        self.assertIs(VramHandoffMode.MANUAL, config.execution.vram_handoff)

    def test_rejects_invalid_vram_handoff(self):
        from feverslop.config.app_config import AppConfig

        for value in ("automatic", 1, None):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "app_config.json"
                config_path.write_text(
                    json.dumps({"execution": {"vram_handoff": value}}),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "execution.vram_handoff.*continuous.*manual",
                ):
                    AppConfig.load(config_path)

    def test_global_model_remains_fallback_for_optional_task_profiles(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"model": "general", "models": {"creative": "story-model"}}}',
                encoding="utf-8",
            )
            config = AppConfig.load(config_path)

        self.assertEqual("story-model", config.llm.model_for("creative"))
        self.assertEqual("general", config.llm.model_for("structured"))

    def test_rejects_duplicate_normalized_llm_model_profiles(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                json.dumps({
                    "llm": {
                        "models": {
                            "Creative": "first-model",
                            " creative ": "second-model",
                        },
                    },
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"Duplicate llm\.models profile: creative"):
                AppConfig.load(config_path)

    def test_existing_single_model_configuration_has_no_profile_requirements(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {"model": "legacy-model"}}', encoding="utf-8")
            config = AppConfig.load(config_path)

        self.assertEqual({}, config.llm.models)
        self.assertEqual("legacy-model", config.llm.model_for("creative"))

    def test_dspy_temperature_defaults_to_conservative_value(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                json.dumps({"llm": {}, "global_library_path": str(Path(temp_dir) / "library")}),
                encoding="utf-8",
            )

            config = AppConfig.load(config_path)

        self.assertEqual(0.4, config.llm.dspy_temperature)

    def test_loads_dspy_temperature_setting(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {"dspy_temperature": 0.25}}', encoding="utf-8")

            config = AppConfig.load(config_path)

        self.assertEqual(0.25, config.llm.dspy_temperature)

    def test_dspy_cache_defaults_to_false(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                json.dumps({"llm": {}, "global_library_path": str(Path(temp_dir) / "library")}),
                encoding="utf-8",
            )

            config = AppConfig.load(config_path)

        self.assertFalse(config.llm.dspy_cache)

    def test_loads_dspy_cache_setting(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {"dspy_cache": true}}', encoding="utf-8")

            config = AppConfig.load(config_path)

        self.assertTrue(config.llm.dspy_cache)

    def test_loads_llm_request_timeout(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {"request_timeout_seconds": 600.0}}', encoding="utf-8")

            config = AppConfig.load(config_path)
            self.assertEqual(600.0, config.llm.request_timeout_seconds)

    def test_llm_request_timeout_defaults_to_180(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {}}', encoding="utf-8")

            config = AppConfig.load(config_path)
            self.assertEqual(180.0, config.llm.request_timeout_seconds)

    def test_llm_max_concurrent_requests_defaults_to_one(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {}}', encoding="utf-8")

            config = AppConfig.load(config_path)
            self.assertEqual(1, config.llm.max_concurrent_requests)

    def test_loads_llm_max_concurrent_requests(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text('{"llm": {"max_concurrent_requests": 2}}', encoding="utf-8")

            config = AppConfig.load(config_path)
            self.assertEqual(2, config.llm.max_concurrent_requests)

    def test_loads_llm_api_key_from_adjacent_dotenv(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            config_path = root / "app_config.json"
            config_path.write_text(
                json.dumps({"llm": {}, "global_library_path": str(root / "library")}),
                encoding="utf-8",
            )
            (root / ".env").write_text(
                'LLM_API_KEY="dotenv-secret" # local key\n', encoding="utf-8",
            )

            config = AppConfig.load(config_path)
            self.assertEqual("dotenv-secret", config.llm.api_key)
            self.assertNotIn("LLM_API_KEY", os.environ)

    def test_app_config_key_takes_precedence_over_dotenv(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            config_path = root / "app_config.json"
            config_path.write_text(
                json.dumps({
                    "llm": {"api_key": "json-secret"},
                    "global_library_path": str(root / "library"),
                }),
                encoding="utf-8",
            )
            (root / ".env").write_text("LLM_API_KEY=dotenv-secret\n", encoding="utf-8")

            config = AppConfig.load(config_path)
            self.assertEqual("json-secret", config.llm.api_key)

    def test_process_environment_key_takes_precedence_over_local_config(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ, {"LLM_API_KEY": "environment-secret"}, clear=True,
        ):
            root = Path(temp_dir)
            config_path = root / "app_config.json"
            config_path.write_text(
                json.dumps({
                    "llm": {"api_key": "json-secret"},
                    "global_library_path": str(root / "library"),
                }),
                encoding="utf-8",
            )
            (root / ".env").write_text("LLM_API_KEY=dotenv-secret\n", encoding="utf-8")

            config = AppConfig.load(config_path)

            self.assertEqual("environment-secret", config.llm.api_key)

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

    def test_missing_comfyui_latent_upscaler_device_defaults_to_none(self):
        from feverslop.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertIsNone(config.comfyui.latent_upscaler_device)

    def test_loads_comfyui_latent_upscaler_device(self):
        from feverslop.config.app_config import AppConfig

        for value in ("cuda", "rocm", "cpu", "auto"):
            with self.subTest(device=value):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config_path = Path(temp_dir) / "app_config.json"
                    config_path.write_text(
                        json.dumps({"comfyui": {"latent_upscaler_device": value}}),
                        encoding="utf-8",
                    )

                    config = AppConfig.load(config_path)

                self.assertEqual(value, config.comfyui.latent_upscaler_device)

    def test_invalid_comfyui_latent_upscaler_device_rejected(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                json.dumps({"comfyui": {"latent_upscaler_device": "tpu"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "comfyui.latent_upscaler_device must be 'cuda', 'rocm', 'cpu', or 'auto'",
            ):
                AppConfig.load(config_path)

    def test_non_string_comfyui_latent_upscaler_device_rejected(self):
        from feverslop.config.app_config import AppConfig

        for value in (5, True):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config_path = Path(temp_dir) / "app_config.json"
                    config_path.write_text(
                        json.dumps({"comfyui": {"latent_upscaler_device": value}}),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        "comfyui.latent_upscaler_device must be a string",
                    ):
                        AppConfig.load(config_path)

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
                      "workflow": "workflows/image/image-model/image_t2i_startframe_ideogram_v1.json",
                      "kind": "template",
                      "template": "documentation/ideogram4_prompt_template.md",
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
        self.assertEqual("workflows/image/image-model/image_t2i_startframe_ideogram_v1.json", transform.workflow)
        self.assertEqual("template", transform.kind)
        self.assertEqual("documentation/ideogram4_prompt_template.md", transform.template)
        self.assertEqual("text", transform.positive_prompt_input)
        self.assertEqual("ideogram4_prompt_debug", transform.debug_dir)
        self.assertEqual(150, transform.max_words)

    def test_loads_chat_template_kwargs(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                json.dumps({
                    "llm": {
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                            "reasoning_effort": "xhigh",
                        },
                    },
                }),
                encoding="utf-8",
            )
            config = AppConfig.load(config_path)

        self.assertEqual(
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
            config.llm.chat_template_kwargs,
        )

    def test_rejects_non_dict_chat_template_kwargs(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                '{"llm": {"chat_template_kwargs": "not-a-dict"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "llm.chat_template_kwargs must be an object"):
                AppConfig.load(config_path)


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
