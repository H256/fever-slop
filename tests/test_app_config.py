import tempfile
import unittest
from pathlib import Path


class AppConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
