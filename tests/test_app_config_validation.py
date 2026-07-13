import tempfile
import unittest
from pathlib import Path


class AppConfigValidationTests(unittest.TestCase):
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


class AppConfigBackwardCompatTests(unittest.TestCase):
    def test_no_required_keys_allows_missing_file(self):
        from feverslop.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertEqual("http://localhost:8080/v1", config.llm.base_url)


if __name__ == "__main__":
    unittest.main()
