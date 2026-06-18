import importlib
import unittest
from pathlib import Path


class PackageLayoutTests(unittest.TestCase):
    def test_autoprompter_package_imports_application_boundaries(self):
        modules = [
            "autoprompter.application.generate_render_plan",
            "autoprompter.adapters.comfyui_rendering",
            "autoprompter.domain.render_plan",
            "autoprompter.ports.rendering",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_root_compatibility_imports_remain_available(self):
        modules = [
            "main",
            "render_ltx",
            "render_storyboard",
            "ltx_video_renderer",
            "storyboard_renderer",
            "workflow_patcher",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_autoprompter_package_modules_are_real_files_under_src(self):
        import autoprompter.application.generate_render_plan as module

        module_path = Path(module.__file__).as_posix()

        self.assertIn("src", module_path)
        self.assertIn("autoprompter/application/generate_render_plan.py", module_path)

    def test_config_modules_resolve_under_src_package(self):
        import autoprompter.config.project_config as project_config
        import autoprompter.config.video_settings as video_settings

        self.assertIn(
            "src/autoprompter/config/project_config.py",
            Path(project_config.__file__).as_posix(),
        )
        self.assertIn(
            "src/autoprompter/config/video_settings.py",
            Path(video_settings.__file__).as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
