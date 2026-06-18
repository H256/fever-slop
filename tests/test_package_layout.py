import importlib
import unittest


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


if __name__ == "__main__":
    unittest.main()
