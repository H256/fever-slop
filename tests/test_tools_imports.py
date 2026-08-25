import importlib
import unittest


class ToolsImportTests(unittest.TestCase):
    def test_tool_modules_import(self):
        for module_name in (
            "tools.normalize_render_plan",
            "tools.project_asset_archive",
            "tools.render_plan_normalizer",
            "tools.repair_scene_srt",
            "tools.trim_existing_ltx_clips",
        ):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_legacy_normalize_facade_preserves_public_symbols(self):
        module = importlib.import_module("tools.normalize_render_plan")
        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.normalize_render_plan_file))
        self.assertIsNotNone(module.console)

    def test_legacy_facades_expose_console_seams(self):
        for module_name in ("tools.repair_scene_srt", "tools.trim_existing_ltx_clips"):
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.import_module(module_name).console)


if __name__ == "__main__":
    unittest.main()
