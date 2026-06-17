import importlib
import unittest


class ToolsImportTests(unittest.TestCase):
    def test_tool_modules_import(self):
        for module_name in (
            "tools.normalize_render_plan",
            "tools.render_plan_normalizer",
            "tools.repair_scene_srt",
            "tools.trim_existing_ltx_clips",
        ):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
