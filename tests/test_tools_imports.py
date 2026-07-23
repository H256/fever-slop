import importlib
from pathlib import Path
import subprocess
import sys
import unittest


class ToolsImportTests(unittest.TestCase):
    def test_tool_modules_import(self):
        for module_name in (
            "tools.normalize_render_plan",
            "tools.project_asset_archive",
            "tools.render_plan_normalizer",
            "tools.repair_scene_srt",
            "tools.trim_existing_ltx_clips",
            "feverslop.tools.benchmark_video_workflows",
        ):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_benchmark_video_workflows_module_is_executable(self):
        completed = subprocess.run(
            [sys.executable, "-m", "feverslop.tools.benchmark_video_workflows", "--help"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--case", completed.stdout)
        self.assertIn("--comfyui-url", completed.stdout)


if __name__ == "__main__":
    unittest.main()
