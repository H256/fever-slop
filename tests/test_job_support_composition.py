import unittest
from pathlib import Path


class JobSupportCompositionTests(unittest.TestCase):
    def test_canonical_helpers_do_not_import_studio_package(self):
        modules = [
            Path(f"src/feverslop/composition/{module_name}.py")
            for module_name in ("logging", "pipeline_actions")
        ]
        existing = [path for path in modules if path.is_file()]
        if not existing:
            self.skipTest("canonical helper modules were removed; studio package is gone")
        for path in existing:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("feverslop.studio", source)
