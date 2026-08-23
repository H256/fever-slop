import unittest
from pathlib import Path


class JobSupportCompositionTests(unittest.TestCase):
    def test_canonical_helpers_do_not_import_studio_package(self):
        for module_name in ("logging", "pipeline_actions"):
            source = Path(f"src/feverslop/composition/{module_name}.py").read_text(encoding="utf-8")
            self.assertNotIn("feverslop.studio", source)
