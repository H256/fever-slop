import unittest
from pathlib import Path

class JobRuntimeCompositionTests(unittest.TestCase):
    def test_canonical_runtime_does_not_import_studio_package(self):
        source = Path("src/feverslop/composition/job_runtime.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)
