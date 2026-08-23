import unittest
from pathlib import Path

class ProjectStoreCompositionTests(unittest.TestCase):
    def test_canonical_service_does_not_import_studio_package(self):
        source = Path("src/feverslop/composition/project_store.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)
