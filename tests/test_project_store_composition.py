import unittest
from pathlib import Path

from feverslop.studio.projects import ProjectStore as LegacyProjectStore


class ProjectStoreCompositionTests(unittest.TestCase):
    def test_legacy_import_uses_canonical_composition_service(self):
        from feverslop.composition.project_store import ProjectStore

        self.assertIs(LegacyProjectStore, ProjectStore)

    def test_canonical_service_does_not_import_studio_package(self):
        source = Path("src/feverslop/composition/project_store.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)

