import unittest
from pathlib import Path

from feverslop.studio.reference_workspace_service import (
    ReferenceWorkspaceService as LegacyReferenceWorkspaceService,
)


class ReferenceWorkspaceCompositionServiceTests(unittest.TestCase):
    def test_legacy_import_uses_canonical_composition_service(self):
        from feverslop.composition.reference_workspace_service import ReferenceWorkspaceService

        self.assertIs(LegacyReferenceWorkspaceService, ReferenceWorkspaceService)

    def test_canonical_service_does_not_import_studio_package(self):
        source = Path("src/feverslop/composition/reference_workspace_service.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)

