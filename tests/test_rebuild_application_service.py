import unittest
from pathlib import Path

from feverslop.studio.rebuild_service import RebuildService as LegacyRebuildService


class RebuildApplicationServiceTests(unittest.TestCase):
    def test_legacy_import_uses_canonical_application_service(self):
        from feverslop.application.rebuild_service import RebuildService

        self.assertIs(LegacyRebuildService, RebuildService)

    def test_canonical_service_does_not_import_studio_package(self):
        source = Path("src/feverslop/application/rebuild_service.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)

