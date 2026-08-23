import unittest
from pathlib import Path

from feverslop.studio.job_service import StudioJobService as LegacyStudioJobService


class JobServiceCompositionTests(unittest.TestCase):
    def test_legacy_import_uses_canonical_composition_service(self):
        from feverslop.composition.job_service import StudioJobService

        self.assertIs(LegacyStudioJobService, StudioJobService)

    def test_canonical_service_does_not_import_studio_package(self):
        source = Path("src/feverslop/composition/job_service.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)

