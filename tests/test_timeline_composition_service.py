import unittest
from pathlib import Path

from feverslop.studio.timeline_service import TimelineStudioService as LegacyTimelineService


class TimelineCompositionServiceTests(unittest.TestCase):
    def test_legacy_import_uses_canonical_composition_service(self):
        from feverslop.composition.timeline_service import TimelineStudioService

        self.assertIs(LegacyTimelineService, TimelineStudioService)

    def test_canonical_service_does_not_import_studio_package(self):
        source = Path("src/feverslop/composition/timeline_service.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)

