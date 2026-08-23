import unittest

from feverslop.composition.project_repository import ProjectRepository
from feverslop.studio.project_repository import ProjectRepository as LegacyProjectRepository


class ProjectRepositoryCompositionTests(unittest.TestCase):
    def test_legacy_import_uses_canonical_composition_service(self):
        self.assertIs(LegacyProjectRepository, ProjectRepository)
