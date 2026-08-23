import unittest

from feverslop.composition.project_repository import ProjectRepository


class ProjectRepositoryCompositionTests(unittest.TestCase):
    def test_project_repository_is_available_from_composition(self):
        self.assertTrue(callable(ProjectRepository))
