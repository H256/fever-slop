import unittest

from feverslop.application.project_requests import (
    ArtifactConflict,
    ArtifactRequest,
    ProjectCreateRequest,
    RenderPlanPatch,
    StudioPathError,
    sanitize_audio_filename,
)
from feverslop.studio.projects import (
    ArtifactConflict as LegacyArtifactConflict,
    ArtifactRequest as LegacyArtifactRequest,
    ProjectCreateRequest as LegacyProjectCreateRequest,
    RenderPlanPatch as LegacyRenderPlanPatch,
    StudioPathError as LegacyStudioPathError,
    sanitize_audio_filename as legacy_sanitize_audio_filename,
)


class ProjectPersistenceContractTests(unittest.TestCase):
    def test_persistence_contracts_have_canonical_application_owner(self):
        self.assertIs(LegacyArtifactConflict, ArtifactConflict)
        self.assertIs(LegacyArtifactRequest, ArtifactRequest)
        self.assertIs(LegacyProjectCreateRequest, ProjectCreateRequest)
        self.assertIs(LegacyRenderPlanPatch, RenderPlanPatch)
        self.assertIs(LegacyStudioPathError, StudioPathError)
        self.assertIs(legacy_sanitize_audio_filename, sanitize_audio_filename)
        self.assertEqual("bad_name.mp3", sanitize_audio_filename(r"../bad name.mp3"))


if __name__ == "__main__":
    unittest.main()
