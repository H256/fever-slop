import unittest

from feverslop.domain.prompt_revisions import (
    PromptField,
    PromptHistory,
    PromptRevision,
)
from feverslop.studio.rebuild_service import (
    PromptSaveConflict,
    RebuildService,
    RevisionSaveResult,
)


class _FakeRevisionStore:
    """Minimal in-memory revision store for testing."""

    def __init__(self) -> None:
        self._revisions: dict[tuple[str, int, str], list[PromptRevision]] = {}
        self._clock_calls: list[str] = []

    def load_history(self, project_id: str, scene_number: int, field: PromptField) -> PromptHistory:
        key = (project_id, scene_number, field.value)
        return PromptHistory(
            scene_number=scene_number,
            field=field,
            revisions=tuple(self._revisions.get(key, [])),
        )

    def save_revision(self, revision: PromptRevision) -> None:
        key = (revision.project_id, revision.scene_number, revision.field.value)
        if key not in self._revisions:
            self._revisions[key] = []
        self._revisions[key].append(revision)

    def list_fields(self, project_id: str, scene_number: int) -> list[PromptField]:
        keys = [k for k in self._revisions if k[0] == project_id and k[1] == scene_number]
        return [PromptField(k[2]) for k in keys]


class RebuildServicePromptSaveTests(unittest.TestCase):
    def test_save_prompt_revision(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        result = service.save_prompt(
            project_id="test",
            scene_number=1,
            field=PromptField.Z_IMAGE_PROMPT,
            value="new prompt value",
        )

        self.assertIsInstance(result, RevisionSaveResult)
        self.assertEqual(result.revision.value, "new prompt value")
        self.assertEqual(result.revision.scene_number, 1)
        self.assertTrue(result.changed)

    def test_save_prompt_increments_revision(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        service.save_prompt(
            project_id="test",
            scene_number=1,
            field=PromptField.Z_IMAGE_PROMPT,
            value="v1",
        )

        result = service.save_prompt(
            project_id="test",
            scene_number=1,
            field=PromptField.Z_IMAGE_PROMPT,
            value="v2",
        )

        self.assertEqual(result.revision.value, "v2")
        self.assertIsNotNone(result.revision.parent_id)

    def test_save_rejects_blank(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        with self.assertRaises(PromptSaveConflict):
            service.save_prompt(
                project_id="test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
                value="",
            )

    def test_save_rejects_unchanged(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        service.save_prompt(
            project_id="test",
            scene_number=1,
            field=PromptField.Z_IMAGE_PROMPT,
            value="current",
        )

        with self.assertRaises(PromptSaveConflict):
            service.save_prompt(
                project_id="test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
                value="current",
            )

    def test_save_returns_changed_flag(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        service.save_prompt(
            project_id="test",
            scene_number=1,
            field=PromptField.Z_IMAGE_PROMPT,
            value="v1",
        )

        result = service.save_prompt(
            project_id="test",
            scene_number=1,
            field=PromptField.Z_IMAGE_PROMPT,
            value="v2",
        )

        self.assertTrue(result.changed)


class RebuildServiceHistoryTests(unittest.TestCase):
    def test_get_history(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        service.save_prompt(
            project_id="test",
            scene_number=2,
            field=PromptField.Z_IMAGE_PROMPT,
            value="test",
        )

        load_result = service.get_history(
            project_id="test",
            scene_number=2,
            field=PromptField.Z_IMAGE_PROMPT,
        )

        self.assertEqual(len(load_result.history.revisions), 1)
        self.assertEqual(load_result.history.revisions[0].value, "test")


class RebuildServiceRestoreTests(unittest.TestCase):
    def test_restore_creates_new_revision(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        first = service.save_prompt(
            project_id="test",
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="original",
        )

        result = service.restore_revision(
            project_id="test",
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            revision_id=first.revision.id,
        )

        self.assertIsInstance(result, RevisionSaveResult)
        self.assertEqual(result.revision.restored_from, first.revision.id)
        self.assertNotEqual(result.revision.id, first.revision.id)

    def test_restore_invalid_id_raises(self):
        store = _FakeRevisionStore()
        service = RebuildService(store=store)

        with self.assertRaises(PromptSaveConflict):
            service.restore_revision(
                project_id="test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
                revision_id="nonexistent",
            )


if __name__ == "__main__":
    unittest.main()
