import tempfile
import unittest
from pathlib import Path

from feverslop.application.prompt_revisions import (
    LoadPromptHistoryUseCase,
    PatchPromptError,
    PatchPromptUseCase,
)
from feverslop.domain.prompt_revisions import (
    PromptField,
    PromptHistory,
    PromptRevision,
)
from feverslop.infra.sqlite_adapter import SqliteRevisionStore
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


class RebuildServiceUseCaseChainTests(unittest.TestCase):
    """Integration tests for PatchPromptUseCase → SqliteRevisionStore → LoadPromptHistoryUseCase.

    Verifies the use-case layer (not the service layer) with a real SQLite store
    so that schema initialization, serialization, and round-trip persistence are
    validated end-to-end.
    """

    def test_patch_prompt_and_load_history_round_trip(self):
        """PatchPromptUseCase saves revision; LoadPromptHistoryUseCase retrieves it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "revisions.sqlite")
            store = SqliteRevisionStore(db_path)

            patch = PatchPromptUseCase(store=store)
            load = LoadPromptHistoryUseCase(store=store)

            # Patch
            revision = patch.execute(
                project_id="use-case-test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
                value="patched prompt",
            )

            self.assertEqual(revision.value, "patched prompt")
            self.assertEqual(revision.project_id, "use-case-test")
            self.assertEqual(revision.scene_number, 1)

            # Load back
            result = load.execute(
                project_id="use-case-test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
            )

            self.assertEqual(len(result.history.revisions), 1)
            self.assertEqual(result.history.revisions[0].value, "patched prompt")
            self.assertIn(PromptField.Z_IMAGE_PROMPT, result.available_fields)

    def test_multiple_patches_create_chain_loaded_by_history(self):
        """Multiple PatchPromptUseCase calls build a revision chain in SQLite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "revisions.sqlite")
            store = SqliteRevisionStore(db_path)

            patch = PatchPromptUseCase(store=store)
            load = LoadPromptHistoryUseCase(store=store)

            r1 = patch.execute(
                project_id="chain-test",
                scene_number=2,
                field=PromptField.I2V_PROMPT,
                value="version 1",
            )
            r2 = patch.execute(
                project_id="chain-test",
                scene_number=2,
                field=PromptField.I2V_PROMPT,
                value="version 2",
            )

            # r2 should reference r1 as parent
            self.assertEqual(r2.parent_id, r1.id)

            # Load via the other use case
            result = load.execute(
                project_id="chain-test",
                scene_number=2,
                field=PromptField.I2V_PROMPT,
            )

            self.assertEqual(len(result.history.revisions), 2)
            self.assertEqual(result.history.revisions[0].value, "version 1")
            self.assertEqual(result.history.revisions[1].value, "version 2")
            self.assertEqual(result.history.revisions[1].parent_id, result.history.revisions[0].id)

    def test_patch_rejects_unchanged_value(self):
        """PatchPromptUseCase raises PatchPromptError on no-op patch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "revisions.sqlite")
            store = SqliteRevisionStore(db_path)

            patch = PatchPromptUseCase(store=store)

            patch.execute(
                project_id="reject-test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
                value="initial",
            )

            with self.assertRaises(PatchPromptError):
                patch.execute(
                    project_id="reject-test",
                    scene_number=1,
                    field=PromptField.Z_IMAGE_PROMPT,
                    value="initial",
                )

    def test_list_fields_reports_all_patched_fields(self):
        """LoadPromptHistoryUseCase returns all fields patched for that scene."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "revisions.sqlite")
            store = SqliteRevisionStore(db_path)

            patch = PatchPromptUseCase(store=store)
            load = LoadPromptHistoryUseCase(store=store)

            patch.execute(
                project_id="fields-test",
                scene_number=3,
                field=PromptField.Z_IMAGE_PROMPT,
                value="image",
            )
            patch.execute(
                project_id="fields-test",
                scene_number=3,
                field=PromptField.I2V_PROMPT,
                value="video",
            )

            result = load.execute(
                project_id="fields-test",
                scene_number=3,
                field=PromptField.Z_IMAGE_PROMPT,
            )

            self.assertIn(PromptField.Z_IMAGE_PROMPT, result.available_fields)
            self.assertIn(PromptField.I2V_PROMPT, result.available_fields)

    def test_persistence_survives_store_reopen(self):
        """Revisions survive reopening the SqliteRevisionStore (new process simulation)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "revisions.sqlite")

            # First store session
            store1 = SqliteRevisionStore(db_path)
            patch1 = PatchPromptUseCase(store=store1)
            patch1.execute(
                project_id="reopen-test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
                value="persisted value",
            )

            # Second store session (fresh object, simulates restart)
            store2 = SqliteRevisionStore(db_path)
            load2 = LoadPromptHistoryUseCase(store=store2)
            result = load2.execute(
                project_id="reopen-test",
                scene_number=1,
                field=PromptField.Z_IMAGE_PROMPT,
            )

            self.assertEqual(len(result.history.revisions), 1)
            self.assertEqual(result.history.revisions[0].value, "persisted value")


class RebuildServiceSqliteIntegrationTests(unittest.TestCase):
    def test_saved_and_restored_revisions_persist_across_service_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "revisions.sqlite"
            project_id = "integration-project"
            service = RebuildService(store=SqliteRevisionStore(str(db_path)))

            original = service.save_prompt(
                project_id=project_id,
                scene_number=4,
                field=PromptField.I2V_PROMPT,
                value="original prompt",
            )
            service.save_prompt(
                project_id=project_id,
                scene_number=4,
                field=PromptField.I2V_PROMPT,
                value="edited prompt",
            )
            restored = service.restore_revision(
                project_id=project_id,
                scene_number=4,
                field=PromptField.I2V_PROMPT,
                revision_id=original.revision.id,
            )

            reloaded_service = RebuildService(store=SqliteRevisionStore(str(db_path)))
            history = reloaded_service.get_history(
                project_id=project_id,
                scene_number=4,
                field=PromptField.I2V_PROMPT,
            ).history

            self.assertEqual(
                ["original prompt", "edited prompt", "original prompt"],
                [revision.value for revision in history.revisions],
            )
            self.assertEqual(original.revision.id, restored.revision.restored_from)
            self.assertEqual(original.revision.id, history.revisions[-1].restored_from)


if __name__ == "__main__":
    unittest.main()
