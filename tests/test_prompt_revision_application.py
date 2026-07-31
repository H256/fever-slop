import datetime
import unittest
from typing import Any

from feverslop.application.prompt_revisions import (
    HistoryLoadResult,
    LoadPromptHistoryUseCase,
    PatchPromptError,
    PatchPromptUseCase,
    RestoreRevisionUseCase,
)
from feverslop.domain.prompt_revisions import (
    PromptField,
    PromptHistory,
    PromptRevision,
)
from feverslop.ports.revision_store import (
    DuplicateRevisionError,
    RevisionStorePort,
)


class _MemoryRevisionStore(RevisionStorePort):
    """In-memory revision store for testing."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, str], list[dict[str, Any]]] = {}

    def save_revision(self, revision: PromptRevision) -> None:
        key = (revision.scene_number, revision.field.value)
        record = revision.__dict__.copy()
        record["created_at_iso"] = revision.created_at.isoformat()
        existing_ids = {r.get("id") for r in self._data.get(key, [])}
        if revision.id in existing_ids:
            raise DuplicateRevisionError(revision.id)
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(record)

    def load_history(self, scene_number: int, field: PromptField) -> PromptHistory:
        key = (scene_number, field.value)
        records = self._data.get(key, [])
        revisions: list[PromptRevision] = []
        for rec in records:
            rev = PromptRevision(
                id=rec["id"],
                scene_number=rec["scene_number"],
                field=rec["field"],
                value=rec["value"],
                parent_id=rec["parent_id"],
                restored_from=rec["restored_from"],
                content_hash=rec["content_hash"],
                created_at=datetime.datetime.fromisoformat(rec["created_at_iso"]),
            )
            revisions.append(rev)
        return PromptHistory(scene_number=scene_number, field=field, revisions=tuple(revisions))

    def list_fields(self, scene_number: int) -> list[PromptField]:
        fields = set()
        for (sn, fv) in self._data:
            if sn == scene_number:
                fields.add(PromptField(fv))
        return list(fields)


class PatchPromptUseCaseTests(unittest.TestCase):
    def test_patch_creates_revision(self):
        store = _MemoryRevisionStore()
        use_case = PatchPromptUseCase(store=store, clock=lambda: datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc))

        result = use_case.execute(
            project_id="test-project",
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
        )

        self.assertIsInstance(result, PromptRevision)
        self.assertEqual(result.scene_number, 3)
        self.assertEqual(result.value, "A singer on stage")

    def test_patch_follows_previous_revision(self):
        store = _MemoryRevisionStore()
        clock = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        use_case = PatchPromptUseCase(store=store, clock=lambda: clock)

        first = use_case.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v1")
        second = use_case.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v2")

        self.assertEqual(second.parent_id, first.id)

    def test_patch_rejects_blank_value(self):
        store = _MemoryRevisionStore()
        use_case = PatchPromptUseCase(store=store, clock=lambda: datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc))

        with self.assertRaises(PatchPromptError):
            use_case.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="")

    def test_patch_duplicate_value_fails_after_readd(self):
        store = _MemoryRevisionStore()
        clock = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        use_case = PatchPromptUseCase(store=store, clock=lambda: clock)

        use_case.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="same value")
        with self.assertRaises(PatchPromptError):
            use_case.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="same value")


class LoadPromptHistoryUseCaseTests(unittest.TestCase):
    def test_load_empty_history(self):
        store = _MemoryRevisionStore()
        use_case = LoadPromptHistoryUseCase(store=store)

        result = use_case.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT)

        self.assertIsInstance(result, HistoryLoadResult)
        self.assertEqual(result.history.scene_number, 3)
        self.assertEqual(len(result.history.revisions), 0)

    def test_load_with_revisions(self):
        store = _MemoryRevisionStore()
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        patch = PatchPromptUseCase(store=store, clock=lambda: ts)
        use_case = LoadPromptHistoryUseCase(store=store)

        patch.execute(project_id="project", scene_number=1, field=PromptField.Z_IMAGE_PROMPT, value="initial")

        result = use_case.execute(project_id="project", scene_number=1, field=PromptField.Z_IMAGE_PROMPT)

        self.assertEqual(len(result.history.revisions), 1)
        self.assertEqual(result.history.revisions[0].value, "initial")

    def test_load_available_fields(self):
        store = _MemoryRevisionStore()
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        patch = PatchPromptUseCase(store=store, clock=lambda: ts)
        use_case = LoadPromptHistoryUseCase(store=store)

        patch.execute(project_id="project", scene_number=1, field=PromptField.Z_IMAGE_PROMPT, value="v1")
        patch.execute(project_id="project", scene_number=1, field=PromptField.I2V_PROMPT, value="i2v1")

        result = use_case.execute(project_id="project", scene_number=1, field=PromptField.Z_IMAGE_PROMPT)

        self.assertIn(PromptField.Z_IMAGE_PROMPT, result.available_fields)
        self.assertIn(PromptField.I2V_PROMPT, result.available_fields)


class RestoreRevisionUseCaseTests(unittest.TestCase):
    def test_restore_creates_new_revision(self):
        store = _MemoryRevisionStore()
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)

        patch = PatchPromptUseCase(store=store, clock=lambda: ts1)
        restore = RestoreRevisionUseCase(store=store, clock=lambda: ts2)

        original = patch.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="initial")
        patch.execute(project_id="project", scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="overridden")

        restored = restore.execute(
            project_id="project",
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            revision_id=original.id,
        )

        self.assertEqual(restored.value, "initial")
        self.assertEqual(restored.restored_from, original.id)
        self.assertNotEqual(restored.id, original.id)

    def test_restore_invalid_id_raises(self):
        store = _MemoryRevisionStore()
        restore = RestoreRevisionUseCase(
            store=store,
            clock=lambda: datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc),
        )

        with self.assertRaises(ValueError):
            restore.execute(
                project_id="project",
                scene_number=3,
                field=PromptField.Z_IMAGE_PROMPT,
                revision_id="nonexistent",
            )


if __name__ == "__main__":
    unittest.main()
