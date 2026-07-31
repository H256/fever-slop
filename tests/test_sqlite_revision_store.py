import datetime
import os
import sqlite3
import tempfile
import unittest

from feverslop.domain.prompt_revisions import (
    PromptField,
    PromptRevision,
    build_revision,
)
from feverslop.infra.sqlite_adapter import (
    DuplicateRevisionError,
    SqliteRevisionStore,
    ensure_schema,
)


class _make_revision:
    @staticmethod
    def create(
        *,
        project_id: str = "proj1",
        scene_number: int = 1,
        field: PromptField = PromptField.Z_IMAGE_PROMPT,
        value: str = "test prompt",
        parent_id: str | None = None,
        now: datetime.datetime | None = None,
    ) -> PromptRevision:
        timestamp = now or datetime.datetime.now(datetime.timezone.utc)
        return build_revision(
            project_id=project_id,
            scene_number=scene_number,
            field=field,
            value=value,
            parent_id=parent_id,
            now=timestamp,
        )


@unittest.skipIf(os.environ.get("SKIP_DB_TESTS"), "Database tests disabled")
class SqliteRevisionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_file.close()
        conn = sqlite3.connect(self.db_file.name)
        ensure_schema(conn)
        conn.close()

    def tearDown(self) -> None:
        if os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)

    def test_save_and_load(self):
        store = SqliteRevisionStore(self.db_file.name)
        revision = _make_revision.create(value="initial prompt")
        store.save_revision(revision)

        history = store.load_history("proj1", revision.scene_number, revision.field)

        self.assertEqual(len(history.revisions), 1)
        self.assertEqual(history.revisions[0].value, "initial prompt")

    def test_save_duplicate_raises(self):
        store = SqliteRevisionStore(self.db_file.name)
        revision = _make_revision.create(value="dup test")
        store.save_revision(revision)

        with self.assertRaises(DuplicateRevisionError):
            store.save_revision(revision)

    def test_load_multiple_revisions_ordered(self):
        store = SqliteRevisionStore(self.db_file.name)
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)

        rev1 = _make_revision.create(value="v1", now=ts1)
        rev2 = _make_revision.create(value="v2", parent_id=rev1.id, now=ts2)
        store.save_revision(rev1)
        store.save_revision(rev2)

        history = store.load_history("proj1", rev1.scene_number, rev1.field)

        self.assertEqual(len(history.revisions), 2)
        self.assertEqual(history.revisions[0].value, "v1")
        self.assertEqual(history.revisions[1].value, "v2")

    def test_list_fields(self):
        store = SqliteRevisionStore(self.db_file.name)
        store.save_revision(_make_revision.create(value="z", field=PromptField.Z_IMAGE_PROMPT, scene_number=5))
        store.save_revision(_make_revision.create(value="i", field=PromptField.I2V_PROMPT, scene_number=5))

        fields = store.list_fields("proj1", 5)
        self.assertIn(PromptField.Z_IMAGE_PROMPT, fields)
        self.assertIn(PromptField.I2V_PROMPT, fields)

    def test_list_fields_empty(self):
        store = SqliteRevisionStore(self.db_file.name)

        fields = store.list_fields("proj1", 99)
        self.assertEqual(fields, [])

    def test_load_empty_history(self):
        store = SqliteRevisionStore(self.db_file.name)

        history = store.load_history("proj1", 99, PromptField.Z_IMAGE_PROMPT)
        self.assertEqual(len(history.revisions), 0)

    def test_concurrent_writes(self):
        store1 = SqliteRevisionStore(self.db_file.name)
        store2 = SqliteRevisionStore(self.db_file.name)

        rev1 = _make_revision.create(value="concurrent1", scene_number=10)
        rev2 = _make_revision.create(value="concurrent2", scene_number=10)
        store1.save_revision(rev1)
        store2.save_revision(rev2)

        history = store2.load_history("proj1", 10, PromptField.Z_IMAGE_PROMPT)
        self.assertEqual(len(history.revisions), 2)


@unittest.skipIf(os.environ.get("SKIP_DB_TESTS"), "Database tests disabled")
class EnsureSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_file.close()

    def tearDown(self) -> None:
        if os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)

    def test_schema_creates_tables(self):
        conn = sqlite3.connect(self.db_file.name)
        ensure_schema(conn)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        self.assertIn("prompt_revisions", tables)

    def test_idempotent(self):
        conn = sqlite3.connect(self.db_file.name)
        ensure_schema(conn)
        conn.close()
        conn = sqlite3.connect(self.db_file.name)
        ensure_schema(conn)
        conn.close()


if __name__ == "__main__":
    import sqlite3
    unittest.main()
