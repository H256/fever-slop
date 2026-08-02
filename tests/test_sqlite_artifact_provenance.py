"""Tests for SQLite artifact provenance store."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from uuid import uuid4

from feverslop.domain.rebuild_policy import ArtifactFingerprint, ArtifactKind
from feverslop.infra.sqlite_adapter import SqliteArtifactProvenance


class TestSqliteArtifactProvenance(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._dir, "provenance.db")
        self.store = SqliteArtifactProvenance(self._db_path)
        self.project_id = uuid4().hex[:12]

    def tearDown(self) -> None:
        for path in [self._db_path, self._db_path + "-wal", self._db_path + "-shm"]:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(self._dir)

    def test_record_and_load_fingerprint(self) -> None:
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=3,
            prompt_hash="abc123",
            workflow_hash="wf001",
        )
        self.store.record_fingerprint(self.project_id, fp)
        loaded = self.store.load_fingerprint(self.project_id, ArtifactKind.SCENE_RENDER, 3)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual("abc123", loaded.prompt_hash)
        self.assertEqual("wf001", loaded.workflow_hash)

    def test_load_missing_fingerprint_returns_none(self) -> None:
        result = self.store.load_fingerprint(self.project_id, ArtifactKind.FINAL_VIDEO)
        self.assertIsNone(result)

    def test_load_all_fingerprints(self) -> None:
        fps = [
            ArtifactFingerprint(artifact_kind=ArtifactKind.SCENE_RENDER, scene_number=1, prompt_hash="p1"),
            ArtifactFingerprint(artifact_kind=ArtifactKind.SCENE_RENDER, scene_number=2, prompt_hash="p2"),
            ArtifactFingerprint(artifact_kind=ArtifactKind.FINAL_VIDEO, scene_number=None),
        ]
        for fp in fps:
            self.store.record_fingerprint(self.project_id, fp)
        all_fps = self.store.load_fingerprints(self.project_id)
        self.assertEqual(3, len(all_fps))

    def test_project_fingerprint_list_uses_recorded_at_index(self) -> None:
        self.store.record_fingerprint(
            self.project_id,
            ArtifactFingerprint(
                artifact_kind=ArtifactKind.FINAL_VIDEO,
                scene_number=None,
            ),
        )
        conn = sqlite3.connect(self._db_path)
        plan = conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM artifact_provenance
            WHERE project_id = ?
            ORDER BY recorded_at DESC
            """,
            (self.project_id,),
        ).fetchall()
        conn.close()

        self.assertTrue(
            any("idx_artifact_provenance_project_recorded_at" in row[3] for row in plan),
            plan,
        )

    def test_duplicate_key_replaces(self) -> None:
        fp1 = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=5,
            prompt_hash="old",
        )
        fp2 = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=5,
            prompt_hash="new",
        )
        self.store.record_fingerprint(self.project_id, fp1)
        self.store.record_fingerprint(self.project_id, fp2)
        loaded = self.store.load_fingerprint(self.project_id, ArtifactKind.SCENE_RENDER, 5)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual("new", loaded.prompt_hash)
        all_fps = self.store.load_fingerprints(self.project_id)
        self.assertEqual(1, len(all_fps))

    def test_global_artifact_without_scene(self) -> None:
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.FINAL_VIDEO,
            scene_number=None,
            workflow_hash="wf999",
        )
        self.store.record_fingerprint(self.project_id, fp)
        loaded = self.store.load_fingerprint(self.project_id, ArtifactKind.FINAL_VIDEO)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual("wf999", loaded.workflow_hash)

    def test_different_projects_isolated(self) -> None:
        fp_a = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="project_a",
        )
        fp_b = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="project_b",
        )
        self.store.record_fingerprint("proj_a", fp_a)
        self.store.record_fingerprint("proj_b", fp_b)
        la = self.store.load_fingerprint("proj_a", ArtifactKind.SCENE_RENDER, 1)
        lb = self.store.load_fingerprint("proj_b", ArtifactKind.SCENE_RENDER, 1)
        assert la is not None
        assert lb is not None
        self.assertEqual("project_a", la.prompt_hash)
        self.assertEqual("project_b", lb.prompt_hash)

    def test_load_all_empty_project(self) -> None:
        self.assertEqual([], self.store.load_fingerprints("nonexistent"))

    def test_concurrent_schema_creation(self) -> None:
        """Multiple connections can co-exist safely."""
        store2 = SqliteArtifactProvenance(self._db_path)
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.AUDIO_TIMELINE,
            scene_number=None,
            workflow_hash="shared",
        )
        store2.record_fingerprint(self.project_id, fp)
        loaded = self.store.load_fingerprint(self.project_id, ArtifactKind.AUDIO_TIMELINE)
        self.assertIsNotNone(loaded)

    def test_all_fingerprint_fields_preserved(self) -> None:
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=7,
            prompt_hash="ph",
            workflow_hash="wh",
            reference_hash="rh",
            timeline_hash="th",
            dimensions_hash="dh",
        )
        self.store.record_fingerprint(self.project_id, fp)
        loaded = self.store.load_fingerprint(self.project_id, ArtifactKind.SCENE_RENDER, 7)
        assert loaded is not None
        self.assertEqual("ph", loaded.prompt_hash)
        self.assertEqual("wh", loaded.workflow_hash)
        self.assertEqual("rh", loaded.reference_hash)
        self.assertEqual("th", loaded.timeline_hash)
        self.assertEqual("dh", loaded.dimensions_hash)


if __name__ == "__main__":
    unittest.main()
