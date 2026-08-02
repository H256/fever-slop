from __future__ import annotations

import datetime
import sqlite3
from contextlib import contextmanager
from typing import Generator

from feverslop.domain.prompt_revisions import (
    DuplicateRevisionError,
    PromptField,
    PromptHistory,
    PromptRevision,
)
from feverslop.domain.rebuild_policy import ArtifactFingerprint, ArtifactKind
from feverslop.ports.rebuild_execution import ArtifactProvenancePort
from feverslop.ports.revision_store import RevisionStorePort

_UTC = datetime.timezone.utc

SCHEMA_VERSION = 3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompt_revisions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    scene_number INTEGER NOT NULL,
    field TEXT NOT NULL,
    value TEXT NOT NULL,
    parent_id TEXT,
    restored_from TEXT,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_revisions_history
    ON prompt_revisions (project_id, scene_number, field, created_at);

CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    scene_number INTEGER,
    prompt_hash TEXT,
    workflow_hash TEXT,
    reference_hash TEXT,
    timeline_hash TEXT,
    dimensions_hash TEXT,
    recorded_at TEXT NOT NULL,
    UNIQUE(project_id, artifact_kind, scene_number)
);

CREATE INDEX IF NOT EXISTS idx_artifact_provenance_project_recorded_at
    ON artifact_provenance (project_id, recorded_at DESC);

"""


def ensure_schema(
    connection: sqlite3.Connection | None = None,
) -> sqlite3.Connection:
    """Ensure the database schema is up to date. Opens a new connection if none provided."""
    if connection is None:
        raise ValueError("Database connection required")

    connection.executescript(SCHEMA_SQL)
    connection.execute("DROP INDEX IF EXISTS idx_prompt_revisions_scene_field")
    connection.execute("DROP INDEX IF EXISTS idx_artifact_provenance_project")
    connection.execute("DROP INDEX IF EXISTS idx_artifact_provenance_kind")

    # Migrate v1 -> v2: add project_id column if missing.
    # NOTE: Legacy rows that were created before v2 will get project_id="" after
    # this migration. These rows become invisible to queries filtered by a real
    # project_id. Users with pre-v2 databases should expect orphaned revision
    # history until they re-create the project and new revisions are saved.
    try:
        cursor = connection.execute(
            "PRAGMA table_info(prompt_revisions)"
        )
        columns = {row["name"] for row in cursor.fetchall()}
        if "project_id" not in columns:
            connection.execute("ALTER TABLE prompt_revisions ADD COLUMN project_id TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass

    # Record schema version
    try:
        connection.execute(
            "INSERT OR REPLACE INTO schema_versions (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )
    except Exception:
        pass

    connection.commit()
    return connection


class SqliteRevisionStore(RevisionStorePort):
    """SQLite-backed implementation of RevisionStorePort."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a connection with WAL mode and return it."""
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        try:
            yield conn
        finally:
            conn.close()

    def save_revision(self, revision: PromptRevision) -> None:
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO prompt_revisions
                        (id, project_id, scene_number, field, value, parent_id, restored_from, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision.id,
                        revision.project_id,
                        revision.scene_number,
                        revision.field.value,
                        revision.value,
                        revision.parent_id,
                        revision.restored_from,
                        revision.content_hash,
                        revision.created_at.isoformat(),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise DuplicateRevisionError(revision.id) from exc

    def load_history(
        self,
        project_id: str,
        scene_number: int,
        field: PromptField,
    ) -> PromptHistory:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM prompt_revisions
                WHERE project_id = ? AND scene_number = ? AND field = ?
                ORDER BY created_at ASC
                """,
                (project_id, scene_number, field.value),
            )
            rows = cursor.fetchall()

        revisions = [
            PromptRevision(
                id=row["id"],
                project_id=row["project_id"] if "project_id" in row.keys() else "",
                scene_number=row["scene_number"],
                field=PromptField(row["field"]),
                value=row["value"],
                parent_id=row["parent_id"],
                restored_from=row["restored_from"],
                content_hash=row["content_hash"],
                created_at=datetime.datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

        return PromptHistory(scene_number=scene_number, field=field, revisions=tuple(revisions))

    def list_fields(self, project_id: str, scene_number: int) -> list[PromptField]:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT field FROM prompt_revisions
                WHERE project_id = ? AND scene_number = ?
                """,
                (project_id, scene_number),
            )
            return [PromptField(row["field"]) for row in cursor.fetchall()]


class SqliteArtifactProvenance(ArtifactProvenancePort):
    """SQLite-backed implementation of ArtifactProvenancePort."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        try:
            yield conn
        finally:
            conn.close()

    def record_fingerprint(self, project_id: str, fingerprint: ArtifactFingerprint) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifact_provenance
                    (project_id, artifact_kind, scene_number, prompt_hash, workflow_hash, reference_hash, timeline_hash, dimensions_hash, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    fingerprint.artifact_kind.value,
                    fingerprint.scene_number,
                    fingerprint.prompt_hash,
                    fingerprint.workflow_hash,
                    fingerprint.reference_hash,
                    fingerprint.timeline_hash,
                    fingerprint.dimensions_hash,
                    datetime.datetime.now(_UTC).isoformat(),
                ),
            )
            conn.commit()

    def load_fingerprint(
        self, project_id: str, kind: ArtifactKind, scene_number: int | None = None
    ) -> ArtifactFingerprint | None:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM artifact_provenance
                WHERE project_id = ? AND artifact_kind = ?
                AND (scene_number = ? OR (scene_number IS NULL AND ? IS NULL))
                ORDER BY recorded_at DESC LIMIT 1
                """,
                (project_id, kind.value, scene_number, scene_number),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return ArtifactFingerprint(
                artifact_kind=ArtifactKind(row["artifact_kind"]),
                scene_number=row["scene_number"],
                prompt_hash=row["prompt_hash"],
                workflow_hash=row["workflow_hash"],
                reference_hash=row["reference_hash"],
                timeline_hash=row["timeline_hash"],
                dimensions_hash=row["dimensions_hash"],
            )

    def load_fingerprints(self, project_id: str) -> list[ArtifactFingerprint]:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM artifact_provenance
                WHERE project_id = ?
                ORDER BY recorded_at DESC
                """,
                (project_id,),
            )
            return [
                ArtifactFingerprint(
                    artifact_kind=ArtifactKind(row["artifact_kind"]),
                    scene_number=row["scene_number"],
                    prompt_hash=row["prompt_hash"],
                    workflow_hash=row["workflow_hash"],
                    reference_hash=row["reference_hash"],
                    timeline_hash=row["timeline_hash"],
                    dimensions_hash=row["dimensions_hash"],
                )
                for row in cursor.fetchall()
            ]
