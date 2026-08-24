from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from feverslop.domain.canonical_plan_migration import (
    MigrationDocument,
    MigrationInput,
    MigrationReport,
)
from feverslop.domain.canonical_plan_regeneration import CanonicalPlanSnapshot
from feverslop.errors import FeverSlopDataError
from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.utils.io import atomic_write_bytes, atomic_write_json


@dataclass(frozen=True)
class MigrationApplyResult:
    applied: bool
    imported_count: int
    backup_dir: Path | None


class CanonicalPlanStore:
    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.layout = SceneArtifactLayout(self.project_dir)

    def load(self) -> MigrationInput:
        documents = []
        for path in self._candidate_paths():
            relative = path.relative_to(self.project_dir).as_posix()
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            try:
                value = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                documents.append(MigrationDocument(relative, digest, error="malformed JSON"))
            else:
                documents.append(MigrationDocument(relative, digest, value=value))
        return MigrationInput(str(self.project_dir), tuple(documents))

    def capture_regeneration(self) -> CanonicalPlanSnapshot:
        path = self.layout.base_plan
        artifact_id = str(path.resolve())
        if not path.is_file():
            return CanonicalPlanSnapshot(artifact_id, False, None, ())
        raw = path.read_bytes()
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FeverSlopDataError(f"Cannot capture canonical base plan: {path}: {exc}") from exc
        if not isinstance(value, list) or any(not isinstance(scene, dict) for scene in value):
            raise FeverSlopDataError(f"Canonical base plan must be a list of objects: {path}")
        return CanonicalPlanSnapshot(
            artifact_id,
            True,
            hashlib.sha256(raw).hexdigest(),
            tuple(value),
        )

    def commit_regeneration(
        self,
        snapshot: CanonicalPlanSnapshot,
        scenes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> Path:
        path = self.layout.base_plan
        if snapshot.artifact_id != str(path.resolve()):
            raise FeverSlopDataError("Canonical regeneration snapshot belongs to another artifact")
        if not snapshot.exists:
            if path.exists():
                raise FeverSlopDataError(
                    "Canonical base plan appeared during regeneration; refusing to overwrite it",
                )
        else:
            if not path.is_file():
                raise FeverSlopDataError(
                    "Canonical base plan changed during regeneration: source disappeared",
                )
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if current_hash != snapshot.sha256:
                raise FeverSlopDataError(
                    "Canonical base plan changed during regeneration; refusing stale write",
                )
        return atomic_write_json(path, list(scenes))

    def apply(
        self,
        report: MigrationReport,
        *,
        run_id: str | None = None,
    ) -> MigrationApplyResult:
        if report.project_id != str(self.project_dir):
            raise FeverSlopDataError("Migration report belongs to a different project")
        if report.unresolved:
            raise FeverSlopDataError(
                f"Cannot apply migration with {len(report.unresolved)} unresolved finding(s)",
            )
        if not report.importable:
            return MigrationApplyResult(False, 0, None)

        source_bytes = self._verified_source_bytes(report)
        backup_dir = self.layout.legacy_migration_dir / self._run_id(run_id)
        if backup_dir.exists() and any(backup_dir.iterdir()):
            raise FeverSlopDataError(f"Migration backup already exists: {backup_dir}")

        for relative, raw in source_bytes.items():
            atomic_write_bytes(backup_dir / Path(relative), raw)
        report_bytes = (
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        atomic_write_bytes(backup_dir / "report.json", report_bytes)

        self._verified_source_bytes(report)
        updated = self._with_overrides(report)
        atomic_write_json(self.layout.base_plan, updated)
        return MigrationApplyResult(True, len(report.importable), backup_dir)

    def _verified_source_bytes(self, report: MigrationReport) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for source in report.sources:
            path = (self.project_dir / source.path).resolve()
            if not path.is_relative_to(self.project_dir) or not path.is_file():
                raise FeverSlopDataError(f"Migration source is missing or outside project: {source.path}")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source.sha256:
                raise FeverSlopDataError(
                    f"Migration source changed since analysis: {source.path}",
                )
            result[source.path] = raw
        return result

    def _with_overrides(self, report: MigrationReport) -> list[dict[str, Any]]:
        updated = deepcopy(report.base_plan)
        by_scene_id = {
            scene["canonical"]["scene_id"]: scene
            for scene in updated
        }
        source_hashes = {source.path: source.sha256 for source in report.sources}
        for finding in report.importable:
            if finding.scene_id is None or finding.role is None:
                raise FeverSlopDataError("Importable finding has no canonical scene and role")
            role = by_scene_id[finding.scene_id]["canonical"]["roles"].setdefault(
                finding.role,
                {},
            )
            role["override"] = {
                "value": deepcopy(finding.value),
                "provenance": {
                    "source": "legacy-plan-migration",
                    "source_path": finding.source_path,
                    "source_sha256": source_hashes[finding.source_path],
                    "field_path": finding.field_path,
                },
            }
        return updated

    def _candidate_paths(self) -> list[Path]:
        paths = [
            self.layout.base_plan,
            self.layout.compact_plan,
            self.layout.anchored_plan,
            self.layout.references_plan,
            self.layout.ingredients_plan,
        ]
        paths.extend(sorted(self.layout.render_dir.glob("render_plan_*.json")))
        return [
            path
            for index, path in enumerate(paths)
            if path.is_file() and path not in paths[:index]
        ]

    @staticmethod
    def _run_id(run_id: str | None) -> str:
        value = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise FeverSlopDataError("Migration run ID contains unsafe characters")
        return value
