"""Per-project ComfyUI workflow import store.

Stores imported workflow snapshots under ``projects/<slug>/workflows/`` and
drives the draft → validated → tested → active state machine. Activation is
gated: a profile may only be activated when its validation AND its test-run
both succeeded.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from feverslop.domain.workflow_import import (
    AnalysisIssue,
    BoundaryMapping,
    ImportedProfile,
    NodeCandidate,
    TestRunResult,
    WorkflowAnalysis,
)
from feverslop.utils.io import atomic_write_bytes, atomic_write_json

#: Legal state transitions for an imported workflow profile.
#: draft/validated may only advance via record_validation/record_test_run;
#: activation is exclusively gated by :meth:`WorkflowImportStore.activate`.
STATE_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "draft": ("validated", "tested", "broken"),
    "validated": ("tested", "broken"),
    "tested": ("active", "broken"),
    "active": ("broken",),
    "broken": ("draft",),
}

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class WorkflowImportError(Exception):
    """Raised for invalid import-store operations."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_name(sha256: str) -> str:
    return f"{sha256[:16]}.json"


def _analysis_from_dict(data: dict[str, Any]) -> WorkflowAnalysis:
    mapping = data.get("mapping")
    return WorkflowAnalysis(
        compatibility=data["compatibility"],
        mapping=BoundaryMapping(**mapping) if mapping else None,
        core_candidates=tuple(NodeCandidate(**c) for c in data.get("core_candidates", ())),
        output_candidates=tuple(NodeCandidate(**c) for c in data.get("output_candidates", ())),
        issues=tuple(AnalysisIssue(**i) for i in data.get("issues", ())),
    )


class WorkflowImportStore:
    """Persists imported workflow snapshots and their lifecycle state."""

    def __init__(self, *, projects_root: Path):
        self.projects_root = projects_root

    def _profile_dir(self, project_id: str, profile_id: str) -> Path:
        return self.projects_root / project_id / "workflows" / profile_id

    # -- persistence -------------------------------------------------------

    def _write_snapshot(self, project_id: str, profile_id: str, payload: bytes) -> str:
        sha = _sha256(payload)
        directory = self._profile_dir(project_id, profile_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _snapshot_name(sha)
        # Self-heal: re-write if the file is missing or no longer hashes to its
        # content-addressed name (e.g. a previously interrupted non-atomic write
        # left a truncated payload). The atomic write then restores it verifiably.
        if not path.exists() or _sha256(path.read_bytes()) != sha:
            atomic_write_bytes(path, payload)
        return sha

    def _read_snapshot(self, project_id: str, profile_id: str, sha: str) -> bytes:
        path = self._profile_dir(project_id, profile_id) / _snapshot_name(sha)
        if not path.exists():
            raise WorkflowImportError(f"workflow snapshot {sha[:16]} is missing")
        return path.read_bytes()

    def _read_profile(self, project_id: str, profile_id: str) -> dict[str, Any]:
        path = self._profile_dir(project_id, profile_id) / "profile.json"
        if not path.exists():
            raise WorkflowImportError(f"workflow profile {profile_id} is not imported")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_profile(self, project_id: str, profile_id: str, record: dict[str, Any]) -> None:
        path = self._profile_dir(project_id, profile_id) / "profile.json"
        atomic_write_json(path, record)

    # -- lifecycle ---------------------------------------------------------

    def import_workflow(
        self,
        *,
        project_id: str,
        profile_id: str,
        pipeline: str,
        purpose: str,
        graph: object,
        reset: bool = False,
    ) -> ImportedProfile:
        if not _PROFILE_ID.match(profile_id):
            raise WorkflowImportError("profile_id must be lowercase [a-z0-9._-]")
        if purpose not in {"preview", "final"}:
            raise WorkflowImportError("purpose must be preview or final")
        if isinstance(graph, bytes):
            payload = graph
        elif isinstance(graph, str):
            payload = graph.encode("utf-8")
        elif isinstance(graph, dict):
            payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        else:
            raise WorkflowImportError("graph must be JSON bytes, text, or a mapping")
        sha = self._write_snapshot(project_id, profile_id, payload)
        record = self._next_record(
            project_id=project_id, profile_id=profile_id, pipeline=pipeline, purpose=purpose,
            sha=sha, reset=reset,
        )
        self._write_profile(project_id, profile_id, record)
        return self.get_profile(project_id, profile_id)

    def _next_record(
        self,
        *,
        project_id: str,
        profile_id: str,
        pipeline: str,
        purpose: str,
        sha: str,
        reset: bool,
    ) -> dict[str, Any]:
        """Build the profile record for an import.

        A fresh import (or ``reset=True``) yields a blank draft. A same-sha
        re-import (a no-op or a snapshot self-heal) preserves the full
        lifecycle record so re-importing does not silently wipe recorded
        validation/test-run state. A changed-sha re-import resets the
        validation-dependent fields to draft: they are stale for the new graph.
        """
        fresh = {
            "profile_id": profile_id,
            "pipeline": pipeline,
            "purpose": purpose,
            "workflow_sha256": sha,
            "status": "draft",
            "mapping": None,
            "test_run": None,
            "analysis": None,
            "validation_valid": None,
        }
        if reset:
            return fresh
        try:
            existing = self._read_profile(project_id, profile_id)
        except WorkflowImportError:
            return fresh
        if existing["workflow_sha256"] == sha:
            return existing
        return fresh

    def get_profile(self, project_id: str, profile_id: str) -> ImportedProfile:
        record = self._read_profile(project_id, profile_id)
        mapping = record.get("mapping")
        analysis = record.get("analysis")
        return ImportedProfile(
            profile_id=record["profile_id"],
            pipeline=record["pipeline"],
            purpose=record["purpose"],
            workflow_sha256=record["workflow_sha256"],
            mapping=BoundaryMapping(**mapping) if mapping else None,
            status=record["status"],
            test_run=TestRunResult(**record["test_run"]) if record.get("test_run") else None,
            analysis=_analysis_from_dict(analysis) if analysis else None,
        )

    def list_profiles(self, project_id: str) -> list[ImportedProfile]:
        base = self.projects_root / project_id / "workflows"
        if not base.is_dir():
            return []
        return [
            self.get_profile(project_id, child.name)
            for child in sorted(base.iterdir())
            if (child / "profile.json").exists()
        ]

    def find_active(self, project_id: str, *, pipeline: str, purpose: str) -> ImportedProfile | None:
        """Return the first active profile matching pipeline/purpose, or None."""
        for profile in self.list_profiles(project_id):
            if profile.status == "active" and profile.pipeline == pipeline and profile.purpose == purpose:
                return profile
        return None

    def set_state(self, project_id: str, profile_id: str, status: str) -> None:
        record = self._read_profile(project_id, profile_id)
        current = record["status"]
        if status not in STATE_TRANSITIONS.get(current, ()):
            raise WorkflowImportError(f"illegal transition {current} -> {status}")
        record["status"] = status
        self._write_profile(project_id, profile_id, record)

    def record_validation(
        self,
        project_id: str,
        profile_id: str,
        *,
        valid: bool,
        analysis: WorkflowAnalysis | None = None,
    ) -> None:
        record = self._read_profile(project_id, profile_id)
        record["validation_valid"] = valid
        record["analysis"] = analysis.to_dict() if analysis else None
        if analysis is not None and analysis.mapping is not None:
            record["mapping"] = analysis.mapping.to_dict()
        if valid and record["status"] == "draft":
            record["status"] = "validated"
        elif not valid:
            record["status"] = "broken"
        self._write_profile(project_id, profile_id, record)

    def record_test_run(
        self, project_id: str, profile_id: str, result: TestRunResult
    ) -> None:
        record = self._read_profile(project_id, profile_id)
        record["test_run"] = result.to_dict()
        if result.success and record["status"] in {"draft", "validated"}:
            record["status"] = "tested"
        elif not result.success:
            record["status"] = "broken"
        self._write_profile(project_id, profile_id, record)

    def activate(self, project_id: str, profile_id: str) -> None:
        """Activate only when validation AND test-run both succeeded."""
        record = self._read_profile(project_id, profile_id)
        if not record.get("validation_valid"):
            raise WorkflowImportError("profile failed validation; test-run gate not met")
        test_run = record.get("test_run")
        if not test_run or not test_run.get("success"):
            raise WorkflowImportError("profile has no successful test-run")
        if record["status"] not in {"validated", "tested"}:
            raise WorkflowImportError(f"profile is {record['status']}; expected validated or tested")
        record["status"] = "active"
        self._write_profile(project_id, profile_id, record)

    def deactivate(self, project_id: str, profile_id: str) -> None:
        record = self._read_profile(project_id, profile_id)
        if record["status"] != "active":
            raise WorkflowImportError("profile is not active")
        record["status"] = "tested"
        self._write_profile(project_id, profile_id, record)

    # -- snapshot immutability ---------------------------------------------

    def verify_snapshot(self, project_id: str, profile_id: str) -> bool:
        """True when the stored snapshot still hashes to the recorded sha."""
        record = self._read_profile(project_id, profile_id)
        payload = self._read_snapshot(project_id, profile_id, record["workflow_sha256"])
        return _sha256(payload) == record["workflow_sha256"]

    def snapshot_path(self, project_id: str, profile_id: str) -> Path:
        """Absolute path of the stored snapshot file for a profile."""
        record = self._read_profile(project_id, profile_id)
        return self._profile_dir(project_id, profile_id) / _snapshot_name(record["workflow_sha256"])

    def snapshot_bytes(self, project_id: str, profile_id: str) -> bytes:
        """Read back the stored snapshot payload for a profile."""
        record = self._read_profile(project_id, profile_id)
        return self._read_snapshot(project_id, profile_id, record["workflow_sha256"])

    def for_project(self, project_id: str) -> "ProjectWorkflowImportStore":
        """Return a project-scoped view for ergonomic per-project access."""
        return ProjectWorkflowImportStore(self, project_id)


class ProjectWorkflowImportStore:
    """Project-scoped view over :class:`WorkflowImportStore`."""

    def __init__(self, store: WorkflowImportStore, project_id: str):
        self._store = store
        self.project_id = project_id

    def find_active(self, *, pipeline: str, purpose: str) -> ImportedProfile | None:
        return self._store.find_active(self.project_id, pipeline=pipeline, purpose=purpose)

    def snapshot_path(self, profile_id: str) -> Path:
        return self._store.snapshot_path(self.project_id, profile_id)
