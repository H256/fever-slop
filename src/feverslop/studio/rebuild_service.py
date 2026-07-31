from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from feverslop.ports.revision_store import RevisionStorePort


@dataclass(frozen=True)
class RevisionSaveResult:
    revision: PromptRevision
    changed: bool


class PromptSaveConflict(Exception):
    """Raised when the prompt has not changed or the value is blank."""


class RebuildService:
    """Orchestrates prompt revision recording and retrieval for the studio UI."""

    def __init__(self, store: RevisionStorePort) -> None:
        self._patch = PatchPromptUseCase(store=store)
        self._load = LoadPromptHistoryUseCase(store=store)
        self._restore = RestoreRevisionUseCase(store=store)

    def save_prompt(
        self,
        *,
        project_id: str,
        scene_number: int,
        field: PromptField,
        value: str,
    ) -> RevisionSaveResult:
        """Record a new prompt revision. Raises PromptSaveConflict on conflict."""
        if not value or not value.strip():
            raise PromptSaveConflict("Prompt value must not be blank")

        # Pre-check unchanged from store, since execute() raises PatchPromptError
        history = self._load.execute(
            project_id=project_id,
            scene_number=scene_number,
            field=field,
        ).history
        if history.revisions and history.revisions[-1].value == value:
            raise PromptSaveConflict("Prompt value unchanged")

        try:
            revision = self._patch.execute(
                project_id=project_id,
                scene_number=scene_number,
                field=field,
                value=value,
            )
        except PatchPromptError as exc:
            raise PromptSaveConflict(str(exc)) from exc

        return RevisionSaveResult(revision=revision, changed=True)

    def get_history(
        self,
        *,
        project_id: str,
        scene_number: int,
        field: PromptField,
    ) -> HistoryLoadResult:
        """Load the prompt history for a scene and field."""
        return self._load.execute(
            project_id=project_id,
            scene_number=scene_number,
            field=field,
        )

    def restore_revision(
        self,
        *,
        project_id: str,
        scene_number: int,
        field: PromptField,
        revision_id: str,
    ) -> RevisionSaveResult:
        """Restore an old revision value as a new revision."""
        if not revision_id:
            raise ValueError("revision_id must not be empty")

        try:
            revision = self._restore.execute(
                project_id=project_id,
                scene_number=scene_number,
                field=field,
                revision_id=revision_id,
            )
        except (PatchPromptError, ValueError) as exc:
            raise PromptSaveConflict(str(exc)) from exc

        return RevisionSaveResult(revision=revision, changed=True)
