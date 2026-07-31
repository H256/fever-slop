from __future__ import annotations

from typing import Protocol

from feverslop.domain.prompt_revisions import PromptField, PromptHistory, PromptRevision


class DuplicateRevisionError(ValueError):
    def __init__(self, revision_id: str) -> None:
        super().__init__(f"Revision {revision_id!r} already exists")
        self.revision_id = revision_id


class RevisionStorePort(Protocol):
    """Port for persisting and loading prompt revision history."""

    def save_revision(self, revision: PromptRevision) -> None: ...

    def load_history(self, scene_number: int, field: PromptField) -> PromptHistory: ...

    def list_fields(self, scene_number: int) -> list[PromptField]: ...
