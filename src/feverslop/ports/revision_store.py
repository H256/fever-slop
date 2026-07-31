from __future__ import annotations

from typing import Protocol

from feverslop.domain.prompt_revisions import PromptField, PromptHistory, PromptRevision


class RevisionStorePort(Protocol):
    """Port for persisting and loading prompt revision history."""

    def save_revision(self, revision: PromptRevision) -> None: ...

    def load_history(self, project_id: str, scene_number: int, field: PromptField) -> PromptHistory: ...

    def list_fields(self, project_id: str, scene_number: int) -> list[PromptField]: ...
