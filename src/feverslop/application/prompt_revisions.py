from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass

from feverslop.domain.prompt_revisions import (
    DuplicateRevisionError,
    PromptField,
    PromptHistory,
    PromptRevision,
    restore_revision,
)
from feverslop.ports.revision_store import RevisionStorePort

Clock = Callable[[], datetime.datetime]


@dataclass(frozen=True)
class HistoryLoadResult:
    history: PromptHistory
    available_fields: list[PromptField]


class PatchPromptError(ValueError):
    pass


class LoadPromptHistoryUseCase:
    def __init__(self, *, store: RevisionStorePort) -> None:
        self._store = store

    def execute(
        self,
        *,
        project_id: str,
        scene_number: int,
        field: PromptField,
    ) -> HistoryLoadResult:
        history = self._store.load_history(project_id, scene_number, field)
        available_fields = self._store.list_fields(project_id, scene_number)
        return HistoryLoadResult(history=history, available_fields=available_fields)


class RestoreRevisionUseCase:
    def __init__(
        self,
        *,
        store: RevisionStorePort,
        clock: Clock | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or _default_clock

    def execute(
        self,
        *,
        project_id: str,
        scene_number: int,
        field: PromptField,
        revision_id: str,
    ) -> PromptRevision:
        history = self._store.load_history(project_id, scene_number, field)
        now = self._clock()
        restored = restore_revision(history, revision_id=revision_id, now=now)
        try:
            self._store.save_revision(restored)
        except DuplicateRevisionError:
            raise PatchPromptError(f"Restored revision {restored.id!r} already exists") from None
        return restored


def _default_clock() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)
