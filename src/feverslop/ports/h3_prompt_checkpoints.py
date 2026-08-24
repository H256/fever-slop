from __future__ import annotations

from typing import Any, Protocol

from feverslop.domain.h3_prompt_checkpoint import (
    H3PromptCheckpoint,
    H3PromptCheckpointInput,
)


class H3PromptCheckpointPort(Protocol):
    def load(self, request: H3PromptCheckpointInput) -> H3PromptCheckpoint | None: ...

    def save(
        self,
        request: H3PromptCheckpointInput,
        generated: dict[str, Any],
    ) -> H3PromptCheckpoint: ...
