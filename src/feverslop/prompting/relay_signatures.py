from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RelayDirection(BaseModel):
    index: int
    prompt: str


class RelayDirectionResult(BaseModel):
    directions: list[RelayDirection] = Field(default_factory=list)


def build_relay_signature_bundle(dspy_module: Any | None = None):
    if dspy_module is None:
        import dspy as dspy_module

    class CompactDirections(dspy_module.Signature):
        """Compact PromptRelay directions while preserving concrete actions and singing rules."""

        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        max_words: int = dspy_module.InputField()
        subject_anchor: str = dspy_module.InputField()
        result: RelayDirectionResult = dspy_module.OutputField()

    return CompactDirections
