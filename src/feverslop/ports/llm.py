from __future__ import annotations

from typing import Protocol


class LLMPort(Protocol):
    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        """Return a completion for a system prompt and user prompt."""


class StoryboardPromptTransformerPort(Protocol):
    def transform_prompt(
        self,
        *,
        scene_number: int,
        original_prompt: str,
        width: int,
        height: int,
    ) -> str:
        """Return the prompt text to send to the storyboard workflow."""
