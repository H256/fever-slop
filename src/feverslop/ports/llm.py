from __future__ import annotations

from pathlib import Path
from typing import Protocol


class LLMPort(Protocol):
    def complete_prompt(self, system_prompt: str, prompt: str, timeout: float | None = None) -> str:
        """Return a completion for a system prompt and user prompt."""


class VisionLLMPort(Protocol):
    def complete_prompt_with_images(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: list[Path],
        timeout: float | None = None,
    ) -> str:
        """Return a completion grounded in the supplied images."""


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
