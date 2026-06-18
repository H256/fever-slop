from __future__ import annotations

from typing import Protocol


class LLMPort(Protocol):
    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        """Return a completion for a system prompt and user prompt."""
