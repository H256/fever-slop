from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from feverslop.prompting.dspy_h3_generator import GeneratedVideoPrompt
from feverslop.prompting.dspy_h3_models import VideoPromptRequest
from feverslop.prompting.model_types import ModelTypeSpec, resolve_model_type


class PromptGenerationService:
    """Validate prompt-generation input and delegate to a configured generator."""

    def __init__(self, generator: Callable[[dict[str, Any]], GeneratedVideoPrompt]):
        self._generator = generator

    def generate(
        self,
        model_type: str | ModelTypeSpec,
        description: str,
        *,
        references: Sequence[dict[str, Any]] | None = None,
        notes: str | None = None,
        duration_seconds: float | None = None,
        music_intent: str | None = None,
        strict_fidelity: bool = True,
    ) -> GeneratedVideoPrompt:
        spec = model_type if isinstance(model_type, ModelTypeSpec) else resolve_model_type(model_type)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be nonblank")

        request = VideoPromptRequest.model_validate(
            {
                "mode": spec.prompt_mode,
                "user_prompt": description.strip(),
                "references": list(references or []),
                "notes": notes,
                "duration_seconds": duration_seconds,
                "music_intent": music_intent,
                "strict_fidelity": strict_fidelity,
            }
        )
        return self._generator(request.model_dump(mode="json"))