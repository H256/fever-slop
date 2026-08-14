from typing import Any

from pydantic import BaseModel, Field


class MSRReferenceDescription(BaseModel):
    id: str
    type: str
    description: str


class MSRRelayPrompt(BaseModel):
    index: int
    prompt: str


class MSRPromptResult(BaseModel):
    references: list[MSRReferenceDescription] = Field(default_factory=list)
    relays: list[MSRRelayPrompt] = Field(default_factory=list)


def build_msr_signature_bundle(dspy_module: Any | None = None):
    if dspy_module is None:
        import dspy as dspy_module

    class Vision(dspy_module.Signature):
        """Describe MSR references and local relay actions from supplied images."""

        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        images: list[dspy_module.Image] = dspy_module.InputField()
        result: MSRPromptResult = dspy_module.OutputField()

    class Segments(dspy_module.Signature):
        """Write one validated local MSR direction for every relay segment."""

        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: MSRPromptResult = dspy_module.OutputField()

    return {"vision": Vision, "segments": Segments}
