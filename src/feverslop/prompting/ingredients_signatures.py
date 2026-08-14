from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IngredientsReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    name: str = ""
    visual_description: str = ""
    image_prompt: str = ""
    t2i_description: str = ""


class IngredientsVisionPayload(BaseModel):
    references: list[IngredientsReferenceMetadata] = Field(default_factory=list)
    target_context: dict[str, Any] = Field(default_factory=dict)
    scene_sheet_description: str = ""


class IngredientsReferenceDescription(BaseModel):
    id: str
    type: str
    position: str = ""
    t2i_description: str


class IngredientsVisionResult(BaseModel):
    references: list[IngredientsReferenceDescription] = Field(default_factory=list)
    shot_invariants: str

    @field_validator("shot_invariants")
    @classmethod
    def validate_shot_invariants(cls, value: str) -> str:
        words = value.split()
        if not 60 <= len(words) <= 160:
            raise ValueError("shot_invariants must contain 60-160 words")
        return value


def build_ingredients_signature_bundle(dspy_module: Any | None = None):
    if dspy_module is None:
        import dspy as dspy_module

    class Vision(dspy_module.Signature):
        """Describe Ingredients references and stable shot invariants from supplied images."""

        guide: str = dspy_module.InputField()
        payload: IngredientsVisionPayload = dspy_module.InputField()
        images: list[dspy_module.Image] = dspy_module.InputField()
        result: IngredientsVisionResult = dspy_module.OutputField()

    return {"vision": Vision}
