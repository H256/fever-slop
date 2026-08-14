from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MusicVideoSubjectLocations(BaseModel):
    subject: str
    actors: list[dict[str, Any]] = Field(default_factory=list)
    locations: list[dict[str, Any]] = Field(default_factory=list)


def build_music_video_signature_bundle(dspy_module: Any | None = None):
    if dspy_module is None:
        import dspy as dspy_module

    class StoryIdea(dspy_module.Signature):
        """Create a concise music-video story idea using the supplied guide."""
        guide: str = dspy_module.InputField()
        lyrics: str = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        story_idea: str = dspy_module.OutputField()

    class StyleBlock(dspy_module.Signature):
        """Create the requested three-part visual style block using the guide."""
        guide: str = dspy_module.InputField()
        lyrics: str = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        style_block: str = dspy_module.OutputField()

    class SubjectLocations(dspy_module.Signature):
        """Extract stable actors and physical locations as structured data."""
        guide: str = dspy_module.InputField()
        story_idea: str = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        result: MusicVideoSubjectLocations = dspy_module.OutputField()

    class ConceptMap(dspy_module.Signature):
        """Map every supplied timed segment to one visual concept."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        concepts: dict[str, Any] = dspy_module.OutputField()

    class Detail(dspy_module.Signature):
        """Create one short visual detail for the requested category."""
        guide: str = dspy_module.InputField()
        label: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        detail: str = dspy_module.OutputField()

    class T2I(dspy_module.Signature):
        """Create one concrete still-image prompt using the supplied guide."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        prompt: str = dspy_module.OutputField()

    class I2V(dspy_module.Signature):
        """Create one dynamic image-to-video prompt using the supplied guide."""
        guide: str = dspy_module.InputField()
        performance_policy: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        prompt: str = dspy_module.OutputField()

    class Summary(dspy_module.Signature):
        """Summarize visual story continuity in a few concise sentences."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        summary: str = dspy_module.OutputField()

    class RepairConcepts(dspy_module.Signature):
        """Repair only missing segment concept keys."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        concepts: dict[str, Any] = dspy_module.OutputField()

    return {
        "story_idea": StoryIdea,
        "style_block": StyleBlock,
        "subject_locations": SubjectLocations,
        "concept_map": ConceptMap,
        "detail": Detail,
        "t2i": T2I,
        "i2v": I2V,
        "summary": Summary,
        "repair_concepts": RepairConcepts,
    }
