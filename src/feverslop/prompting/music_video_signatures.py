from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MusicVideoSubjectLocations(BaseModel):
    subject: str
    actors: list[dict[str, Any]] = Field(default_factory=list)
    locations: list[dict[str, Any]] = Field(default_factory=list)


class NarrativeMilestoneBinding(BaseModel):
    """One deterministic placement of a narrative milestone."""

    milestone_id: str
    location_id: str
    relative_position: float


class NarrativeMilestoneBindingsResult(BaseModel):
    bindings: list[NarrativeMilestoneBinding] = Field(default_factory=list)


class MusicVideoNarrativeContract(BaseModel):
    """Structured story arch the concept LLM is constrained by.

    ``location_order`` / ``milestone_order`` entries may be ``{id, source}``
    objects or plain strings; the batcher normalizes both. Every other field
    is optional and omitted (not invented) when the story does not warrant it.
    """

    location_order: list[Any] = Field(default_factory=list)
    milestone_order: list[Any] = Field(default_factory=list)
    milestone_bindings: list[NarrativeMilestoneBinding] = Field(default_factory=list)
    terminal_states: dict[str, Any] = Field(default_factory=dict)
    actor_allowed_locations: dict[str, Any] = Field(default_factory=dict)
    chronology_exceptions: dict[str, Any] = Field(default_factory=dict)
    one_shot_milestones: list[str] = Field(default_factory=list)


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
        # Declared so a configured cast_idea reaches the model instead of
        # being dropped as an out-of-signature field; empty when unset.
        cast_idea: str = dspy_module.InputField(
            desc="Explicit cast/design guidance from the project config; may be empty.",
        )
        result: MusicVideoSubjectLocations = dspy_module.OutputField()

    class NarrativeContract(dspy_module.Signature):
        """Derive the structured narrative contract (story arch) from the story idea.

        The supplied ``locations`` and ``actors`` are the canonical structured
        ids; the contract must reference only those ids (never invent new ones).
        """

        guide: str = dspy_module.InputField()
        story_idea: str = dspy_module.InputField()
        locations: list[dict[str, Any]] = dspy_module.InputField()
        actors: list[dict[str, Any]] = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        contract: MusicVideoNarrativeContract = dspy_module.OutputField()

    class NarrativeMilestoneBindings(dspy_module.Signature):
        """Place only missing narrative milestones in canonical locations."""

        guide: str = dspy_module.InputField()
        story_idea: str = dspy_module.InputField()
        location_order: list[Any] = dspy_module.InputField()
        milestone_order: list[Any] = dspy_module.InputField()
        missing_milestone_ids: list[str] = dspy_module.InputField()
        result: NarrativeMilestoneBindingsResult = dspy_module.OutputField()

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
        "narrative_contract": NarrativeContract,
        "narrative_milestone_bindings": NarrativeMilestoneBindings,
        "concept_map": ConceptMap,
        "detail": Detail,
        "t2i": T2I,
        "i2v": I2V,
        "summary": Summary,
        "repair_concepts": RepairConcepts,
    }
