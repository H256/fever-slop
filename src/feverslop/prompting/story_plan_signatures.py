"""Typed DSPy signatures for the small story-plan jobs (issue #1385).

The story plan is produced by divide-and-conquer: four narrow jobs (bible,
beat allocation and acting) instead of one model writing the full
screenplay plus concepts. Every output model uses ``extra="forbid"`` so the
LLM cannot smuggle in timestamps, lyrics, render settings, frame counts, or
audio bindings as extra fields; the service's deterministic validator is the
backstop for what slips past the typed contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StoryBibleResult(BaseModel):
    """Source-backed narrative facts keyed by supplied canonical ids only."""

    model_config = ConfigDict(extra="forbid")

    premise: str = ""
    theme: str = ""
    character_notes: dict[str, str] = Field(default_factory=dict)
    location_notes: dict[str, str] = Field(default_factory=dict)
    prop_notes: dict[str, str] = Field(default_factory=dict)
    narrative_facts: list[str] = Field(default_factory=list)


class BeatAllocationDraft(BaseModel):
    """One narrative beat in allocation (narrative) order."""

    model_config = ConfigDict(extra="forbid")

    phase: str
    description: str
    milestone_id: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    prop_ids: list[str] = Field(default_factory=list)


class BeatAllocationResult(BaseModel):
    """A compact model-authored beat sheet in narrative order."""

    model_config = ConfigDict(extra="forbid")

    beats: list[BeatAllocationDraft] = Field(min_length=1)


class ActorStateRef(BaseModel):
    """A canonical character id plus a short state label (not prose)."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    state: str


class SegmentBriefDraft(BaseModel):
    """Structured acting direction for one supplied segment/shot.

    No id, no audio_ref, no timestamp/lyric/render fields: those are
    service-assigned or forbidden by the plan contract.
    """

    model_config = ConfigDict(extra="forbid")

    target: str
    beat_id: str | None = None
    milestone_id: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    prop_ids: list[str] = Field(default_factory=list)
    vocal_presentation: str
    visual_direction: str = ""
    objective: str = ""
    emotional_turn: str = ""
    actor_states: list[ActorStateRef] = Field(default_factory=list)


class CharacterArcDraft(BaseModel):
    """How one canonical character changes across service-assigned beats."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    beat_ids: list[str] = Field(default_factory=list)
    from_state: str = ""
    to_state: str = ""


class ActingResult(BaseModel):
    """Structured per-brief acting plus character arcs; not free prose."""

    model_config = ConfigDict(extra="forbid")

    arcs: list[CharacterArcDraft] = Field(default_factory=list)
    briefs: list[SegmentBriefDraft] = Field(default_factory=list)


def build_story_plan_signature_bundle(dspy_module: Any | None = None) -> dict[str, Any]:
    if dspy_module is None:
        import dspy as dspy_module

    class StoryPlanBible(dspy_module.Signature):
        """Extract a source-backed typed story bible.

        Reference only the supplied canonical ids; never invent ids,
        timestamps, lyrics, render settings, or audio bindings. The user
        direction is the highest-priority input.
        """

        guide: str = dspy_module.InputField()
        story_text: str = dspy_module.InputField()
        creative_direction: str = dspy_module.InputField(
            desc="Explicit user direction; highest-priority input.",
        )
        characters: list[dict[str, Any]] = dspy_module.InputField()
        locations: list[dict[str, Any]] = dspy_module.InputField()
        props: list[dict[str, Any]] = dspy_module.InputField()
        bible: dict[str, Any] = dspy_module.OutputField()

    class BeatAllocation(dspy_module.Signature):
        """Write a small story arc in narrative order.

        The LAST beat reserves the terminal window (phase ``resolution``);
        Python assigns every segment to this beat sheet deterministically.
        Do not emit per-segment allocations, timestamps, or frame counts.
        """

        guide: str = dspy_module.InputField()
        creative_direction: str = dspy_module.InputField(
            desc="Explicit user direction; highest-priority input.",
        )
        bible: dict[str, Any] = dspy_module.InputField()
        characters: list[dict[str, Any]] = dspy_module.InputField()
        locations: list[dict[str, Any]] = dspy_module.InputField()
        props: list[dict[str, Any]] = dspy_module.InputField()
        allocation: dict[str, Any] = dspy_module.OutputField()

    class Acting(dspy_module.Signature):
        """Write structured per-brief acting direction.

        Per-brief ``objective``, ``emotional_turn``, and ``actor_states``;
        structured, not free prose; reference only supplied beat ids and
        canonical ids. The user direction is the highest-priority input.
        """

        guide: str = dspy_module.InputField()
        creative_direction: str = dspy_module.InputField(
            desc="Explicit user direction; highest-priority input.",
        )
        bible: dict[str, Any] = dspy_module.InputField()
        beats: list[dict[str, Any]] = dspy_module.InputField(
            desc="Service-assigned beat ids, phase, description, and per-brief allocation.",
        )
        expected_brief_ids: list[str] = dspy_module.InputField(
            desc="Exact service-assigned brief ids to return, with no omissions or extras.",
        )
        segments: list[dict[str, Any]] = dspy_module.InputField(
            desc="Compact segment/shot descriptors: id, fingerprint, duration only.",
        )
        characters: list[dict[str, Any]] = dspy_module.InputField()
        locations: list[dict[str, Any]] = dspy_module.InputField()
        props: list[dict[str, Any]] = dspy_module.InputField()
        result: dict[str, Any] = dspy_module.OutputField()

    return {
        "bible": StoryPlanBible,
        "beat_allocation": BeatAllocation,
        "acting": Acting,
    }
