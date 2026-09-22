"""Strict canonical StoryPlan contract (issue #1384).

This module defines the domain contract every later planning task consumes:
a versioned, self-consistent story plan that *references* (never carries)
authoritative audio data, plus the pure rules that keep the plan the single
source of truth for narrative direction.

Scope (per #1384):
- Strict, versioned StoryPlan models with stable IDs and reference checks.
- A reference-only audio segment binding (segment id + content fingerprint).
- Creative overrides that take effect only when the resulting plan validates.
- Singer-conflict resolution that never fabricates an audio binding.

Non-goals (per #1384):
- DSPy/LLM planning.
- Production pipeline wiring.
- Audio analysis, lyric text/timestamps, stage1 segment IDs/timing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from feverslop.domain.artifact_hash import is_sha256_hex
from feverslop.errors import FeverSlopDataError

#: Version of the story plan serialization contract.
STORY_PLAN_SCHEMA_VERSION = "story-plan/v1"

#: Schema versions this contract can load.
SUPPORTED_SCHEMA_VERSIONS = frozenset({STORY_PLAN_SCHEMA_VERSION})

#: Revision of the planner that produced the plan.
PLANNER_REVISION = "planner/v4"

#: Maximum length of any stable ID.
MAX_ID = 128

#: Keys a segment brief must never carry; authoritative audio data stays in
#: the stage1 segment artifact, referenced by fingerprint only.
_FORBIDDEN_AUDIO_DATA_KEYS = frozenset(
    {
        "timestamp",
        "timestamps",
        "start_seconds",
        "end_seconds",
        "lyrics",
        "lyric_text",
        "lyric_timestamps",
        "vocal_evidence",
        "vocal_classification",
    }
)


class StoryMode(str, Enum):
    """How the plan is consumed downstream."""

    music_video = "music_video"
    narrative_film = "narrative_film"


class StoryPhase(str, Enum):
    """Narrative position of a beat."""

    opening = "opening"
    development = "development"
    climax = "climax"
    resolution = "resolution"


class VocalPresentation(str, Enum):
    """How vocals are presented in a segment."""

    on_screen = "on_screen"
    offscreen = "offscreen"
    instrumental = "instrumental"


#: Phases that may only close the story (last beat).
TERMINAL_PHASES = frozenset({StoryPhase.resolution})


def _validate_id(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if len(normalized) > MAX_ID:
        raise ValueError(f"{field_name} exceeds {MAX_ID} characters")
    return normalized


def _validate_sha256(value: str, field_name: str) -> str:
    if not is_sha256_hex(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


class PlanProvenance(BaseModel):
    """Who produced the plan and which sources it drew from."""

    model_config = ConfigDict(extra="forbid", strict=True)

    producer: str = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)
    notes: str = ""


class AudioSegmentRef(BaseModel):
    """Reference-only binding to an authoritative stage1 segment.

    Carries the segment id and a content fingerprint; never timestamps,
    lyrics, or vocal evidence.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    segment_id: str
    fingerprint: str

    @field_validator("segment_id")
    @classmethod
    def _norm_segment_id(cls, value: str) -> str:
        return _validate_id(value, "segment_id")

    @field_validator("fingerprint")
    @classmethod
    def _check_fingerprint(cls, value: str) -> str:
        return _validate_sha256(value, "fingerprint")


class CanonicalCharacter(BaseModel):
    """A canonical character the plan reasons about."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    name: str = Field(min_length=1)
    description: str = ""
    is_singer: bool = False

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "character id")


class CanonicalLocation(BaseModel):
    """A canonical location the plan reasons about."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    name: str = Field(min_length=1)
    description: str = ""

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "location id")


class CanonicalProp(BaseModel):
    """A canonical prop the plan reasons about."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    name: str = Field(min_length=1)
    description: str = ""

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "prop id")


class StoryBeat(BaseModel):
    """One narrative beat in the story arc."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    phase: StoryPhase
    description: str = Field(min_length=1)
    character_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    prop_ids: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "beat id")


class CharacterArc(BaseModel):
    """How a character changes across beats."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    character_id: str
    beat_ids: list[str] = Field(default_factory=list)
    from_state: str = ""
    to_state: str = ""

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "arc id")


class BriefActorState(BaseModel):
    """One character's playable state for a planned segment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    character_id: str
    state: str = ""
    physical_state: str = ""

    @field_validator("character_id")
    @classmethod
    def _norm_character_id(cls, value: str) -> str:
        return _validate_id(value, "brief actor-state character id")


class SegmentBrief(BaseModel):
    """Creative direction for one stage1 segment (music video) or shot (film).

    ``target`` is the stage1 segment id for ``music_video`` mode or the shot
    id for ``narrative_film`` mode.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    target: str
    beat_id: str | None = None
    milestone_id: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    prop_ids: list[str] = Field(default_factory=list)
    vocal_presentation: VocalPresentation = VocalPresentation.offscreen
    visual_direction: str = ""
    objective: str = ""
    emotional_turn: str = ""
    actor_states: list[BriefActorState] = Field(default_factory=list)
    exclusive: bool = False
    audio_ref: AudioSegmentRef | None = None

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "brief id")

    @field_validator("target")
    @classmethod
    def _norm_target(cls, value: str) -> str:
        return _validate_id(value, "target")

    @model_validator(mode="before")
    @classmethod
    def _reject_authoritative_audio_data(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            carried = sorted(set(data) & _FORBIDDEN_AUDIO_DATA_KEYS)
            if carried:
                raise ValueError(f"brief must not carry authoritative audio data: {carried}")
        return data


class PlanDiagnostic(BaseModel):
    """A non-fatal problem reported by an override or planner rule."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    subject_id: str | None = None


class CreativeOverride(BaseModel):
    """A requested change to the plan, applied only if the result validates."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    source: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    field_path: str = Field(min_length=1)
    value: str

    @field_validator("id")
    @classmethod
    def _norm_id(cls, value: str) -> str:
        return _validate_id(value, "override id")


class StoryPlan(BaseModel):
    """The canonical, self-consistent story plan."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = STORY_PLAN_SCHEMA_VERSION
    planner_revision: str = PLANNER_REVISION
    mode: StoryMode
    source_fingerprint: str
    provenance: PlanProvenance
    characters: list[CanonicalCharacter] = Field(default_factory=list)
    locations: list[CanonicalLocation] = Field(default_factory=list)
    props: list[CanonicalProp] = Field(default_factory=list)
    beats: list[StoryBeat] = Field(default_factory=list)
    arcs: list[CharacterArc] = Field(default_factory=list)
    segments: list[SegmentBrief] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "StoryPlan":
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported story plan schema: {self.schema_version}")
        if not is_sha256_hex(self.source_fingerprint):
            raise ValueError("source_fingerprint must be a lowercase SHA-256 hex digest")
        character_ids = _unique_ids([c.id for c in self.characters], "character")
        location_ids = _unique_ids([location.id for location in self.locations], "location")
        prop_ids = _unique_ids([p.id for p in self.props], "prop")
        beat_ids = _unique_ids([b.id for b in self.beats], "beat")
        _unique_ids([a.id for a in self.arcs], "arc")
        _unique_ids([s.id for s in self.segments], "segment")
        for beat in self.beats:
            _refs_within(character_ids, beat.character_ids, f"beat {beat.id} characters")
            if beat.location_id is not None:
                _require(location_ids, beat.location_id, f"beat {beat.id} location")
            _refs_within(prop_ids, beat.prop_ids, f"beat {beat.id} props")
        for arc in self.arcs:
            _require(character_ids, arc.character_id, f"arc {arc.id} character")
            _refs_within(beat_ids, arc.beat_ids, f"arc {arc.id} beats")
        for brief in self.segments:
            if brief.beat_id is not None:
                _require(beat_ids, brief.beat_id, f"brief {brief.id} beat")
            _refs_within(character_ids, brief.character_ids, f"brief {brief.id} characters")
            if brief.location_id is not None:
                _require(location_ids, brief.location_id, f"brief {brief.id} location")
            _refs_within(prop_ids, brief.prop_ids, f"brief {brief.id} props")
            _refs_within(
                character_ids,
                [state.character_id for state in brief.actor_states],
                f"brief {brief.id} actor states",
            )
        _unique_ids([s.target for s in self.segments], "brief target")
        _check_beat_order(self.beats)
        _check_exclusive_allocations(self.segments)
        if self.mode is StoryMode.music_video:
            for brief in self.segments:
                if brief.audio_ref is None:
                    raise ValueError(f"music video brief {brief.id} requires an audio_ref")
                if brief.audio_ref.segment_id != brief.target:
                    raise ValueError(
                        f"music video brief {brief.id} audio_ref.segment_id must match target"
                    )
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StoryPlan":
        return cls.model_validate(payload, strict=True)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str) -> "StoryPlan":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FeverSlopDataError(f"story plan is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise FeverSlopDataError("story plan must be a JSON object")
        try:
            return cls.model_validate_json(text, strict=True)
        except ValidationError as exc:
            raise FeverSlopDataError("story plan does not match the strict contract") from exc


@dataclass(frozen=True, slots=True)
class OverrideResult:
    """Outcome of applying one creative override to a plan."""

    plan: StoryPlan
    applied: bool
    diagnostics: tuple[PlanDiagnostic, ...]


def resolve_singer_conflict(
    direction: VocalPresentation,
    *,
    has_audio_binding: bool,
) -> VocalPresentation:
    """Resolve a singer presentation conflict without fabricating a binding.

    An ``on_screen`` direction with no audio binding degrades to
    ``offscreen``; every other direction passes through unchanged.
    """
    if direction is VocalPresentation.on_screen and not has_audio_binding:
        return VocalPresentation.offscreen
    return direction


_OVERRIDE_COLLECTIONS: dict[str, type[BaseModel]] = {
    "characters": CanonicalCharacter,
    "locations": CanonicalLocation,
    "props": CanonicalProp,
    "beats": StoryBeat,
    "arcs": CharacterArc,
    "segments": SegmentBrief,
}


def apply_creative_override(plan: StoryPlan, override: CreativeOverride) -> OverrideResult:
    """Apply one creative override; it takes effect only if the result validates.

    ``field_path`` is either ``<plan-field>`` or ``<collection>.<item_id>.<field>``
    with collection in characters/locations/props/beats/arcs/segments. The
    override value is coerced to the target field's type; an unknown target or
    uncoercible value yields a diagnostic, never an exception (apply is total).
    """
    parts = override.field_path.split(".")
    if len(parts) == 1:
        field_name = parts[0]
        if field_name not in StoryPlan.model_fields:
            return _rejected(plan, override, "override_unknown_target", f"unknown plan field: {field_name}")
        annotation = StoryPlan.model_fields[field_name].annotation
    elif len(parts) == 3:
        collection_name, item_id, field_name = parts
        model_cls = _OVERRIDE_COLLECTIONS.get(collection_name)
        if model_cls is None:
            return _rejected(plan, override, "override_unknown_target", f"unknown collection: {collection_name}")
        if not any(entry.id == item_id for entry in getattr(plan, collection_name)):
            return _rejected(
                plan, override, "override_unknown_target", f"unknown {collection_name} item: {item_id}"
            )
        if field_name not in model_cls.model_fields:
            return _rejected(
                plan, override, "override_unknown_target", f"unknown {collection_name} field: {field_name}"
            )
        annotation = model_cls.model_fields[field_name].annotation
    else:
        return _rejected(
            plan, override, "override_unknown_target", f"invalid field path: {override.field_path}"
        )

    try:
        value = _coerce_value(override.value, annotation)
    except ValueError as exc:
        return _rejected(plan, override, "override_rejected", str(exc))

    candidate = plan.model_dump()
    if len(parts) == 1:
        candidate[field_name] = value
    else:
        collection_name = parts[0]
        for index, entry in enumerate(candidate[collection_name]):
            if entry["id"] == parts[1]:
                candidate[collection_name][index][parts[2]] = value
                break
    try:
        updated = StoryPlan.model_validate(candidate, strict=False)
    except ValidationError:
        return _rejected(plan, override, "override_rejected", "override produced an invalid plan")
    return OverrideResult(plan=updated, applied=True, diagnostics=())


def _coerce_value(value: str, annotation: Any) -> Any:
    """Coerce an override value to the target annotation.

    Raises ``ValueError`` for uncoercible values or unsupported annotations.
    """
    if annotation is str:
        return value
    args = get_args(annotation)
    if args and type(None) in args:
        if value in ("", "null", "none"):
            return None
        concrete = [a for a in args if a is not type(None)]
        if len(concrete) != 1:
            raise ValueError(f"unsupported annotation: {annotation!r}")
        return _coerce_value(value, concrete[0])
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as exc:
            raise ValueError(f"no {annotation.__name__} member named {value!r}") from exc
    if annotation is bool:
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot coerce {value!r} to bool")
    if annotation is int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"cannot coerce {value!r} to int") from exc
    if annotation is float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"cannot coerce {value!r} to float") from exc
    raise ValueError(f"unsupported annotation: {annotation!r}")


def _rejected(
    plan: StoryPlan, override: CreativeOverride, code: str, message: str
) -> OverrideResult:
    return OverrideResult(
        plan=plan,
        applied=False,
        diagnostics=(PlanDiagnostic(code=code, message=message, subject_id=override.id),),
    )


def _check_beat_order(beats: list[StoryBeat]) -> None:
    for index, beat in enumerate(beats):
        if index < len(beats) - 1 and beat.phase in TERMINAL_PHASES:
            raise ValueError(f"beat {beat.id} ends the story before the last beat")


def _check_exclusive_allocations(segments: list[SegmentBrief]) -> None:
    seen: set[str] = set()
    for brief in segments:
        if brief.exclusive and brief.beat_id is not None:
            if brief.beat_id in seen:
                raise ValueError(f"beat {brief.beat_id} has overlapping exclusive allocations")
            seen.add(brief.beat_id)


def _unique_ids(ids: list[str], kind: str) -> set[str]:
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise ValueError(f"duplicate {kind} id: {value}")
        seen.add(value)
    return seen


def _require(known: set[str], value: str, label: str) -> None:
    if value not in known:
        raise ValueError(f"{label} references unknown id: {value}")


def _refs_within(known: set[str], values: list[str], label: str) -> None:
    for value in values:
        _require(known, value, label)
