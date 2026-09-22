from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from feverslop.domain.movie_continuity import (
    MovieContinuityCharacterState,
    MovieContinuityLedger,
    MovieContinuityLocationState,
    MovieContinuityPlan,
    MovieContinuityStyleBible,
    MovieNarrativeBeat,
    MovieSceneContinuityPacket,
)
from feverslop.domain.movie_references import (
    MovieActor,
    MovieBible,
    MovieContinuityRule,
    MovieLocation,
    StoryArch,
)
from feverslop.domain.story_plan import StoryMode, StoryPlan

__all__ = [
    "CinematicShot",
    "MovieAct",
    "MovieActor",
    "MovieBible",
    "MovieCharacterArc",
    "MovieContinuityCharacterState",
    "MovieContinuityLedger",
    "MovieContinuityLocationState",
    "MovieContinuityPlan",
    "MovieContinuityRule",
    "MovieContinuityStyleBible",
    "MovieLocation",
    "MovieNarrativeBeat",
    "MovieNarrativePlan",
    "MovieProject",
    "MovieSceneBlueprint",
    "MovieSceneCard",
    "MovieSceneContinuityPacket",
    "MovieScreenplayArtifact",
    "MovieScreenplayScene",
    "MovieSetupPayoff",
    "MovieShotCard",
    "MovieStoryDesign",
    "MovieTurningPoint",
    "Screenplay",
    "StoryArch",
    "story_plan_shot_durations",
    "story_plan_to_cinematic_shots",
    "story_plan_to_scene_cards",
    "story_plan_to_shot_cards",
    "story_plan_to_screenplay_scenes",
]


@dataclass(frozen=True)
class CinematicShot:
    shot_id: str
    description: str
    duration_seconds: float
    camera: str
    action: str
    expression: str
    location: str
    dialogue: str = ""
    actor_ids: tuple[str, ...] = ()
    location_id: str = ""
    continuity_notes: str = ""
    story_state_before: str = ""
    story_state_after: str = ""
    cause_from_previous: str = ""
    narrative_purpose: str = ""
    conflict_or_tension: str = ""
    turning_point: str = ""
    sets_up_next: str = ""
    transition_from_previous: str = "cut"


@dataclass(frozen=True)
class MovieScreenplayScene:
    scene_id: str
    heading: str
    summary: str
    action: str
    dialogue: str = ""
    actor_ids: tuple[str, ...] = ()
    location_id: str = ""
    source_span: str = ""
    dramatic_purpose: str = ""
    conflict: str = ""
    emotional_turn: str = ""
    subtext: str = ""
    dialogue_function: str = ""


@dataclass(frozen=True)
class MovieScreenplayArtifact:
    title: str
    source_type: str
    dialogue_language: str
    scenes: tuple[MovieScreenplayScene, ...]


@dataclass(frozen=True)
class MovieAct:
    act_id: str
    title: str
    purpose: str
    scene_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MovieTurningPoint:
    id: str
    scene_id: str
    description: str


@dataclass(frozen=True)
class MovieSetupPayoff:
    id: str
    setup_scene_id: str
    payoff_scene_id: str
    description: str


@dataclass(frozen=True)
class MovieCharacterArc:
    actor_id: str
    want: str = ""
    need: str = ""
    starting_state: str = ""
    ending_state: str = ""


@dataclass(frozen=True)
class MovieSceneBlueprint:
    scene_id: str
    purpose: str
    conflict: str
    emotional_turn: str
    subtext: str
    dialogue_function: str
    required_actors: tuple[str, ...] = ()
    location_id: str = ""
    expected_duration: float = 0.0


@dataclass(frozen=True)
class MovieStoryDesign:
    title: str
    premise: str
    theme: str
    act_structure: tuple[MovieAct, ...]
    turning_points: tuple[MovieTurningPoint, ...]
    setup_payoff_threads: tuple[MovieSetupPayoff, ...]
    character_arcs: tuple[MovieCharacterArc, ...]
    scene_blueprint: tuple[MovieSceneBlueprint, ...]


@dataclass(frozen=True)
class MovieNarrativePlan:
    title: str
    sequences: tuple[dict, ...]
    causal_chain: tuple[dict, ...]
    open_threads: tuple[str, ...] = ()


@dataclass(frozen=True)
class MovieSceneCard:
    scene_id: str
    shot_ids: tuple[str, ...]
    dramatic_purpose: str
    story_state_before: str
    story_state_after: str
    active_actor_ids: tuple[str, ...]
    location_id: str
    dialogue: str = ""


@dataclass(frozen=True)
class MovieShotCard:
    shot_id: str
    scene_id: str
    action: str
    camera: str
    acting: str
    dialogue: str = ""
    start_frame_brief: str = ""
    end_frame_brief: str = ""
    transition_from_previous: str = "cut"
    transition_reason: str = ""


@dataclass(frozen=True)
class Screenplay:
    text: str


@dataclass(frozen=True)
class MovieProject:
    slug: str
    name: str
    bible: MovieBible
    story_arch: StoryArch
    shots: tuple[CinematicShot, ...]
    duration_seconds: float
    width: int
    height: int
    mode: str
    config: dict | None = None

    def __post_init__(self) -> None:
        if not self.slug.strip():
            raise ValueError("MovieProject.slug must be non-blank")
        if not self.name.strip():
            raise ValueError("MovieProject.name must be non-blank")
        if not self.shots:
            raise ValueError("MovieProject.shots must not be empty")
        if not self.mode.strip():
            raise ValueError("MovieProject.mode must be non-blank")
        if self.config is not None:
            object.__setattr__(self, "config", types.MappingProxyType(self.config))


def _require_narrative_film(plan: StoryPlan) -> None:
    if plan.mode is not StoryMode.narrative_film:
        raise ValueError(f"expected a narrative_film story plan, got {plan.mode.value}")


def _render_plan_shot_entries(render_plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    shots = render_plan.get("shots") or []
    entries: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(shots, start=1):
        if not isinstance(entry, Mapping):
            continue
        shot_id = str(entry.get("shot_id") or f"shot_{index:04}").strip()
        if shot_id:
            entries[shot_id] = entry
    return entries


def _render_plan_duration(entry: Mapping[str, Any] | None) -> float:
    if entry is None:
        return 0.0
    return float(entry.get("duration_seconds") or entry.get("duration") or 0.0)


def story_plan_shot_durations(plan: StoryPlan, render_plan: Mapping[str, Any]) -> dict[str, float]:
    """Map the plan's shot IDs to durations from the render plan.

    A shot ID missing from the render plan maps to 0.0; the plan never
    fabricates timing.
    """
    _require_narrative_film(plan)
    entries = _render_plan_shot_entries(render_plan)
    return {brief.target: _render_plan_duration(entries.get(brief.target)) for brief in plan.segments}


def story_plan_to_cinematic_shots(plan: StoryPlan, render_plan: Mapping[str, Any]) -> tuple[CinematicShot, ...]:
    """Map the plan's segment briefs to cinematic shots.

    Shot IDs come from the plan; timing and render details come from the
    render plan (the existing source of truth).
    """
    _require_narrative_film(plan)
    entries = _render_plan_shot_entries(render_plan)
    shots: list[CinematicShot] = []
    for brief in plan.segments:
        entry = entries.get(brief.target)
        source: Mapping[str, Any] = entry if entry is not None else {}
        shots.append(
            CinematicShot(
                shot_id=brief.target,
                description=brief.visual_direction,
                duration_seconds=_render_plan_duration(entry),
                camera=str(source.get("camera") or ""),
                action=str(source.get("action") or ""),
                expression=str(source.get("acting") or source.get("expression") or ""),
                location=str(source.get("location") or ""),
                dialogue=str(source.get("dialogue") or ""),
                actor_ids=tuple(brief.character_ids),
                location_id=brief.location_id or "",
            )
        )
    return tuple(shots)


def story_plan_to_scene_cards(plan: StoryPlan) -> tuple[MovieSceneCard, ...]:
    """Map the plan's beats to scene cards.

    Each beat becomes one scene; segments referencing it via ``beat_id``
    become the scene's shot IDs in plan order.
    """
    _require_narrative_film(plan)
    shot_ids_by_beat: dict[str, list[str]] = {beat.id: [] for beat in plan.beats}
    for brief in plan.segments:
        if brief.beat_id is not None:
            shot_ids_by_beat[brief.beat_id].append(brief.target)
    return tuple(
        MovieSceneCard(
            scene_id=beat.id,
            shot_ids=tuple(shot_ids_by_beat[beat.id]),
            dramatic_purpose=beat.description,
            story_state_before="",
            story_state_after="",
            active_actor_ids=tuple(beat.character_ids),
            location_id=beat.location_id or "",
        )
        for beat in plan.beats
    )


def story_plan_to_shot_cards(plan: StoryPlan) -> tuple[MovieShotCard, ...]:
    """Map the plan's segment briefs to shot cards.

    The shot ID comes from the brief; the scene ID is the beat the brief
    references (empty when the brief is not bound to a beat).
    """
    _require_narrative_film(plan)
    return tuple(
        MovieShotCard(
            shot_id=brief.target,
            scene_id=brief.beat_id or "",
            action=brief.visual_direction,
            camera="",
            acting="",
        )
        for brief in plan.segments
    )


def story_plan_to_screenplay_scenes(plan: StoryPlan) -> tuple[MovieScreenplayScene, ...]:
    """Map the plan's beats to screenplay scenes.

    Each beat becomes one scene; the beat description carries the heading,
    summary, and dramatic purpose.
    """
    _require_narrative_film(plan)
    return tuple(
        MovieScreenplayScene(
            scene_id=beat.id,
            heading=beat.description,
            summary=beat.description,
            action=beat.description,
            actor_ids=tuple(beat.character_ids),
            location_id=beat.location_id or "",
            dramatic_purpose=beat.description,
        )
        for beat in plan.beats
    )
