from __future__ import annotations

import re
import types
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from feverslop.domain.movie import CinematicShot, MovieBible


@dataclass(frozen=True)
class MovieContinuityStyleBible:
    visual_style: str = ""
    palette: str = ""
    lighting: str = ""
    camera: str = ""
    negative_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class MovieContinuityCharacterState:
    character_id: str
    base_identity: str = ""
    wardrobe: str = ""
    carried_props: tuple[str, ...] = ()
    physical_state: str = ""
    emotional_state: str = ""
    last_location: str = ""
    last_action: str = ""


@dataclass(frozen=True)
class MovieContinuityLocationState:
    location_id: str
    name: str = ""
    time_of_day: str = ""
    lighting: str = ""
    props: tuple[str, ...] = ()
    environmental_state: str = ""


@dataclass(frozen=True)
class MovieSceneContinuityPacket:
    shot_id: str
    location_id: str = ""
    incoming: tuple[str, ...] = ()
    required_carryovers: tuple[str, ...] = ()
    allowed_changes: tuple[str, ...] = ()
    outgoing: tuple[str, ...] = ()
    characters: dict[str, MovieContinuityCharacterState] | None = None
    location: MovieContinuityLocationState | None = None

    def __post_init__(self) -> None:
        if self.characters is not None:
            object.__setattr__(self, "characters", types.MappingProxyType(self.characters))

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.shot_id,
                self.location_id,
                self.incoming,
                self.required_carryovers,
                self.allowed_changes,
                self.outgoing,
                dict(self.characters) if self.characters is not None else None,
                self.location,
            ),
        )


@dataclass(frozen=True)
class MovieNarrativeBeat:
    shot_id: str
    story_state_before: str = ""
    story_state_after: str = ""
    cause_from_previous: str = ""
    narrative_purpose: str = ""
    conflict_or_tension: str = ""
    turning_point: str = ""
    sets_up_next: str = ""


@dataclass(frozen=True)
class MovieContinuityLedger:
    style_bible: MovieContinuityStyleBible
    characters: dict[str, MovieContinuityCharacterState]
    locations: dict[str, MovieContinuityLocationState]
    scene_order: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "characters", types.MappingProxyType(self.characters))
        object.__setattr__(self, "locations", types.MappingProxyType(self.locations))

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.style_bible,
                dict(self.characters),
                dict(self.locations),
                self.scene_order,
            ),
        )


@dataclass(frozen=True)
class MovieContinuityPlan:
    continuity_ledger: MovieContinuityLedger
    scene_continuity: dict[str, MovieSceneContinuityPacket]
    narrative_chain: tuple[MovieNarrativeBeat, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_continuity", types.MappingProxyType(self.scene_continuity))

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.continuity_ledger,
                dict(self.scene_continuity),
                self.narrative_chain,
            ),
        )

    @classmethod
    def fallback(cls, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
        """Build an entire continuity plan from a style bible and shot list.

        Derives character states from bible actors, location states from
        bible locations, and constructs a narrative chain and scene
        continuity packets from the shot sequence.
        """
        style_bible = MovieContinuityStyleBible(
            visual_style="; ".join(bible.style_constraints),
            negative_constraints=("visible text", "unmotivated wardrobe changes", "unexplained prop changes"),
        )
        characters: dict[str, MovieContinuityCharacterState] = {
            actor.id: MovieContinuityCharacterState(
                character_id=actor.id,
                base_identity=actor.visual_description or actor.name,
                wardrobe=actor.visual_description or actor.name,
                emotional_state=actor.role,
            )
            for actor in bible.actors
        }
        locations: dict[str, MovieContinuityLocationState] = {
            location.id: MovieContinuityLocationState(
                location_id=location.id,
                name=location.name,
                environmental_state=location.visual_description,
            )
            for location in bible.locations
        }
        scene_continuity: dict[str, MovieSceneContinuityPacket] = {}
        narrative_chain: list[MovieNarrativeBeat] = []
        previous_outgoing: tuple[str, ...] = ()
        previous_after = bible.premise or bible.story_arch.premise
        for index, shot in enumerate(shots):
            shot_characters: dict[str, MovieContinuityCharacterState] = {
                actor_id: replace(
                    characters.get(actor_id)
                    or MovieContinuityCharacterState(character_id=actor_id, base_identity=actor_id.replace("_", " ").title()),
                    last_location=shot.location_id or shot.location,
                    last_action=shot.action or shot.description,
                    emotional_state=shot.expression,
                )
                for actor_id in shot.actor_ids
            }
            location = locations.get(shot.location_id) or MovieContinuityLocationState(
                location_id=shot.location_id or "primary_location",
                name=shot.location,
                environmental_state=shot.location,
            )
            carryovers = tuple(
                item
                for item in [
                    *_safe_continuity_facts(rule.description for rule in bible.continuity if rule.description),
                    *(f"{actor_id} identity and wardrobe remain consistent" for actor_id in shot.actor_ids),
                    f"location remains {location.name or shot.location}" if location.name or shot.location else "",
                ]
                if item
            )
            outgoing = tuple(item for item in [shot.action or shot.description, shot.dialogue, shot.expression] if item)
            scene_continuity[shot.shot_id] = MovieSceneContinuityPacket(
                shot_id=shot.shot_id,
                location_id=shot.location_id,
                incoming=previous_outgoing,
                required_carryovers=carryovers,
                allowed_changes=tuple(item for item in [shot.action, shot.camera] if item),
                outgoing=outgoing,
                characters=shot_characters,
                location=location,
            )
            next_shot = shots[index + 1] if index + 1 < len(shots) else None
            state_before = previous_after or f"Before {shot.shot_id}, the story is ready for the next beat."
            state_after = shot.action or shot.description or f"{shot.shot_id} completes its story beat."
            narrative_chain.append(
                MovieNarrativeBeat(
                    shot_id=shot.shot_id,
                    story_state_before=state_before,
                    story_state_after=state_after,
                    cause_from_previous="Opening beat establishes the premise." if index == 0 else f"Previous beat leaves: {'; '.join(previous_outgoing) or previous_after}",
                    narrative_purpose=shot.description,
                    conflict_or_tension=shot.expression or "story tension continues",
                    turning_point=shot.action or shot.description,
                    sets_up_next=(next_shot.description if next_shot else "Final beat resolves the current movie arc."),
                ),
            )
            previous_outgoing = outgoing
            previous_after = state_after
        return cls(
            continuity_ledger=MovieContinuityLedger(
                style_bible=style_bible,
                characters=characters,
                locations=locations,
                scene_order=tuple(shot.shot_id for shot in shots),
            ),
            scene_continuity=scene_continuity,
            narrative_chain=tuple(narrative_chain),
        )

    def normalize(self, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
        """Merge this partial plan with a fallback built from bible and shots.

        Missing ledger sections, scene packets, or narrative beats are filled
        from the fallback. Present values take priority.
        """
        fallback = self.fallback(bible, shots)
        shot_ids = [shot.shot_id for shot in shots]
        return MovieContinuityPlan(
            continuity_ledger=self.continuity_ledger or fallback.continuity_ledger,
            scene_continuity={
                shot_id: self.scene_continuity.get(shot_id) or fallback.scene_continuity[shot_id] for shot_id in shot_ids
            },
            narrative_chain=tuple(
                _narrative_for_shot(shot_id, self.narrative_chain, fallback.narrative_chain) for shot_id in shot_ids
            ),
        )


# ---------------------------------------------------------------------------
# Helper functions (used by MovieContinuityPlan.fallback and application layer)
# ---------------------------------------------------------------------------


def _safe_continuity_facts(value: Any) -> tuple[str, ...]:
    candidates = _split_continuity_text(value)
    facts: list[str] = []
    for candidate in candidates:
        fact = " ".join(str(candidate or "").split()).strip(" .")
        if not fact or _looks_like_continuity_noise(fact):
            continue
        if fact not in facts:
            facts.append(fact)
    return tuple(facts)


def _split_continuity_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;\n]+", value) if part.strip()]
    if hasattr(value, "__iter__"):
        parts: list[str] = []
        for item in value:
            parts.extend(_split_continuity_text(item))
        return parts
    return [str(value).strip()] if str(value).strip() else []


def _looks_like_continuity_noise(text: str) -> bool:
    """Filter out screenplay-like dumps that shouldn't be continuity facts."""
    if len(text) > 300:
        return True
    return False


def _narrative_for_shot(
    shot_id: str,
    planned: tuple[MovieNarrativeBeat, ...],
    fallback: tuple[MovieNarrativeBeat, ...],
) -> MovieNarrativeBeat:
    by_id = {beat.shot_id: beat for beat in planned}
    fallback_by_id = {beat.shot_id: beat for beat in fallback}
    candidate = by_id.get(shot_id)
    base = fallback_by_id[shot_id]
    if candidate is None:
        return base
    return MovieNarrativeBeat(
        shot_id=shot_id,
        story_state_before=candidate.story_state_before or base.story_state_before,
        story_state_after=candidate.story_state_after or base.story_state_after,
        cause_from_previous=candidate.cause_from_previous or base.cause_from_previous,
        narrative_purpose=candidate.narrative_purpose or base.narrative_purpose,
        conflict_or_tension=candidate.conflict_or_tension or base.conflict_or_tension,
        turning_point=candidate.turning_point or base.turning_point,
        sets_up_next=candidate.sets_up_next or base.sets_up_next,
    )
