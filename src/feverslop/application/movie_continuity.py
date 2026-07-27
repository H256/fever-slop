from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from feverslop.domain.movie import (
    CinematicShot,
    MovieBible,
    MovieContinuityCharacterState,
    MovieContinuityLedger,
    MovieContinuityLocationState,
    MovieContinuityPlan,
    MovieContinuityStyleBible,
    MovieNarrativeBeat,
    MovieSceneContinuityPacket,
)
from feverslop.application.movie_common import (
    _looks_like_screenplay_dump,
)
from feverslop.domain.movie_utils import string_list


def apply_movie_continuity_to_shots(shots: tuple[CinematicShot, ...], continuity_plan: MovieContinuityPlan) -> tuple[CinematicShot, ...]:
    narrative_by_id = {beat.shot_id: beat for beat in continuity_plan.narrative_chain}
    packet_by_id = continuity_plan.scene_continuity
    updated = []
    for shot in shots:
        beat = narrative_by_id.get(shot.shot_id)
        packet = packet_by_id.get(shot.shot_id)
        continuity_notes = "; ".join(
            part
            for part in [
                "; ".join(_safe_continuity_facts(packet.incoming)) if packet else "",
                "; ".join(_safe_continuity_facts(packet.required_carryovers)) if packet else "",
                "; ".join(_safe_continuity_facts(packet.outgoing)) if packet else "",
            ]
            if part
        )
        updated.append(
            replace(
                shot,
                continuity_notes=shot.continuity_notes or continuity_notes,
                story_state_before=shot.story_state_before or (beat.story_state_before if beat else ""),
                story_state_after=shot.story_state_after or (beat.story_state_after if beat else ""),
                cause_from_previous=shot.cause_from_previous or (beat.cause_from_previous if beat else ""),
                narrative_purpose=shot.narrative_purpose or (beat.narrative_purpose if beat else ""),
                conflict_or_tension=shot.conflict_or_tension or (beat.conflict_or_tension if beat else ""),
                turning_point=shot.turning_point or (beat.turning_point if beat else ""),
                sets_up_next=shot.sets_up_next or (beat.sets_up_next if beat else ""),
            )
        )
    return tuple(updated)


def build_movie_continuity_fallback(*, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    style_bible = MovieContinuityStyleBible(
        visual_style="; ".join(bible.style_constraints),
        negative_constraints=("visible text", "unmotivated wardrobe changes", "unexplained prop changes"),
    )
    characters = {
        actor.id: MovieContinuityCharacterState(
            character_id=actor.id,
            base_identity=actor.visual_description or actor.name,
            wardrobe=actor.visual_description or actor.name,
            emotional_state=actor.role,
        )
        for actor in bible.actors
    }
    locations = {
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
        shot_characters = {
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
            )
        )
        previous_outgoing = outgoing
        previous_after = state_after
    return MovieContinuityPlan(
        continuity_ledger=MovieContinuityLedger(
            style_bible=style_bible,
            characters=characters,
            locations=locations,
            scene_order=tuple(shot.shot_id for shot in shots),
        ),
        scene_continuity=scene_continuity,
        narrative_chain=tuple(narrative_chain),
    )


def _safe_continuity_facts(value: Any) -> tuple[str, ...]:
    candidates = _split_continuity_text(value)
    facts: list[str] = []
    for candidate in candidates:
        fact = " ".join(str(candidate or "").split()).strip(" .")
        if not fact or _looks_like_screenplay_dump(fact):
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


def normalize_movie_continuity_plan(plan: MovieContinuityPlan, *, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    fallback = build_movie_continuity_fallback(bible=bible, shots=shots)
    shot_ids = [shot.shot_id for shot in shots]
    return MovieContinuityPlan(
        continuity_ledger=plan.continuity_ledger or fallback.continuity_ledger,
        scene_continuity={shot_id: plan.scene_continuity.get(shot_id) or fallback.scene_continuity[shot_id] for shot_id in shot_ids},
        narrative_chain=tuple(_narrative_for_shot(shot_id, plan.narrative_chain, fallback.narrative_chain) for shot_id in shot_ids),
    )


def movie_continuity_plan_to_dict(plan: MovieContinuityPlan) -> dict:
    from dataclasses import asdict
    return asdict(plan)


def movie_continuity_plan_from_dict(data: dict, *, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    fallback = build_movie_continuity_fallback(bible=bible, shots=shots)
    ledger_data = data.get("continuity_ledger") or {}
    style_data = ledger_data.get("style_bible") or {}
    style_bible = MovieContinuityStyleBible(
        visual_style=str(style_data.get("visual_style") or fallback.continuity_ledger.style_bible.visual_style),
        palette=str(style_data.get("palette") or ""),
        lighting=str(style_data.get("lighting") or ""),
        camera=str(style_data.get("camera") or ""),
        negative_constraints=tuple(string_list(style_data.get("negative_constraints"))),
    )
    characters = {
        character_id: _character_state_from_dict(character_id, value)
        for character_id, value in (ledger_data.get("characters") or {}).items()
        if isinstance(value, dict)
    } or fallback.continuity_ledger.characters
    locations = {
        location_id: _location_state_from_dict(location_id, value)
        for location_id, value in (ledger_data.get("locations") or {}).items()
        if isinstance(value, dict)
    } or fallback.continuity_ledger.locations
    scene_data = data.get("scene_continuity") or {}
    scene_continuity = {
        shot.shot_id: _scene_packet_from_dict(shot.shot_id, scene_data.get(shot.shot_id) or {}, fallback.scene_continuity[shot.shot_id])
        for shot in shots
    }
    narrative_data = data.get("narrative_chain") or []
    narrative_chain = tuple(
        _narrative_for_shot(
            shot.shot_id,
            tuple(_narrative_from_dict(item) for item in narrative_data if isinstance(item, dict)),
            fallback.narrative_chain,
        )
        for shot in shots
    )
    return MovieContinuityPlan(
        continuity_ledger=MovieContinuityLedger(
            style_bible=style_bible,
            characters=characters,
            locations=locations,
            scene_order=tuple(str(item) for item in ledger_data.get("scene_order") or fallback.continuity_ledger.scene_order),
        ),
        scene_continuity=scene_continuity,
        narrative_chain=narrative_chain,
    )


def _narrative_for_shot(shot_id: str, planned: tuple[MovieNarrativeBeat, ...], fallback: tuple[MovieNarrativeBeat, ...]) -> MovieNarrativeBeat:
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


def _character_state_from_dict(character_id: str, data: dict) -> MovieContinuityCharacterState:
    return MovieContinuityCharacterState(
        character_id=str(data.get("character_id") or character_id),
        base_identity=str(data.get("base_identity") or ""),
        wardrobe=str(data.get("wardrobe") or data.get("base_identity") or ""),
        carried_props=tuple(string_list(data.get("carried_props"))),
        physical_state=str(data.get("physical_state") or ""),
        emotional_state=str(data.get("emotional_state") or ""),
        last_location=str(data.get("last_location") or ""),
        last_action=str(data.get("last_action") or ""),
    )


def _location_state_from_dict(location_id: str, data: dict) -> MovieContinuityLocationState:
    return MovieContinuityLocationState(
        location_id=str(data.get("location_id") or location_id),
        name=str(data.get("name") or location_id),
        time_of_day=str(data.get("time_of_day") or ""),
        lighting=str(data.get("lighting") or ""),
        props=tuple(string_list(data.get("props"))),
        environmental_state=str(data.get("environmental_state") or data.get("state") or ""),
    )


def _scene_packet_from_dict(shot_id: str, data: dict, fallback: MovieSceneContinuityPacket) -> MovieSceneContinuityPacket:
    location_data = data.get("location") if isinstance(data.get("location"), dict) else {}
    characters_data = data.get("characters") if isinstance(data.get("characters"), dict) else {}
    return MovieSceneContinuityPacket(
        shot_id=str(data.get("shot_id") or shot_id),
        location_id=str(data.get("location_id") or fallback.location_id),
        incoming=tuple(string_list(data.get("incoming"))) or fallback.incoming,
        required_carryovers=tuple(string_list(data.get("required_carryovers"))) or fallback.required_carryovers,
        allowed_changes=tuple(string_list(data.get("allowed_changes"))) or fallback.allowed_changes,
        outgoing=tuple(string_list(data.get("outgoing"))) or fallback.outgoing,
        characters={
            character_id: _character_state_from_dict(character_id, value)
            for character_id, value in characters_data.items()
            if isinstance(value, dict)
        }
        or fallback.characters,
        location=_location_state_from_dict(fallback.location_id, location_data) if location_data else fallback.location,
    )


def _narrative_from_dict(data: dict) -> MovieNarrativeBeat:
    return MovieNarrativeBeat(
        shot_id=str(data.get("shot_id") or ""),
        story_state_before=str(data.get("story_state_before") or ""),
        story_state_after=str(data.get("story_state_after") or ""),
        cause_from_previous=str(data.get("cause_from_previous") or ""),
        narrative_purpose=str(data.get("narrative_purpose") or ""),
        conflict_or_tension=str(data.get("conflict_or_tension") or ""),
        turning_point=str(data.get("turning_point") or ""),
        sets_up_next=str(data.get("sets_up_next") or ""),
    )
