from __future__ import annotations

from dataclasses import replace

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
from feverslop.domain.movie_continuity import (
    _narrative_for_shot,
    _safe_continuity_facts,
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
            ),
        )
    return tuple(updated)


def build_movie_continuity_fallback(*, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    return MovieContinuityPlan.fallback(bible=bible, shots=shots)


def normalize_movie_continuity_plan(plan: MovieContinuityPlan, *, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    return plan.normalize(bible=bible, shots=shots)


def movie_continuity_plan_to_dict(plan: MovieContinuityPlan) -> dict:
    return {
        "continuity_ledger": _ledger_to_dict(plan.continuity_ledger),
        "scene_continuity": {
            shot_id: _packet_to_dict(packet)
            for shot_id, packet in plan.scene_continuity.items()
        },
        "narrative_chain": [_beat_to_dict(beat) for beat in plan.narrative_chain],
    }


def _ledger_to_dict(ledger: MovieContinuityLedger) -> dict:
    return {
        "style_bible": {
            "visual_style": ledger.style_bible.visual_style,
            "palette": ledger.style_bible.palette,
            "lighting": ledger.style_bible.lighting,
            "camera": ledger.style_bible.camera,
            "negative_constraints": ledger.style_bible.negative_constraints,
        },
        "characters": {
            cid: {
                "character_id": state.character_id,
                "base_identity": state.base_identity,
                "wardrobe": state.wardrobe,
                "carried_props": state.carried_props,
                "physical_state": state.physical_state,
                "emotional_state": state.emotional_state,
                "last_location": state.last_location,
                "last_action": state.last_action,
            }
            for cid, state in ledger.characters.items()
        },
        "locations": {
            lid: {
                "location_id": loc.location_id,
                "name": loc.name,
                "time_of_day": loc.time_of_day,
                "lighting": loc.lighting,
                "props": loc.props,
                "environmental_state": loc.environmental_state,
            }
            for lid, loc in ledger.locations.items()
        },
        "scene_order": ledger.scene_order,
    }


def _packet_to_dict(packet: MovieSceneContinuityPacket) -> dict:
    return {
        "shot_id": packet.shot_id,
        "location_id": packet.location_id,
        "incoming": packet.incoming,
        "required_carryovers": packet.required_carryovers,
        "allowed_changes": packet.allowed_changes,
        "outgoing": packet.outgoing,
        "characters": (
            {
                cid: {
                    "character_id": state.character_id,
                    "base_identity": state.base_identity,
                    "wardrobe": state.wardrobe,
                    "carried_props": state.carried_props,
                    "physical_state": state.physical_state,
                    "emotional_state": state.emotional_state,
                    "last_location": state.last_location,
                    "last_action": state.last_action,
                }
                for cid, state in packet.characters.items()
            }
            if packet.characters is not None
            else None
        ),
        "location": (
            {
                "location_id": packet.location.location_id,
                "name": packet.location.name,
                "time_of_day": packet.location.time_of_day,
                "lighting": packet.location.lighting,
                "props": packet.location.props,
                "environmental_state": packet.location.environmental_state,
            }
            if packet.location is not None
            else None
        ),
    }


def _beat_to_dict(beat: MovieNarrativeBeat) -> dict:
    return {
        "shot_id": beat.shot_id,
        "story_state_before": beat.story_state_before,
        "story_state_after": beat.story_state_after,
        "cause_from_previous": beat.cause_from_previous,
        "narrative_purpose": beat.narrative_purpose,
        "conflict_or_tension": beat.conflict_or_tension,
        "turning_point": beat.turning_point,
        "sets_up_next": beat.sets_up_next,
    }


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
