from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from feverslop.domain.movie import (
    CinematicShot,
    MovieAct,
    MovieBible,
    MovieCharacterArc,
    MovieNarrativePlan,
    MovieSceneBlueprint,
    MovieSceneCard,
    MovieScreenplayArtifact,
    MovieScreenplayScene,
    MovieSetupPayoff,
    MovieShotCard,
    MovieStoryDesign,
    MovieTurningPoint,
)
from feverslop.domain.movie_utils import (
    safe_id,
    string_list,
    transition_from_previous,
)
from feverslop.domain.screenplay import parse_screenplay
from feverslop.ports.movie import ScenePlanningPort


def generate_movie_story_design(*, planner: ScenePlanningPort, request: Any, bible: MovieBible, story_arch, config: dict, source_text: str) -> MovieStoryDesign:
    raw = planner.generate_movie_story_design(
        title=request.name,
        source_type=request.source_type,
        story_text=source_text,
        desired_length=float(request.desired_length),
        bible=bible,
        story_arch=story_arch,
        config=config,
    )
    if isinstance(raw, MovieStoryDesign):
        return raw
    if isinstance(raw, dict) and raw.get("scene_blueprint"):
        max_actors = int(config.get("max_scene_actors") or (bible.runtime_constraints or {}).get("max_scene_actors") or 4)
        return movie_story_design_from_dict(raw, fallback_title=request.name, bible=bible, max_scene_actors=max_actors)
    return build_movie_story_design_fallback(request=request, bible=bible, story_arch=story_arch, config=config)


def generate_movie_screenplay(*, planner: ScenePlanningPort, request: Any, bible: MovieBible, story_arch, story_design: MovieStoryDesign, config: dict, source_text: str) -> MovieScreenplayArtifact:
    raw = planner.generate_movie_screenplay(
        title=request.name,
        source_type=request.source_type,
        story_text=source_text,
        desired_length=float(request.desired_length),
        bible=bible,
        story_arch=story_arch,
        story_design=story_design,
        config=config,
    )
    if isinstance(raw, MovieScreenplayArtifact):
        return raw
    if isinstance(raw, dict) and raw.get("scenes"):
        return movie_screenplay_from_dict(raw, fallback_title=request.name, source_type=request.source_type, bible=bible)
    return build_movie_screenplay_fallback(request=request, bible=bible, story_arch=story_arch, story_design=story_design, config=config)


def build_movie_story_design_fallback(*, request: Any, bible: MovieBible, story_arch, config: dict) -> MovieStoryDesign:
    if request.source_type == "screenplay":
        parsed_scenes = _screenplay_scenes_from_text(request.story_text, bible=bible)
    else:
        parsed_scenes = _screenplay_scenes_from_beats(story_arch.beats or (story_arch.premise,), bible=bible)
    if not parsed_scenes:
        parsed_scenes = _screenplay_scenes_from_beats((story_arch.premise,), bible=bible)
    total_duration = float(getattr(request, "desired_length", 0) or len(parsed_scenes) or 1)
    expected_duration = max(1.0, total_duration / max(1, len(parsed_scenes)))
    max_actors = int(config.get("max_scene_actors") or (bible.runtime_constraints or {}).get("max_scene_actors") or 4)
    blueprints = tuple(
        MovieSceneBlueprint(
            scene_id=scene.scene_id,
            purpose=_scene_purpose(scene, index, len(parsed_scenes)),
            conflict=_scene_conflict(scene),
            emotional_turn=_scene_emotional_turn(index, len(parsed_scenes)),
            subtext=_scene_subtext(scene),
            dialogue_function=_scene_dialogue_function(scene),
            required_actors=tuple(scene.actor_ids[:max_actors]),
            location_id=scene.location_id,
            expected_duration=expected_duration,
        )
        for index, scene in enumerate(parsed_scenes, start=1)
    )
    scene_ids = tuple(blueprint.scene_id for blueprint in blueprints)
    acts = (
        MovieAct(act_id="act_1", title="Setup", purpose="Establish the premise, world, and central dramatic pressure.", scene_ids=_act_slice(scene_ids, 0, 1 / 3)),
        MovieAct(act_id="act_2", title="Confrontation", purpose="Escalate conflict and force irreversible choices.", scene_ids=_act_slice(scene_ids, 1 / 3, 2 / 3)),
        MovieAct(act_id="act_3", title="Resolution", purpose="Resolve the central dramatic pressure and leave the final emotional state.", scene_ids=_act_slice(scene_ids, 2 / 3, 1)),
    )
    turning_points = (
        MovieTurningPoint(id="inciting_turn", scene_id=scene_ids[0], description="The first scene turns the premise into immediate dramatic pressure."),
        MovieTurningPoint(id="final_turn", scene_id=scene_ids[-1], description="The final scene resolves the main emotional and narrative pressure."),
    )
    setup_payoffs = (
        MovieSetupPayoff(
            id="central_thread",
            setup_scene_id=scene_ids[0],
            payoff_scene_id=scene_ids[-1],
            description="The opening dramatic pressure receives its final consequence by the end.",
        ),
    )
    arcs = tuple(
        MovieCharacterArc(
            actor_id=actor.id,
            want=f"{actor.name} pursues the visible goal implied by the premise.",
            need=f"{actor.name} must confront the emotional cost of the premise.",
            starting_state="Defined by the opening pressure.",
            ending_state="Changed by the final consequence.",
        )
        for actor in bible.actors
    )
    return MovieStoryDesign(
        title=getattr(request, "name", "") or story_arch.title,
        premise=story_arch.premise,
        theme=str(config.get("theme") or "The premise is expressed through escalating choices and consequences.").strip(),
        act_structure=acts,
        turning_points=turning_points,
        setup_payoff_threads=setup_payoffs,
        character_arcs=arcs,
        scene_blueprint=blueprints,
    )


def build_movie_screenplay_fallback(*, request: Any, bible: MovieBible, story_arch, story_design: MovieStoryDesign | None = None, config: dict) -> MovieScreenplayArtifact:
    dialogue_language = str(config.get("dialogue_language") or (bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    if request.source_type == "screenplay":
        scenes = _screenplay_scenes_from_text(request.story_text, bible=bible)
    else:
        scenes = _screenplay_scenes_from_beats(story_arch.beats or (story_arch.premise,), bible=bible)
    scenes = _apply_story_design_to_screenplay(scenes, story_design)
    return MovieScreenplayArtifact(
        title=request.name,
        source_type=request.source_type,
        dialogue_language=dialogue_language,
        scenes=scenes,
    )


def generate_movie_narrative_plan(*, planner: ScenePlanningPort, request: Any, bible: MovieBible, screenplay: MovieScreenplayArtifact, config: dict) -> MovieNarrativePlan:
    raw = planner.generate_movie_narrative_plan(
        title=request.name,
        source_type=request.source_type,
        desired_length=float(request.desired_length),
        bible=bible,
        screenplay=screenplay,
        config=config,
    )
    if isinstance(raw, MovieNarrativePlan):
        return raw
    if isinstance(raw, dict) and (raw.get("sequences") or raw.get("causal_chain")):
        return movie_narrative_plan_from_dict(raw, fallback_title=request.name)
    return build_movie_narrative_plan_fallback(screenplay=screenplay)


def build_movie_narrative_plan_fallback(*, screenplay: MovieScreenplayArtifact) -> MovieNarrativePlan:
    scene_ids = tuple(scene.scene_id for scene in screenplay.scenes)
    sequences = (
        {
            "sequence_id": "sequence_0001",
            "title": screenplay.title,
            "scene_ids": list(scene_ids),
            "dramatic_function": "Carry the movie premise through a causal beginning, middle, and resolution.",
        },
    )
    causal_chain = []
    previous = "The movie begins from the stated premise."
    for index, scene in enumerate(screenplay.scenes, start=1):
        after = scene.action or scene.summary or scene.heading
        causal_chain.append(
            {
                "scene_id": scene.scene_id,
                "story_state_before": previous,
                "story_state_after": after,
                "cause_from_previous": "Opening scene establishes the premise." if index == 1 else f"Previous scene leaves: {previous}",
                "sets_up_next": screenplay.scenes[index].summary if index < len(screenplay.scenes) else "The final scene resolves the current arc.",
            },
        )
        previous = after
    return MovieNarrativePlan(title=screenplay.title, sequences=sequences, causal_chain=tuple(causal_chain), open_threads=())


def build_movie_scene_cards(*, screenplay: MovieScreenplayArtifact, shots: tuple[CinematicShot, ...]) -> tuple[MovieSceneCard, ...]:
    cards = []
    for index, scene in enumerate(screenplay.scenes):
        shot = shots[min(index, len(shots) - 1)] if shots else None
        shot_ids = (shot.shot_id,) if shot else ()
        cards.append(
            MovieSceneCard(
                scene_id=scene.scene_id,
                shot_ids=shot_ids,
                dramatic_purpose=scene.dramatic_purpose or scene.summary or scene.action or scene.heading,
                story_state_before=shot.story_state_before if shot else "",
                story_state_after=shot.story_state_after if shot else scene.action,
                active_actor_ids=scene.actor_ids or (shot.actor_ids if shot else ()),
                location_id=scene.location_id or (shot.location_id if shot else ""),
                dialogue=scene.dialogue,
            ),
        )
    return tuple(cards)


def build_movie_shot_cards(*, shots: tuple[CinematicShot, ...], scene_cards: tuple[MovieSceneCard, ...]) -> tuple[MovieShotCard, ...]:
    scene_by_shot = {shot_id: card for card in scene_cards for shot_id in card.shot_ids}
    cards = []
    for shot in shots:
        scene_card = scene_by_shot.get(shot.shot_id)
        scene_id = scene_card.scene_id if scene_card else shot.shot_id.replace("shot", "scene", 1)
        action = shot.action or shot.description
        cards.append(
            MovieShotCard(
                shot_id=shot.shot_id,
                scene_id=scene_id,
                action=action,
                camera=shot.camera,
                acting=shot.expression,
                dialogue=shot.dialogue,
                start_frame_brief=_start_frame_brief(shot),
                end_frame_brief=_end_frame_brief(shot),
                transition_from_previous=transition_from_previous(shot.transition_from_previous),
                transition_reason=_transition_reason(shot),
            ),
        )
    return tuple(cards)


def _transition_reason(shot: CinematicShot) -> str:
    if transition_from_previous(shot.transition_from_previous) != "continuous":
        return "hard cut or new setup"
    return "planned as a direct continuation of the previous shot"


def movie_screenplay_to_dict(screenplay: MovieScreenplayArtifact) -> dict:
    return {
        "title": screenplay.title,
        "source_type": screenplay.source_type,
        "dialogue_language": screenplay.dialogue_language,
        "scenes": [asdict(scene) for scene in screenplay.scenes],
    }


def movie_story_design_to_dict(design: MovieStoryDesign) -> dict:
    return {
        "title": design.title,
        "premise": design.premise,
        "theme": design.theme,
        "act_structure": [asdict(act) for act in design.act_structure],
        "turning_points": [asdict(turn) for turn in design.turning_points],
        "setup_payoff_threads": [asdict(thread) for thread in design.setup_payoff_threads],
        "character_arcs": [asdict(arc) for arc in design.character_arcs],
        "scene_blueprint": [asdict(blueprint) for blueprint in design.scene_blueprint],
    }


def movie_story_design_from_dict(data: dict, *, fallback_title: str, bible: MovieBible, max_scene_actors: int = 4) -> MovieStoryDesign:
    return MovieStoryDesign(
        title=str(data.get("title") or fallback_title),
        premise=str(data.get("premise") or "").strip(),
        theme=str(data.get("theme") or "").strip(),
        act_structure=tuple(
            MovieAct(
                act_id=str(item.get("act_id") or item.get("id") or f"act_{index}"),
                title=str(item.get("title") or f"Act {index}"),
                purpose=str(item.get("purpose") or item.get("dramatic_function") or "").strip(),
                scene_ids=tuple(str(scene_id) for scene_id in item.get("scene_ids") or [] if str(scene_id).strip()),
            )
            for index, item in enumerate(data.get("act_structure") or [], start=1)
            if isinstance(item, dict)
        ),
        turning_points=tuple(
            MovieTurningPoint(
                id=str(item.get("id") or f"turn_{index}"),
                scene_id=str(item.get("scene_id") or "").strip(),
                description=str(item.get("description") or "").strip(),
            )
            for index, item in enumerate(data.get("turning_points") or [], start=1)
            if isinstance(item, dict)
        ),
        setup_payoff_threads=tuple(
            MovieSetupPayoff(
                id=str(item.get("id") or f"thread_{index}"),
                setup_scene_id=str(item.get("setup_scene_id") or "").strip(),
                payoff_scene_id=str(item.get("payoff_scene_id") or "").strip(),
                description=str(item.get("description") or "").strip(),
            )
            for index, item in enumerate(data.get("setup_payoff_threads") or [], start=1)
            if isinstance(item, dict)
        ),
        character_arcs=tuple(
            MovieCharacterArc(
                actor_id=_valid_actor_id(str(item.get("actor_id") or ""), bible),
                want=str(item.get("want") or "").strip(),
                need=str(item.get("need") or "").strip(),
                starting_state=str(item.get("starting_state") or "").strip(),
                ending_state=str(item.get("ending_state") or "").strip(),
            )
            for item in data.get("character_arcs") or []
            if isinstance(item, dict) and _valid_actor_id(str(item.get("actor_id") or ""), bible)
        ),
        scene_blueprint=tuple(
            MovieSceneBlueprint(
                scene_id=str(item.get("scene_id") or f"scene_{index:04}"),
                purpose=str(item.get("purpose") or item.get("dramatic_purpose") or "").strip(),
                conflict=str(item.get("conflict") or "").strip(),
                emotional_turn=str(item.get("emotional_turn") or "").strip(),
                subtext=str(item.get("subtext") or "").strip(),
                dialogue_function=str(item.get("dialogue_function") or "").strip(),
                required_actors=tuple(_valid_actor_ids(item.get("required_actors") or item.get("actor_ids") or [], bible)[:max_scene_actors]),
                location_id=_valid_location_id(str(item.get("location_id") or ""), bible),
                expected_duration=float(item.get("expected_duration") or item.get("duration_seconds") or 0),
            )
            for index, item in enumerate(data.get("scene_blueprint") or [], start=1)
            if isinstance(item, dict)
        ),
    )


def movie_screenplay_from_dict(data: dict, *, fallback_title: str, source_type: str, bible: MovieBible) -> MovieScreenplayArtifact:
    scenes = []
    for index, raw in enumerate(data.get("scenes") or [], start=1):
        if not isinstance(raw, dict):
            continue
        scenes.append(
            MovieScreenplayScene(
                scene_id=str(raw.get("scene_id") or f"scene_{index:04}"),
                heading=str(raw.get("heading") or f"Scene {index}").strip(),
                summary=str(raw.get("summary") or raw.get("action") or "").strip(),
                action=str(raw.get("action") or raw.get("summary") or "").strip(),
                dialogue=str(raw.get("dialogue") or "").strip(),
                actor_ids=tuple(_valid_actor_ids(raw.get("actor_ids") or [], bible)),
                location_id=_valid_location_id(str(raw.get("location_id") or ""), bible),
                source_span=str(raw.get("source_span") or "").strip(),
                dramatic_purpose=str(raw.get("dramatic_purpose") or "").strip(),
                conflict=str(raw.get("conflict") or "").strip(),
                emotional_turn=str(raw.get("emotional_turn") or "").strip(),
                subtext=str(raw.get("subtext") or "").strip(),
                dialogue_function=str(raw.get("dialogue_function") or "").strip(),
            ),
        )
    if not scenes:
        scenes = list(_screenplay_scenes_from_beats((str(data.get("premise") or fallback_title),), bible=bible))
    return MovieScreenplayArtifact(
        title=str(data.get("title") or fallback_title),
        source_type=str(data.get("source_type") or source_type),
        dialogue_language=str(data.get("dialogue_language") or (bible.runtime_constraints or {}).get("dialogue_language") or "").strip(),
        scenes=tuple(scenes),
    )


def movie_screenplay_to_markdown(screenplay: MovieScreenplayArtifact) -> str:
    parts = [f"# {screenplay.title}", ""]
    for scene in screenplay.scenes:
        parts.extend([f"## {scene.heading}", "", scene.action or scene.summary])
        metadata = [
            f"Purpose: {scene.dramatic_purpose}" if scene.dramatic_purpose else "",
            f"Conflict: {scene.conflict}" if scene.conflict else "",
            f"Emotional turn: {scene.emotional_turn}" if scene.emotional_turn else "",
            f"Subtext: {scene.subtext}" if scene.subtext else "",
            f"Dialogue function: {scene.dialogue_function}" if scene.dialogue_function else "",
        ]
        metadata = [item for item in metadata if item]
        if metadata:
            parts.extend(["", *metadata])
        if scene.dialogue:
            parts.extend(["", scene.dialogue])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def movie_narrative_plan_to_dict(plan: MovieNarrativePlan) -> dict:
    return {
        "title": plan.title,
        "sequences": list(plan.sequences),
        "causal_chain": list(plan.causal_chain),
        "open_threads": list(plan.open_threads),
    }


def movie_narrative_plan_from_dict(data: dict, *, fallback_title: str) -> MovieNarrativePlan:
    return MovieNarrativePlan(
        title=str(data.get("title") or fallback_title),
        sequences=tuple(item for item in data.get("sequences") or [] if isinstance(item, dict)),
        causal_chain=tuple(item for item in data.get("causal_chain") or [] if isinstance(item, dict)),
        open_threads=tuple(str(item).strip() for item in data.get("open_threads") or [] if str(item).strip()),
    )


def movie_scene_cards_to_dict(cards: tuple[MovieSceneCard, ...]) -> dict:
    return {"scene_cards": [asdict(card) for card in cards]}


def movie_scene_cards_from_dict(data: dict) -> tuple[MovieSceneCard, ...]:
    return tuple(
        MovieSceneCard(
            scene_id=str(card.get("scene_id") or ""),
            shot_ids=tuple(str(shot_id) for shot_id in card.get("shot_ids") or []),
            dramatic_purpose=str(card.get("dramatic_purpose") or ""),
            story_state_before=str(card.get("story_state_before") or ""),
            story_state_after=str(card.get("story_state_after") or ""),
            active_actor_ids=tuple(str(actor_id) for actor_id in card.get("active_actor_ids") or []),
            location_id=str(card.get("location_id") or ""),
            dialogue=str(card.get("dialogue") or ""),
        )
        for card in data.get("scene_cards") or []
        if isinstance(card, dict)
    )


def movie_shot_cards_to_dict(cards: tuple[MovieShotCard, ...]) -> dict:
    memory_pack = {}
    if cards:
        first = cards[0]
        memory_pack = {
            "current_shot": {
                "shot_id": first.shot_id,
                "description": first.action,
                "scene_id": first.scene_id,
            },
        }
    return {"shot_cards": [asdict(card) for card in cards], "memory_pack": memory_pack}


def _screenplay_scenes_from_text(text: str, *, bible: MovieBible) -> tuple[MovieScreenplayScene, ...]:
    scenes = [
        _screenplay_scene_from_parsed(index, scene, bible=bible)
        for index, scene in enumerate(parse_screenplay(text), start=1)
    ]
    return tuple(scenes) or _screenplay_scenes_from_beats((text,), bible=bible)


def _screenplay_scene_from_parsed(index: int, scene, *, bible: MovieBible) -> MovieScreenplayScene:
    action = scene.action
    dialogue = scene.dialogue
    actor_ids = tuple(_valid_actor_ids(_dialogue_actor_ids(dialogue), bible)) or _default_bible_actor_ids(bible)
    location_id = _location_id_from_heading(scene.heading, bible)
    summary = action or dialogue or scene.heading
    return MovieScreenplayScene(
        scene_id=f"scene_{index:04}",
        heading=scene.heading,
        summary=summary,
        action=action,
        dialogue=dialogue,
        actor_ids=actor_ids,
        location_id=location_id,
        source_span=f"lines:{scene.start_line}-{scene.end_line}",
    )


def _screenplay_scenes_from_beats(beats: tuple[str, ...], *, bible: MovieBible) -> tuple[MovieScreenplayScene, ...]:
    actor_ids = _default_bible_actor_ids(bible)
    location_id = bible.locations[0].id if bible.locations else "primary_location"
    return tuple(
        MovieScreenplayScene(
            scene_id=f"scene_{index:04}",
            heading=f"SCENE {index}",
            summary=str(beat).strip(),
            action=str(beat).strip(),
            actor_ids=actor_ids,
            location_id=location_id,
            source_span=f"beat:{index}",
        )
        for index, beat in enumerate(beats, start=1)
        if str(beat).strip()
    )


def _dialogue_actor_ids(dialogue: str) -> list[str]:
    ids = []
    for match in re.finditer(r"\b([A-Z][A-Z0-9 _'-]{1,30}):", dialogue):
        actor_id = safe_id(match.group(1))
        if actor_id and actor_id not in ids:
            ids.append(actor_id)
    return ids


def _valid_actor_ids(raw_ids: Any, bible: MovieBible) -> tuple[str, ...]:
    valid = {actor.id for actor in bible.actors}
    ids = []
    for raw_id in string_list(raw_ids):
        actor_id = safe_id(raw_id)
        if actor_id in valid and actor_id not in ids:
            ids.append(actor_id)
    return tuple(ids)


def _valid_actor_id(raw_id: str, bible: MovieBible) -> str:
    ids = _valid_actor_ids([raw_id], bible)
    return ids[0] if ids else ""


def _valid_location_id(location_id: str, bible: MovieBible) -> str:
    valid = {location.id for location in bible.locations}
    safe = safe_id(location_id)
    if safe in valid:
        return safe
    return bible.locations[0].id if bible.locations else "primary_location"


def _apply_story_design_to_screenplay(scenes: tuple[MovieScreenplayScene, ...], story_design: MovieStoryDesign | None) -> tuple[MovieScreenplayScene, ...]:
    if story_design is None:
        return scenes
    blueprint_by_scene_id = {blueprint.scene_id: blueprint for blueprint in story_design.scene_blueprint}
    enriched = []
    for scene in scenes:
        blueprint = blueprint_by_scene_id.get(scene.scene_id)
        if blueprint is None:
            enriched.append(scene)
            continue
        enriched.append(
            MovieScreenplayScene(
                scene_id=scene.scene_id,
                heading=scene.heading,
                summary=scene.summary,
                action=scene.action,
                dialogue=scene.dialogue,
                actor_ids=scene.actor_ids or blueprint.required_actors,
                location_id=scene.location_id or blueprint.location_id,
                source_span=scene.source_span,
                dramatic_purpose=blueprint.purpose,
                conflict=blueprint.conflict,
                emotional_turn=blueprint.emotional_turn,
                subtext=blueprint.subtext,
                dialogue_function=blueprint.dialogue_function,
            ),
        )
    return tuple(enriched)


def _act_slice(scene_ids: tuple[str, ...], start_ratio: float, end_ratio: float) -> tuple[str, ...]:
    if not scene_ids:
        return ()
    start = min(len(scene_ids) - 1, int(len(scene_ids) * start_ratio))
    end = max(start + 1, int(len(scene_ids) * end_ratio))
    return scene_ids[start:min(len(scene_ids), end)]


def _scene_purpose(scene: MovieScreenplayScene, index: int, total: int) -> str:
    if index == 1:
        return f"Establish the central situation: {scene.summary or scene.action or scene.heading}"
    if index == total:
        return f"Resolve the dramatic consequence: {scene.summary or scene.action or scene.heading}"
    return f"Escalate the premise through a concrete story beat: {scene.summary or scene.action or scene.heading}"


def _scene_conflict(scene: MovieScreenplayScene) -> str:
    base = scene.summary or scene.action or scene.heading
    return f"The visible goal is pressured by opposition or uncertainty in: {base}"


def _scene_emotional_turn(index: int, total: int) -> str:
    if index == 1:
        return "The emotional state moves from orientation into tension."
    if index == total:
        return "The emotional state lands on the final consequence."
    return "The emotional state changes under escalating pressure."


def _scene_subtext(scene: MovieScreenplayScene) -> str:
    if scene.dialogue:
        return "Characters speak around the emotional cost rather than explaining it directly."
    return "The scene carries its meaning through behavior, environment, and withheld emotion."


def _scene_dialogue_function(scene: MovieScreenplayScene) -> str:
    if scene.dialogue:
        return "Dialogue reveals pressure, relationship, and decision rather than exposition."
    return "No spoken dialogue; silence and action carry the dramatic information."


def _default_bible_actor_ids(bible: MovieBible) -> tuple[str, ...]:
    return (bible.actors[0].id,) if bible.actors else ("main_character",)


def _location_id_from_heading(heading: str, bible: MovieBible) -> str:
    lower = heading.lower()
    for location in bible.locations:
        if location.id.lower() in lower or location.name.lower() in lower:
            return location.id
    return bible.locations[0].id if bible.locations else "primary_location"


def _start_frame_brief(shot: CinematicShot) -> str:
    return f"Opening frame: {shot.action or shot.description}; actors {', '.join(shot.actor_ids) or 'none'} in {shot.location or shot.location_id}."


def _end_frame_brief(shot: CinematicShot) -> str:
    return f"Ending frame: {shot.story_state_after or shot.action or shot.description}; preserve actor identity and location geography."



