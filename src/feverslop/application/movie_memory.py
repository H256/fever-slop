from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from feverslop.domain.movie import (
    CinematicShot,
    MovieBible,
    MovieNarrativePlan,
    MovieSceneCard,
    MovieScreenplayArtifact,
    MovieScreenplayScene,
    MovieShotCard,
)

_SCREENPLAY_HEADING_RE = re.compile(r"\b(?:INT|EXT|INT/EXT)\.\s+", re.IGNORECASE)


def generate_movie_screenplay(*, planner, request: Any, bible: MovieBible, story_arch, config: dict, source_text: str) -> MovieScreenplayArtifact:
    generator = getattr(planner, "generate_movie_screenplay", None)
    if callable(generator):
        raw = generator(
            title=request.name,
            source_type=request.source_type,
            story_text=source_text,
            desired_length=float(request.desired_length),
            bible=bible,
            story_arch=story_arch,
            config=config,
        )
        if isinstance(raw, MovieScreenplayArtifact):
            return raw
        if isinstance(raw, dict) and raw.get("scenes"):
            return movie_screenplay_from_dict(raw, fallback_title=request.name, source_type=request.source_type, bible=bible)
    return build_movie_screenplay_fallback(request=request, bible=bible, story_arch=story_arch, config=config)


def build_movie_screenplay_fallback(*, request: Any, bible: MovieBible, story_arch, config: dict) -> MovieScreenplayArtifact:
    dialogue_language = str(config.get("dialogue_language") or (bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    if request.source_type == "screenplay":
        scenes = _screenplay_scenes_from_text(request.story_text, bible=bible)
    else:
        scenes = _screenplay_scenes_from_beats(story_arch.beats or (story_arch.premise,), bible=bible)
    return MovieScreenplayArtifact(
        title=request.name,
        source_type=request.source_type,
        dialogue_language=dialogue_language,
        scenes=scenes,
    )


def generate_movie_narrative_plan(*, planner, request: Any, bible: MovieBible, screenplay: MovieScreenplayArtifact, config: dict) -> MovieNarrativePlan:
    generator = getattr(planner, "generate_movie_narrative_plan", None)
    if callable(generator):
        raw = generator(
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
            }
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
                dramatic_purpose=scene.summary or scene.action or scene.heading,
                story_state_before=shot.story_state_before if shot else "",
                story_state_after=shot.story_state_after if shot else scene.action,
                active_actor_ids=scene.actor_ids or (shot.actor_ids if shot else ()),
                location_id=scene.location_id or (shot.location_id if shot else ""),
                dialogue=scene.dialogue,
            )
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
            )
        )
    return tuple(cards)


def movie_screenplay_to_dict(screenplay: MovieScreenplayArtifact) -> dict:
    return {
        "title": screenplay.title,
        "source_type": screenplay.source_type,
        "dialogue_language": screenplay.dialogue_language,
        "scenes": [asdict(scene) for scene in screenplay.scenes],
    }


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
            )
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
            }
        }
    return {"shot_cards": [asdict(card) for card in cards], "memory_pack": memory_pack}


def _screenplay_scenes_from_text(text: str, *, bible: MovieBible) -> tuple[MovieScreenplayScene, ...]:
    scenes: list[MovieScreenplayScene] = []
    heading = ""
    body: list[str] = []
    start_line = 1
    lines = text.splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if _SCREENPLAY_HEADING_RE.match(line):
            if heading:
                scenes.append(_screenplay_scene_from_parts(len(scenes) + 1, heading, body, start_line=start_line, end_line=line_number - 1, bible=bible))
            heading = line
            body = []
            start_line = line_number
        elif heading:
            body.append(line)
    if heading:
        scenes.append(_screenplay_scene_from_parts(len(scenes) + 1, heading, body, start_line=start_line, end_line=len(lines), bible=bible))
    return tuple(scenes) or _screenplay_scenes_from_beats((text,), bible=bible)


def _screenplay_scene_from_parts(index: int, heading: str, body: list[str], *, start_line: int, end_line: int, bible: MovieBible) -> MovieScreenplayScene:
    dialogue, actions = _split_screenplay_dialogue(body)
    action = " ".join(actions).strip()
    actor_ids = tuple(_valid_actor_ids(_dialogue_actor_ids(dialogue), bible)) or _default_bible_actor_ids(bible)
    location_id = _location_id_from_heading(heading, bible)
    summary = action or dialogue or heading
    return MovieScreenplayScene(
        scene_id=f"scene_{index:04}",
        heading=heading,
        summary=summary,
        action=action,
        dialogue=dialogue,
        actor_ids=actor_ids,
        location_id=location_id,
        source_span=f"lines:{start_line}-{end_line}",
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


def _split_screenplay_dialogue(lines: list[str]) -> tuple[str, list[str]]:
    dialogue: list[str] = []
    actions: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_screenplay_character_cue(line) and index + 1 < len(lines):
            dialogue.append(f"{line}: {lines[index + 1]}")
            index += 2
            continue
        if ":" in line and line.split(":", 1)[0].strip().isupper():
            dialogue.append(line)
        else:
            actions.append(line)
        index += 1
    return " ".join(dialogue).strip(), actions


def _is_screenplay_character_cue(line: str) -> bool:
    words = line.split()
    return bool(words) and len(words) <= 4 and line.upper() == line and not _SCREENPLAY_HEADING_RE.match(line)


def _dialogue_actor_ids(dialogue: str) -> list[str]:
    ids = []
    for match in re.finditer(r"\b([A-Z][A-Z0-9 _'-]{1,30}):", dialogue):
        actor_id = _safe_id(match.group(1))
        if actor_id and actor_id not in ids:
            ids.append(actor_id)
    return ids


def _valid_actor_ids(raw_ids: object, bible: MovieBible) -> tuple[str, ...]:
    valid = {actor.id for actor in bible.actors}
    ids = []
    for raw_id in _string_list(raw_ids):
        actor_id = _safe_id(raw_id)
        if actor_id in valid and actor_id not in ids:
            ids.append(actor_id)
    return tuple(ids)


def _valid_location_id(location_id: str, bible: MovieBible) -> str:
    valid = {location.id for location in bible.locations}
    safe = _safe_id(location_id)
    if safe in valid:
        return safe
    return bible.locations[0].id if bible.locations else "primary_location"


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


def _safe_id(value: object) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return raw


def _string_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
