from __future__ import annotations

import re

from feverslop.domain.movie import CinematicShot, StoryArch


class DeterministicMoviePlanner:
    def generate_story_arch(self, *, title: str, source_type: str, story_text: str, desired_length: float) -> StoryArch:
        text = " ".join(story_text.strip().split())
        beats = _split_screenplay_beats(story_text) if source_type == "screenplay" else _split_beats(text)
        return StoryArch(title=title, premise=text, beats=tuple(beats))

    def plan_shots(self, *, story_arch: StoryArch, desired_length: float, width: int, height: int) -> tuple[CinematicShot, ...]:
        beats = story_arch.beats or (story_arch.premise,)
        duration = max(1.0, float(desired_length) / len(beats))
        shots = []
        for index, beat in enumerate(beats, start=1):
            screenplay = _parse_screenplay_beat(beat)
            if screenplay is not None:
                description = screenplay["action"] or screenplay["dialogue"] or screenplay["location"]
                shots.append(
                    CinematicShot(
                        shot_id=f"shot_{index:04}",
                        description=description,
                        duration_seconds=duration,
                        camera=screenplay["camera"],
                        action=screenplay["action"],
                        expression=screenplay["expression"],
                        location=screenplay["location"],
                        dialogue=screenplay["dialogue"],
                    )
                )
                continue
            shots.append(
                CinematicShot(
                    shot_id=f"shot_{index:04}",
                    description=beat,
                    duration_seconds=duration,
                    camera="slow dolly with motivated cinematic framing",
                    action=beat,
                    expression="subtle emotionally grounded acting",
                    location="story-consistent cinematic location",
                )
            )
        return tuple(shots)


def _split_beats(text: str) -> list[str]:
    parts = [part.strip(" .!?") for part in re.split(r"[.!?]+", text.replace("\n", " ")) if part.strip()]
    return parts[:12] or [text]


_HEADING_RE = re.compile(r"^(INT\.|EXT\.|INT/EXT\.)\s+(.+)$", re.IGNORECASE)


def _split_screenplay_beats(text: str) -> list[str]:
    beats: list[str] = []
    heading = ""
    body: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _HEADING_RE.match(line)
        if match:
            if heading:
                beats.append("\n".join([heading, *body]))
            heading = f"{match.group(1).upper()} {match.group(2).strip()}"
            body = []
        elif heading:
            body.append(line)
    if heading:
        beats.append("\n".join([heading, *body]))
    return beats or _split_beats(" ".join(text.strip().split()))


def _parse_screenplay_beat(beat: str) -> dict[str, str] | None:
    lines = [line.strip() for line in beat.splitlines() if line.strip()]
    if not lines:
        return None
    heading = _HEADING_RE.match(lines[0])
    if heading is None:
        return None

    kind = heading.group(1).upper()
    location = heading.group(2).strip()
    dialogue: list[str] = []
    actions: list[str] = []
    index = 1
    while index < len(lines):
        line = lines[index]
        if _is_character_cue(line) and index + 1 < len(lines):
            dialogue.append(f"{line}: {lines[index + 1]}")
            index += 2
            continue
        actions.append(line)
        index += 1

    action = " ".join(actions).strip()
    camera = "controlled interior dolly with motivated cinematic framing" if kind.startswith("INT") else "wide exterior establishing move with cinematic depth"
    return {
        "location": location,
        "dialogue": " ".join(dialogue),
        "action": action,
        "camera": camera,
        "expression": "emotion and facial acting follow the dialogue beats" if dialogue else "subtle emotionally grounded acting",
    }


def _is_character_cue(line: str) -> bool:
    words = line.split()
    return bool(words) and len(words) <= 4 and line.upper() == line and not _HEADING_RE.match(line)
