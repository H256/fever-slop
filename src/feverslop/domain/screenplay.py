from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(INT\.|EXT\.|INT/EXT\.)\s+(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedScreenplayScene:
    heading: str
    kind: str
    location: str
    body: tuple[str, ...]
    action: str
    dialogue: str
    start_line: int
    end_line: int


def looks_like_screenplay(text: str) -> bool:
    return any(HEADING_RE.match(normalize_screenplay_line(line)) for line in str(text or "").splitlines())


def parse_screenplay(text: str) -> tuple[ParsedScreenplayScene, ...]:
    scenes: list[ParsedScreenplayScene] = []
    heading = ""
    kind = ""
    location = ""
    body: list[str] = []
    start_line = 1
    last_line = 1
    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = normalize_screenplay_line(raw_line)
        if not line or is_screenplay_metadata_line(line):
            continue
        match = HEADING_RE.match(line)
        if match:
            if heading:
                scenes.append(_scene_from_parts(heading, kind, location, body, start_line=start_line, end_line=last_line))
            kind = match.group(1).upper()
            location = match.group(2).strip()
            heading = f"{kind} {location}"
            body = []
            start_line = line_number
            last_line = line_number
            continue
        if heading:
            body.append(line)
            last_line = line_number
    if heading:
        scenes.append(_scene_from_parts(heading, kind, location, body, start_line=start_line, end_line=last_line))
    return tuple(scenes)


def normalize_screenplay_line(raw_line: str) -> str:
    line = str(raw_line or "").strip()
    previous = None
    while line and previous != line:
        previous = line
        for marker in ("**", "__", "*", "_"):
            if line.startswith(marker) and line.endswith(marker) and len(line) >= len(marker) * 2:
                line = line[len(marker) : -len(marker)].strip()
    return line


def is_screenplay_metadata_line(line: str) -> bool:
    upper = line.upper().strip()
    return upper.startswith("TITLE:") or bool(re.match(r"SCENE\s+\d+\b", upper)) or upper in {"FADE OUT.", "FADE TO BLACK."}


def split_screenplay_dialogue(lines: list[str] | tuple[str, ...]) -> tuple[str, list[str]]:
    dialogue: list[str] = []
    actions: list[str] = []
    index = 0
    cleaned = [normalize_screenplay_line(line) for line in lines if normalize_screenplay_line(line)]
    while index < len(cleaned):
        line = cleaned[index]
        if is_parenthetical_line(line):
            index += 1
            continue
        if is_screenplay_character_cue(line):
            dialogue_line_index = _next_dialogue_line_index(cleaned, index + 1)
            if dialogue_line_index is not None:
                dialogue.append(f"{line}: {cleaned[dialogue_line_index]}")
                index = dialogue_line_index + 1
                continue
        if ":" in line and line.split(":", 1)[0].strip().isupper():
            dialogue.append(line)
        else:
            actions.append(line)
        index += 1
    return " ".join(dialogue).strip(), actions


def is_screenplay_character_cue(line: str) -> bool:
    normalized = normalize_screenplay_line(line)
    words = normalized.split()
    return bool(words) and len(words) <= 4 and normalized.upper() == normalized and not HEADING_RE.match(normalized)


def is_parenthetical_line(line: str) -> bool:
    normalized = normalize_screenplay_line(line)
    return normalized.startswith("(") and normalized.endswith(")")


def _scene_from_parts(heading: str, kind: str, location: str, body: list[str], *, start_line: int, end_line: int) -> ParsedScreenplayScene:
    dialogue, actions = split_screenplay_dialogue(tuple(body))
    action = " ".join(actions).strip()
    return ParsedScreenplayScene(
        heading=heading,
        kind=kind,
        location=location,
        body=tuple(body),
        action=action,
        dialogue=dialogue,
        start_line=start_line,
        end_line=end_line,
    )


def _next_dialogue_line_index(lines: list[str], start: int) -> int | None:
    index = start
    while index < len(lines):
        line = lines[index]
        if is_parenthetical_line(line):
            index += 1
            continue
        if is_screenplay_character_cue(line) or HEADING_RE.match(line):
            return None
        return index
    return None
