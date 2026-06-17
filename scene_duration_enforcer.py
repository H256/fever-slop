from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class SrtScene:
    scene: int
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_srt_timestamp(value: str) -> float:
    value = value.strip()
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis_total = round(seconds * 1000)

    millis = millis_total % 1000
    total_seconds = millis_total // 1000

    sec = total_seconds % 60
    total_minutes = total_seconds // 60

    minute = total_minutes % 60
    hour = total_minutes // 60

    return f"{hour:02}:{minute:02}:{sec:02},{millis:03}"


def parse_scene_srt(path: str | Path) -> list[SrtScene]:
    text = Path(path).read_text(encoding="utf-8").strip()

    if not text:
        return []

    blocks = re.split(r"\n\s*\n", text)
    scenes: list[SrtScene] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if len(lines) < 2:
            continue

        try:
            scene_no = int(lines[0])
        except ValueError:
            continue

        match = re.match(
            r"(.+?)\s*-->\s*(.+)",
            lines[1],
        )

        if not match:
            continue

        start = parse_srt_timestamp(match.group(1))
        end = parse_srt_timestamp(match.group(2))
        body = "\n".join(lines[2:]) if len(lines) > 2 else ""

        scenes.append(SrtScene(scene=scene_no, start=start, end=end, text=body))

    return scenes


def write_scene_srt(path: str | Path, scenes: list[SrtScene]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks = []

    for index, scene in enumerate(scenes, start=1):
        body = scene.text or f"Scene {index}"
        blocks.append(
            "\n".join([
                str(index),
                f"{format_srt_timestamp(scene.start)} --> {format_srt_timestamp(scene.end)}",
                body,
            ])
        )

    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def enforce_scene_duration_constraints(
    scenes: list[SrtScene],
    min_duration: float,
    max_duration: float,
) -> list[SrtScene]:
    """
    Repairs scene durations after beat-based SRT generation.

    Goal:
    - No scene shorter than min_duration unless the entire song/remaining tail is shorter.
    - No scene longer than max_duration.
    - Preserve absolute song coverage from first start to last end.
    - Preserve chronological order.
    - Prefer merging tiny scenes into following/previous scenes instead of keeping micro-cuts.

    This is intentionally conservative:
    - It does not try to re-beat-detect.
    - It only repairs the segment windows so downstream Stage1/Storyboard/LTX agree.
    """

    if not scenes:
        return []

    if min_duration <= 0:
        raise ValueError("min_duration must be > 0")

    if max_duration < min_duration:
        raise ValueError("max_duration must be >= min_duration")

    # First split scenes that exceed max_duration.
    split_scenes: list[SrtScene] = []

    for scene in scenes:
        duration = scene.duration

        if duration <= max_duration:
            split_scenes.append(scene)
            continue

        parts = max(1, int(duration // max_duration))
        if duration / parts > max_duration:
            parts += 1

        part_duration = duration / parts

        for i in range(parts):
            start = scene.start + i * part_duration
            end = scene.end if i == parts - 1 else scene.start + (i + 1) * part_duration
            split_scenes.append(
                SrtScene(
                    scene=scene.scene,
                    start=start,
                    end=end,
                    text=scene.text,
                )
            )

    # Then merge short scenes.
    merged: list[SrtScene] = []
    buffer: SrtScene | None = None

    for scene in split_scenes:
        if buffer is None:
            buffer = scene
        else:
            buffer = SrtScene(
                scene=buffer.scene,
                start=buffer.start,
                end=scene.end,
                text=buffer.text or scene.text,
            )

        if buffer.duration >= min_duration:
            # If buffer got too long, split it again into legal chunks.
            if buffer.duration > max_duration:
                merged.extend(_split_scene_to_max(buffer, max_duration))
            else:
                merged.append(buffer)
            buffer = None

    if buffer is not None:
        if not merged:
            merged.append(buffer)
        else:
            previous = merged.pop()
            combined = SrtScene(
                scene=previous.scene,
                start=previous.start,
                end=buffer.end,
                text=previous.text or buffer.text,
            )

            if combined.duration <= max_duration:
                merged.append(combined)
            else:
                merged.extend(_split_scene_to_max(combined, max_duration))

    # Final guard: if any remaining short scene exists, merge it with nearest neighbor if possible.
    stable = False
    while not stable:
        stable = True

        for i, scene in enumerate(list(merged)):
            if scene.duration >= min_duration or len(merged) == 1:
                continue

            stable = False

            if i < len(merged) - 1:
                neighbor = merged[i + 1]
                combined = SrtScene(
                    scene=scene.scene,
                    start=scene.start,
                    end=neighbor.end,
                    text=scene.text or neighbor.text,
                )
                del merged[i:i + 2]
                merged[i:i] = _split_scene_to_max(combined, max_duration)
                break

            if i > 0:
                neighbor = merged[i - 1]
                combined = SrtScene(
                    scene=neighbor.scene,
                    start=neighbor.start,
                    end=scene.end,
                    text=neighbor.text or scene.text,
                )
                del merged[i - 1:i + 1]
                merged[i - 1:i - 1] = _split_scene_to_max(combined, max_duration)
                break

    # Renumber and force contiguous boundaries from the repaired ranges.
    repaired = []
    for index, scene in enumerate(merged, start=1):
        repaired.append(
            SrtScene(
                scene=index,
                start=scene.start,
                end=scene.end,
                text=scene.text or f"Scene {index}",
            )
        )

    return repaired


def _split_scene_to_max(scene: SrtScene, max_duration: float) -> list[SrtScene]:
    if scene.duration <= max_duration:
        return [scene]

    parts = max(1, int(scene.duration // max_duration))
    if scene.duration / parts > max_duration:
        parts += 1

    part_duration = scene.duration / parts
    result = []

    for i in range(parts):
        start = scene.start + i * part_duration
        end = scene.end if i == parts - 1 else scene.start + (i + 1) * part_duration
        result.append(
            SrtScene(
                scene=scene.scene,
                start=start,
                end=end,
                text=scene.text,
            )
        )

    return result


def enforce_scene_srt_file(
    input_srt: str | Path,
    output_srt: str | Path,
    min_duration: float,
    max_duration: float,
) -> Path:
    scenes = parse_scene_srt(input_srt)
    repaired = enforce_scene_duration_constraints(
        scenes=scenes,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    return write_scene_srt(output_srt, repaired)


def validate_scene_durations(
    scenes: list[SrtScene],
    min_duration: float,
    max_duration: float,
    allow_single_short_tail: bool = True,
) -> list[str]:
    errors = []

    for index, scene in enumerate(scenes):
        is_last = index == len(scenes) - 1

        if scene.duration < min_duration:
            if not (allow_single_short_tail and is_last and len(scenes) == 1):
                errors.append(
                    f"Scene {index + 1} too short: {scene.duration:.3f}s < {min_duration:.3f}s"
                )

        if scene.duration > max_duration:
            errors.append(
                f"Scene {index + 1} too long: {scene.duration:.3f}s > {max_duration:.3f}s"
            )

    return errors
