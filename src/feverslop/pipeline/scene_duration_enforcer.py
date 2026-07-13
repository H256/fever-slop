from __future__ import annotations

from pathlib import Path

from feverslop.domain.srt import SrtBlock, SrtScene, format_srt_timestamp, parse_srt_blocks
from feverslop.ports.artifacts import ArtifactStore


def _to_srt_scene(block: SrtBlock) -> SrtScene:
    return SrtScene(scene=block.index, start=block.start, end=block.end, text=block.text)


def parse_scene_srt(path: str | Path) -> list[SrtScene]:
    """Parse SRT file to SrtScene objects using shared domain parser."""
    blocks = parse_srt_blocks(path)
    return [_to_srt_scene(block) for block in blocks]


def write_scene_srt(
    path: str | Path,
    scenes: list[SrtScene],
    *,
    artifact_store: ArtifactStore,
) -> Path:
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

    return artifact_store.write_text(path, "\n\n".join(blocks) + "\n")


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

    return renumber_scenes(
        merge_short_scenes(
            split_long_scenes(scenes, max_duration=max_duration),
            min_duration=min_duration,
            max_duration=max_duration,
        )
    )


def split_long_scenes(scenes: list[SrtScene], *, max_duration: float) -> list[SrtScene]:
    split_scenes: list[SrtScene] = []
    for scene in scenes:
        split_scenes.extend(_split_scene_to_max(scene, max_duration))
    return split_scenes


def merge_short_scenes(scenes: list[SrtScene], *, min_duration: float, max_duration: float) -> list[SrtScene]:
    merged: list[SrtScene] = []
    buffer: SrtScene | None = None

    for scene in scenes:
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
            merged.extend(_split_scene_to_max(buffer, max_duration))
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
            merged.extend(_split_scene_to_max(combined, max_duration))

    return merge_remaining_short_scenes(merged, min_duration=min_duration, max_duration=max_duration)


def merge_remaining_short_scenes(scenes: list[SrtScene], *, min_duration: float, max_duration: float) -> list[SrtScene]:
    merged = list(scenes)
    stable = False
    while not stable:
        stable = True
        for i, scene in enumerate(list(merged)):
            if scene.duration >= min_duration or len(merged) == 1:
                continue
            stable = False
            if i < len(merged) - 1:
                neighbor = merged[i + 1]
                combined = SrtScene(scene=scene.scene, start=scene.start, end=neighbor.end, text=scene.text or neighbor.text)
                del merged[i:i + 2]
                merged[i:i] = _split_scene_to_max(combined, max_duration)
                break
            if i > 0:
                neighbor = merged[i - 1]
                combined = SrtScene(scene=neighbor.scene, start=neighbor.start, end=scene.end, text=neighbor.text or scene.text)
                del merged[i - 1:i + 1]
                merged[i - 1:i - 1] = _split_scene_to_max(combined, max_duration)
                break
    return merged


def renumber_scenes(scenes: list[SrtScene]) -> list[SrtScene]:
    return [
        SrtScene(
            scene=index,
            start=scene.start,
            end=scene.end,
            text=scene.text or f"Scene {index}",
        )
        for index, scene in enumerate(scenes, start=1)
    ]


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
    *,
    artifact_store: ArtifactStore,
) -> Path:
    scenes = parse_scene_srt(input_srt)
    repaired = enforce_scene_duration_constraints(
        scenes=scenes,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    return write_scene_srt(output_srt, repaired, artifact_store=artifact_store)


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
