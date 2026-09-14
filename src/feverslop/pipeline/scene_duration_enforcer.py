from __future__ import annotations

import math
from pathlib import Path

from feverslop.domain.srt import (
    SrtBlock,
    SrtScene as _SrtScene,
    format_srt_timestamp,
    parse_srt_blocks,
)
from feverslop.ports.artifacts import ArtifactStore

def _to_srt_scene(block: SrtBlock) -> _SrtScene:
    return _SrtScene(scene=block.index, start=block.start, end=block.end, text=block.text)


def parse_srt_scenes(path: str | Path) -> list[_SrtScene]:
    """Parse SRT file to SrtScene objects using shared domain parser."""
    blocks = parse_srt_blocks(path)
    return [_to_srt_scene(block) for block in blocks]


def write_scene_srt(
    path: str | Path,
    scenes: list[_SrtScene],
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
            ]),
        )

    return artifact_store.write_text(path, "\n\n".join(blocks) + "\n")


def enforce_scene_duration_constraints(
    scenes: list[_SrtScene],
    min_duration: float,
    max_duration: float,
) -> list[_SrtScene]:
    """Repairs scene durations after beat-based SRT generation.

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

    merged = renumber_scenes(
        merge_short_scenes(
            split_long_scenes(scenes, max_duration=max_duration),
            min_duration=min_duration,
            max_duration=max_duration,
        ),
    )
    return _repartition_feasible(merged, min_duration, max_duration)


def _repartition_feasible(
    scenes: list[_SrtScene], min_duration: float, max_duration: float
) -> list[_SrtScene]:
    """Best-effort repair for feasible bands that the merge/split pass still
    leaves with scenes below ``min_duration`` (the tight ``max < 2*min`` band,
    where the merge over-splits before the repair pass can recover).

    Only runs when the total coverage can actually be partitioned into scenes
    that all respect ``min_duration`` (a feasible band) AND the current output
    still contains a short scene. In that case it re-distributes the same
    covered window into a legal scene count, so no scene is short or long.
    Infeasible bands (e.g. a clamped ``min == max`` render budget) are left
    untouched: no split can satisfy ``min_duration`` there, so the caller is
    expected to validate with feasibility in mind.
    """
    if not scenes:
        return scenes
    start = scenes[0].start
    end = scenes[-1].end
    total = end - start
    if total <= min_duration:
        # A single scene: the whole window is either legal or unavoidably
        # short (infeasible). Nothing to repartition.
        return scenes
    if not any(scene.end - scene.start < min_duration - 1e-9 for scene in scenes):
        return scenes
    # A legal scene count k satisfies k*min <= total <= k*max.
    k_min = math.ceil(total / max_duration - 1e-9)
    k_max = int(total / min_duration + 1e-9)
    if k_max < 1 or k_min > k_max:
        # Infeasible band: no count makes every scene >= min. Leave as-is.
        return scenes
    # k_min is the smallest legal count, so its equal parts fall within
    # [min_duration, max_duration]; this yields a fully legal output with the
    # fewest re-cuts (longest scenes), matching the merge preference.
    return _split_window_to_parts(start, end, k_min, scenes[0].scene)


def _split_window_to_parts(
    start: float, end: float, parts: int, scene_number: int
) -> list[_SrtScene]:
    """Split ``[start, end]`` into ``parts`` equal pieces (coverage-preserving)."""
    if parts <= 1:
        return [_SrtScene(scene=scene_number, start=start, end=end, text="")]
    part_duration = (end - start) / parts
    result: list[_SrtScene] = []
    for i in range(parts):
        piece_start = start + i * part_duration
        piece_end = end if i == parts - 1 else start + (i + 1) * part_duration
        result.append(_SrtScene(scene=scene_number, start=piece_start, end=piece_end, text=""))
    return renumber_scenes(result)


def split_long_scenes(scenes: list[_SrtScene], *, max_duration: float) -> list[_SrtScene]:
    split_scenes: list[_SrtScene] = []
    for scene in scenes:
        split_scenes.extend(_split_scene_to_max(scene, max_duration))
    return split_scenes


def merge_short_scenes(scenes: list[_SrtScene], *, min_duration: float, max_duration: float) -> list[_SrtScene]:
    merged: list[_SrtScene] = []
    buffer: _SrtScene | None = None

    for scene in scenes:
        if buffer is None:
            buffer = scene
        else:
            buffer = _SrtScene(
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
            combined = _SrtScene(
                scene=previous.scene,
                start=previous.start,
                end=buffer.end,
                text=previous.text or buffer.text,
            )
            merged.extend(_split_scene_to_max(combined, max_duration))

    return merge_remaining_short_scenes(merged, min_duration=min_duration, max_duration=max_duration)


def merge_remaining_short_scenes(scenes: list[_SrtScene], *, min_duration: float, max_duration: float) -> list[_SrtScene]:
    if min_duration <= 0:
        raise ValueError("min_duration must be > 0")
    if max_duration < min_duration:
        raise ValueError("max_duration must be >= min_duration")
    merged = list(scenes)
    prev_short_count = len(scenes) + 1  # ensure first iteration runs
    for _ in range(100):  # hard safety cap
        stable = True
        for i, scene in enumerate(list(merged)):
            if scene.duration >= min_duration or len(merged) == 1:
                continue
            stable = False
            if i < len(merged) - 1:
                neighbor = merged[i + 1]
                combined = _SrtScene(scene=scene.scene, start=scene.start, end=neighbor.end, text=scene.text or neighbor.text)
                del merged[i:i + 2]
                merged[i:i] = _split_scene_to_max(combined, max_duration)
                break
            if i > 0:
                neighbor = merged[i - 1]
                combined = _SrtScene(scene=neighbor.scene, start=neighbor.start, end=scene.end, text=neighbor.text or scene.text)
                del merged[i - 1:i + 1]
                merged[i - 1:i - 1] = _split_scene_to_max(combined, max_duration)
                break
        if stable:
            break
        short_count = sum(1 for s in merged if s.duration < min_duration)
        if short_count >= prev_short_count:
            # No progress — merging+splitting is cycling without helping
            break
        prev_short_count = short_count
    return merged


def renumber_scenes(scenes: list[_SrtScene]) -> list[_SrtScene]:
    return [
        _SrtScene(
            scene=index,
            start=scene.start,
            end=scene.end,
            text=scene.text or f"Scene {index}",
        )
        for index, scene in enumerate(scenes, start=1)
    ]


def _split_scene_to_max(scene: _SrtScene, max_duration: float) -> list[_SrtScene]:
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
            _SrtScene(
                scene=scene.scene,
                start=start,
                end=end,
                text=scene.text,
            ),
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
    scenes = parse_srt_scenes(input_srt)
    repaired = enforce_scene_duration_constraints(
        scenes=scenes,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    return write_scene_srt(output_srt, repaired, artifact_store=artifact_store)


def _is_duration_feasible(total: float, min_duration: float, max_duration: float, epsilon: float) -> bool:
    """Whether ``total`` can be split into >= 1 whole segments each within
    ``[min_duration, max_duration]``.

    Such a split exists iff there is an integer ``k >= 1`` with
    ``k * min_duration <= total <= k * max_duration``, i.e.
    ``ceil(total / max_duration) <= floor(total / min_duration)``.

    Used to tell apart a legitimately short scene (the enforcer's best
    effort when the constraints cannot all be met, e.g. a clamped render
    budget that leaves no scene at or above ``min_duration``) from a real
    enforcer violation.
    """
    if total <= 0 or min_duration <= 0:
        return False
    k_min = math.ceil(total / max_duration - epsilon)
    k_max = int(total / min_duration + epsilon)
    return k_min >= 1 and k_min <= k_max


def validate_scene_durations(
    scenes: list[_SrtScene],
    min_duration: float,
    max_duration: float,
    allow_single_short_tail: bool = True,
) -> list[str]:
    errors = []
    # Tolerance for IEEE-754 subtraction artifacts (nanoseconds, not meaningful ms)
    epsilon = 1e-9

    if scenes:
        total = scenes[-1].end - scenes[0].start
        min_enforceable = _is_duration_feasible(total, min_duration, max_duration, epsilon)
        if not allow_single_short_tail:
            min_enforceable = True
    else:
        min_enforceable = False

    for index, scene in enumerate(scenes):
        duration = scene.end - scene.start

        if duration < min_duration - epsilon and min_enforceable:
            errors.append(
                f"Scene {index + 1} too short: {duration:.3f}s < {min_duration:.3f}s",
            )

        if duration > max_duration + epsilon:
            errors.append(
                f"Scene {index + 1} too long: {duration:.3f}s > {max_duration:.3f}s",
            )

    return errors
