from __future__ import annotations

from pathlib import Path

from feverslop.config.video_settings import VideoSettings
from feverslop.domain.performance_timeline import lean_performance_projection, project_performance
from feverslop.domain.srt import parse_srt_blocks
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.reporting import Reporter
from feverslop.utils.sub_step_progress import SubStepProgress


def parse_scene_dicts(srt_file: str | Path) -> list[dict]:
    """Parse an SRT file into a list of plain dicts.

    Returns dicts with keys: scene, start, end, label.
    Use ``scene_duration_enforcer.parse_srt_scenes`` (returns ``list[SrtScene]``)
    when you need the typed dataclass with ``.duration`` property.
    """
    blocks = parse_srt_blocks(srt_file)
    return [
        {
            "scene": block.index,
            "start": block.start,
            "end": block.end,
            "label": block.text or f"SCENE {block.index}",
        }
        for block in blocks
    ]


def overlap(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> tuple[float, float] | None:
    start = max(a_start, b_start)
    end = min(a_end, b_end)

    if end <= start:
        return None

    return start, end


def lyrics_for_time_range(
    lyrics: str,
    source_start: float,
    source_end: float,
    range_start: float,
    range_end: float,
    word_timestamps: list[dict] | tuple[dict, ...] | None = None,
) -> str:
    """Return words assigned to a subrange using Whisper timestamps when available."""
    words_with_timestamps = word_timestamps or ()
    if words_with_timestamps:
        selected = []
        for item in words_with_timestamps:
            try:
                word_start = float(item["start"])
                word_end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            midpoint = (word_start + word_end) / 2
            if range_start <= midpoint < range_end:
                word = str(item.get("word", "")).strip()
                if word:
                    selected.append(word)
        return " ".join(selected)

    # Legacy timeline files do not contain word timestamps.
    words = str(lyrics or "").split()
    source_duration = float(source_end) - float(source_start)
    if not words or source_duration <= 0 or range_end <= range_start:
        return ""

    start_ratio = max(0.0, min(1.0, (float(range_start) - float(source_start)) / source_duration))
    end_ratio = max(0.0, min(1.0, (float(range_end) - float(source_start)) / source_duration))
    start_index = min(len(words), max(0, round(start_ratio * len(words))))
    end_index = min(len(words), max(start_index, round(end_ratio * len(words))))
    return " ".join(words[start_index:end_index])


def build_scene_prompt_relay(
    scene_srt_file: str | Path,
    vocal_timeline_json: str | Path,
    output_json_file: str | Path,
    video_settings: VideoSettings,
    singing_prompt_template: str = (
        "same scene, character sings with expressive lip sync, performing the lyrics: {lyrics}"
    ),
    instrumental_prompt: str = (
        "same scene, instrumental section, character is not singing, no lip movement"
    ),
    min_segment_duration: float = 0.25,
    *,
    artifact_store: ArtifactStore,
    reporter: Reporter | None = None,
) -> Path:
    scenes = parse_scene_dicts(scene_srt_file)
    timeline = artifact_store.read_json(vocal_timeline_json)

    result = []

    progress = SubStepProgress(reporter, "Performance relay projection", len(scenes))
    progress.update(0, force=True)
    for current, scene in enumerate(scenes, start=1):
        scene_start = float(scene["start"])
        scene_end = float(scene["end"])
        scene_duration = scene_end - scene_start

        prompt_relay = []
        performance_intervals = []
        # Real pauses remain explicit regardless of the legacy duration hint.
        for phase in lean_performance_projection(project_performance(timeline, scene_start, scene_end)):
            rel_start = phase["start"] - scene_start
            rel_end = phase["end"] - scene_start
            frame_start = video_settings.seconds_to_frame(rel_start)
            frame_end = video_settings.seconds_to_frame(rel_end)
            performance_intervals.append({**phase, "start_seconds": rel_start, "end_seconds": rel_end})
            if frame_end <= frame_start:
                continue
            if phase["state"] == "singing":
                prompt = (singing_prompt_template.format(lyrics=phase["lyrics"])
                          if phase["lyrics"] else "same scene, sustained vocal continues with synchronized performance")
            else:
                prompt = instrumental_prompt
            prompt_relay.append({
                **phase,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "start_seconds": rel_start,
                "end_seconds": rel_end,
                "prompt": prompt,
            })

        scene_data = {
            "scene": scene["scene"],
            "scene_label": scene["label"],
            "abs_start_seconds": round(scene_start, 2),
            "abs_end_seconds": round(scene_end, 2),
            "duration_seconds": round(scene_duration, 2),
            "duration_frames": video_settings.seconds_to_frame(scene_duration),
            "fps": video_settings.fps,
            "width": video_settings.width,
            "height": video_settings.height,
            "prompt_relay": prompt_relay,
            "performance_intervals": performance_intervals,
        }

        result.append(scene_data)
        progress.update(current)

    return artifact_store.write_json(output_json_file, result)
