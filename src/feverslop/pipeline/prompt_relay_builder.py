from __future__ import annotations

from pathlib import Path

from feverslop.domain.srt import parse_srt_blocks
from feverslop.ports.artifacts import ArtifactStore
from feverslop.config.video_settings import VideoSettings


def parse_scene_dicts(srt_file: str | Path) -> list[dict]:
    """Parse an SRT file into a list of plain dicts.

    Returns dicts with keys: scene, start, end, label.
    Use ``scene_duration_enforcer.parse_scene_srt`` (returns ``list[SrtScene]``)
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


def parse_scene_srt(srt_file: str | Path) -> list[dict]:
    """Parse SRT file to scene dicts.

    .. deprecated::
        Use :func:`parse_scene_dicts` instead to disambiguate from
        ``scene_duration_enforcer.parse_scene_srt`` which returns
        ``list[SrtScene]``.
    """
    import warnings

    warnings.warn(
        "parse_scene_srt is deprecated, use parse_scene_dicts instead. "
        "This function returns list[dict] and is distinct from "
        "scene_duration_enforcer.parse_scene_srt which returns list[SrtScene].",
        FutureWarning,
        stacklevel=2,
    )
    return parse_scene_dicts(srt_file)


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
    lyrics_words = str(lyrics or "").split()
    timestamp_word_count = sum(
        bool(str(item.get("word", "")).strip())
        for item in words_with_timestamps
        if isinstance(item, dict)
    )
    if words_with_timestamps and timestamp_word_count == len(lyrics_words):
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
    words = lyrics_words
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
) -> Path:
    scenes = parse_scene_dicts(scene_srt_file)
    timeline = artifact_store.read_json(vocal_timeline_json)

    result = []

    for scene in scenes:
        scene_start = float(scene["start"])
        scene_end = float(scene["end"])
        scene_duration = scene_end - scene_start

        cuts = {scene_start, scene_end}
        relevant_vocals = []

        for seg in timeline:
            seg_type = seg.get("type") or seg.get("kind")
            lyrics = seg.get("lyrics") or seg.get("text") or ""

            if seg_type != "vocals" or not lyrics.strip():
                continue

            ov = overlap(
                scene_start,
                scene_end,
                float(seg["start"]),
                float(seg["end"]),
            )

            if ov is None:
                continue

            ov_start, ov_end = ov
            cuts.add(ov_start)
            cuts.add(ov_end)

            relevant_vocals.append({
                "start": ov_start,
                "end": ov_end,
                "source_start": float(seg["start"]),
                "source_end": float(seg["end"]),
                "word_timestamps": seg.get("word_timestamps") or (),
                "lyrics": lyrics.strip(),
            })

        prompt_relay = []
        sorted_cuts = sorted(cuts)

        for abs_start, abs_end in zip(sorted_cuts, sorted_cuts[1:]):
            if abs_end - abs_start < min_segment_duration:
                continue

            lyrics_here = []

            for vocal in relevant_vocals:
                if overlap(abs_start, abs_end, vocal["start"], vocal["end"]):
                    lyric_text = lyrics_for_time_range(
                        vocal["lyrics"],
                        vocal["source_start"],
                        vocal["source_end"],
                        abs_start,
                        abs_end,
                        vocal["word_timestamps"],
                    )
                    if lyric_text:
                        lyrics_here.append(lyric_text)

            rel_start = abs_start - scene_start
            rel_end = abs_end - scene_start

            frame_start = video_settings.seconds_to_frame(rel_start)
            frame_end = video_settings.seconds_to_frame(rel_end)

            if frame_end <= frame_start:
                continue

            if lyrics_here:
                state = "singing"
                prompt = singing_prompt_template.format(
                    lyrics=" ".join(lyrics_here).strip()
                )
            else:
                state = "instrumental"
                prompt = instrumental_prompt

            prompt_relay.append({
                "frame_start": frame_start,
                "frame_end": frame_end,
                "start_seconds": round(rel_start, 2),
                "end_seconds": round(rel_end, 2),
                "state": state,
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
        }

        result.append(scene_data)

    return artifact_store.write_json(output_json_file, result)
