from __future__ import annotations

from pathlib import Path

from feverslop.ports.artifacts import ArtifactStore
from feverslop.pipeline.prompt_relay_builder import lyrics_for_time_range, parse_scene_dicts, overlap


def build_stage1_segment_json(
    scene_srt_file: str | Path,
    vocal_timeline_json: str | Path,
    output_json_file: str | Path,
    min_vocal_ratio_for_vocals: float = 0.65,
    min_vocal_ratio_for_mixed: float = 0.10,
    *,
    artifact_store: ArtifactStore,
) -> Path:
    scenes = parse_scene_dicts(scene_srt_file)
    timeline = artifact_store.read_json(vocal_timeline_json)

    result = []

    for scene in scenes:
        scene_start = float(scene["start"])
        scene_end = float(scene["end"])
        scene_duration = max(scene_end - scene_start, 1e-6)

        vocal_time = 0.0
        lyrics = []

        for seg in timeline:
            seg_type = seg.get("type") or seg.get("kind")
            seg_lyrics = seg.get("lyrics") or seg.get("text") or ""

            ov = overlap(
                scene_start,
                scene_end,
                float(seg["start"]),
                float(seg["end"]),
            )

            if ov is None:
                continue

            ov_start, ov_end = ov
            ov_duration = ov_end - ov_start

            if seg_type == "vocals" and seg_lyrics.strip():
                vocal_time += ov_duration
                lyric_text = lyrics_for_time_range(
                    seg_lyrics,
                    float(seg["start"]),
                    float(seg["end"]),
                    ov_start,
                    ov_end,
                    seg.get("word_timestamps") or (),
                )
                if lyric_text:
                    lyrics.append(lyric_text)

        vocal_ratio = vocal_time / scene_duration

        if vocal_ratio >= min_vocal_ratio_for_vocals:
            segment_type = "vocals"
        elif vocal_ratio >= min_vocal_ratio_for_mixed:
            segment_type = "mixed"
        else:
            segment_type = "instrumental"

        item = {
            "segment_id": f"segment_{scene['scene']:03}",
            "scene": scene["scene"],
            "start": round(scene_start, 2),
            "end": round(scene_end, 2),
            "duration": round(scene_duration, 2),
            "type": segment_type,
        }

        if lyrics:
            item["lyrics"] = " ".join(lyrics)

        result.append(item)

    return artifact_store.write_json(output_json_file, result)
