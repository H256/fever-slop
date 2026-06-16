from __future__ import annotations

from pathlib import Path
import json

from prompt_relay_builder import parse_scene_srt, overlap


def build_stage1_segment_json(
    scene_srt_file: str | Path,
    vocal_timeline_json: str | Path,
    output_json_file: str | Path,
    min_vocal_ratio_for_vocals: float = 0.65,
    min_vocal_ratio_for_mixed: float = 0.10,
) -> Path:
    scenes = parse_scene_srt(scene_srt_file)
    timeline = json.loads(Path(vocal_timeline_json).read_text(encoding="utf-8"))

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
                lyrics.append(seg_lyrics.strip())

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

    output_json_file = Path(output_json_file)
    output_json_file.parent.mkdir(parents=True, exist_ok=True)

    with output_json_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return output_json_file