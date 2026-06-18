from __future__ import annotations

from pathlib import Path
import json
import re

from adapters.local_artifacts import JsonArtifactStore
from ports.artifacts import ArtifactStore
from video_settings import VideoSettings


def parse_srt_time(value: str) -> float:
    h, m, rest = value.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_scene_srt(srt_file: str | Path) -> list[dict]:
    text = Path(srt_file).read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\s*\n", text)

    scenes = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        index = int(lines[0])
        start_raw, end_raw = [x.strip() for x in lines[1].split("-->")]
        label = lines[2] if len(lines) >= 3 else f"SCENE {index}"

        scenes.append({
            "scene": index,
            "start": parse_srt_time(start_raw),
            "end": parse_srt_time(end_raw),
            "label": label,
        })

    return scenes


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
    artifact_store: ArtifactStore | None = None,
) -> Path:
    artifact_store = artifact_store or JsonArtifactStore()
    scenes = parse_scene_srt(scene_srt_file)
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
                    lyrics_here.append(vocal["lyrics"])

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
