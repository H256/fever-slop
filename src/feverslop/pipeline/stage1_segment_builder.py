from __future__ import annotations

from pathlib import Path

from feverslop.domain.performance_timeline import lean_performance_projection, project_performance
from feverslop.pipeline.prompt_relay_builder import parse_scene_dicts
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.reporting import Reporter
from feverslop.utils.sub_step_progress import SubStepProgress


def build_stage1_segment_json(
    scene_srt_file: str | Path,
    vocal_timeline_json: str | Path,
    output_json_file: str | Path,
    min_vocal_ratio_for_vocals: float = 0.65,
    min_vocal_ratio_for_mixed: float = 0.10,
    *,
    artifact_store: ArtifactStore,
    reporter: Reporter | None = None,
) -> Path:
    scenes = parse_scene_dicts(scene_srt_file)
    timeline = artifact_store.read_json(vocal_timeline_json)

    result = []

    progress = SubStepProgress(reporter, "Stage 1 performance mapping", len(scenes))
    progress.update(0, force=True)
    for current, scene in enumerate(scenes, start=1):
        scene_start = float(scene["start"])
        scene_end = float(scene["end"])
        scene_duration = max(scene_end - scene_start, 1e-6)

        performance = lean_performance_projection(project_performance(timeline, scene_start, scene_end))
        vocal_time = sum(phase["end"] - phase["start"] for phase in performance if phase["state"] == "singing")
        lyrics = [phase["lyrics"] for phase in performance if phase["lyrics"]]

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
            "performance_intervals": performance,
            "word_timestamps": [word for phase in performance for word in phase["word_timestamps"]],
            "reason_codes": list(dict.fromkeys(reason for phase in performance for reason in phase["reason_codes"])),
            "performance_conflicts": [conflict for phase in performance for conflict in phase["performance_conflicts"]],
        }

        if lyrics:
            item["lyrics"] = " ".join(lyrics)

        result.append(item)
        progress.update(current)

    return artifact_store.write_json(output_json_file, result)
