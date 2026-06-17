"""
Patch this into main.py after BeatSceneDurationGenerator.generate_from_json_file(...)
and before build_stage1_segment_json(...).

Required import:

from scene_duration_enforcer import (
    enforce_scene_srt_file,
    parse_scene_srt,
    validate_scene_durations,
)

Recommended path variable:

scene_srt_raw = timeline_dir / f"scenes_{song_id}_raw.srt"
scene_srt = timeline_dir / f"scenes_{song_id}.srt"

Use scene_srt_raw as the direct generator output, then repair into scene_srt.
"""

# ------------------------------------------------------------------
log_step("4. Beat-Aligned Scene SRT")

scene_cfg = config.scene_generation
scene_generator = BeatSceneDurationGenerator(
    min_duration=scene_cfg.min_duration,
    max_duration=scene_cfg.max_duration,
    bias=scene_cfg.bias,
    duration_preset=scene_cfg.duration_preset,
    seed=scene_cfg.seed,
)

scene_generator.generate_from_json_file(
    beat_json_path=beat_json,
    output_srt_path=scene_srt_raw,
)
log_file("Raw Scene SRT", scene_srt_raw)

enforce_scene_srt_file(
    input_srt=scene_srt_raw,
    output_srt=scene_srt,
    min_duration=scene_cfg.min_duration,
    max_duration=scene_cfg.max_duration,
)
log_file("Repaired Scene SRT", scene_srt)

duration_errors = validate_scene_durations(
    parse_scene_srt(scene_srt),
    min_duration=scene_cfg.min_duration,
    max_duration=scene_cfg.max_duration,
)

if duration_errors:
    raise ValueError(
        "Scene duration constraints failed after repair:\n"
        + "\n".join(duration_errors)
    )
