from __future__ import annotations

from autoprompter.application.pipeline_context import GenerateRenderPlanContext
from autoprompter.audio.beat_analysis import BeatSceneDurationGenerator
from autoprompter.pipeline.prompt_relay_builder import build_scene_prompt_relay
from autoprompter.pipeline.scene_duration_enforcer import (
    enforce_scene_srt_file,
    parse_scene_srt,
    validate_scene_durations,
)
from autoprompter.pipeline.stage1_segment_builder import build_stage1_segment_json


class SceneTimelinePipeline:
    """Application service boundary for scene SRT, stage1 mapping, and relay skeletons."""

    required_keys = {"config", "video_settings", "timeline_json", "beat_json"}
    produced_keys = {
        "scene_srt",
        "stage1_segments_json",
        "stage1_segments",
        "ltx_prompt_relay_json",
        "repaired_scenes",
    }

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        config = context["config"]
        video_settings = context["video_settings"]
        timeline_json = context["timeline_json"]
        beat_json = context["beat_json"]
        scene_srt_raw = context["scene_srt_raw"]
        scene_srt = context["scene_srt"]
        stage1_segments_json = context["stage1_segments_json"]
        ltx_prompt_relay_json = context["ltx_prompt_relay_json"]
        artifact_store = context["artifact_store"]
        log_step = context["log_step"]
        log_file = context["log_file"]
        console = context["console"]

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
        repaired_scenes = parse_scene_srt(scene_srt)
        duration_errors = validate_scene_durations(
            repaired_scenes,
            min_duration=scene_cfg.min_duration,
            max_duration=scene_cfg.max_duration,
        )
        if duration_errors:
            raise ValueError(
                "Scene duration constraints failed after repair:\n"
                + "\n".join(duration_errors)
            )
        shortest_scene = min((scene.duration for scene in repaired_scenes), default=0.0)
        longest_scene = max((scene.duration for scene in repaired_scenes), default=0.0)
        console.print(
            f"[green]✓[/green] Scene duration range: "
            f"[yellow]{shortest_scene:.2f}s[/yellow].."
            f"[yellow]{longest_scene:.2f}s[/yellow] "
            f"from [yellow]{len(repaired_scenes)}[/yellow] scenes"
        )

        log_step("5. Stage 1 Segment Mapping")
        build_stage1_segment_json(
            scene_srt_file=scene_srt,
            vocal_timeline_json=timeline_json,
            output_json_file=stage1_segments_json,
            artifact_store=artifact_store,
        )
        log_file("Stage 1 Segments JSON", stage1_segments_json)
        stage1_segments = artifact_store.read_json(stage1_segments_json)
        type_counts: dict[str, int] = {}
        for seg in stage1_segments:
            type_counts[seg["type"]] = type_counts.get(seg["type"], 0) + 1
        console.print(
            f"[green]✓[/green] Stage 1 segments: [yellow]{len(stage1_segments)}[/yellow] "
            f"{type_counts}"
        )

        log_step("6. LTX Prompt Relay Skeleton")
        build_scene_prompt_relay(
            scene_srt_file=scene_srt,
            vocal_timeline_json=timeline_json,
            output_json_file=ltx_prompt_relay_json,
            video_settings=video_settings,
            artifact_store=artifact_store,
        )
        log_file("LTX Prompt Relay JSON", ltx_prompt_relay_json)

        context.update(
            {
                "repaired_scenes": repaired_scenes,
                "stage1_segments": stage1_segments,
            }
        )
        return context
