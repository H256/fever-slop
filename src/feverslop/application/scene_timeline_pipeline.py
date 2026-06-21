from __future__ import annotations

from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext


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

    def __init__(
        self,
        *,
        scene_generator_factory: Callable[[Any], Any],
        enforce_scene_srt_file: Callable[..., Any],
        parse_scene_srt: Callable[..., list[Any]],
        validate_scene_durations: Callable[..., list[str]],
        build_stage1_segment_json: Callable[..., Any],
        build_scene_prompt_relay: Callable[..., Any],
    ):
        self.scene_generator_factory = scene_generator_factory
        self.enforce_scene_srt_file = enforce_scene_srt_file
        self.parse_scene_srt = parse_scene_srt
        self.validate_scene_durations = validate_scene_durations
        self.build_stage1_segment_json = build_stage1_segment_json
        self.build_scene_prompt_relay = build_scene_prompt_relay

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
        scene_generator = self.scene_generator_factory(scene_cfg)
        scene_generator.generate_from_json_file(
            beat_json_path=beat_json,
            output_srt_path=scene_srt_raw,
        )
        log_file("Raw Scene SRT", scene_srt_raw)
        self.enforce_scene_srt_file(
            input_srt=scene_srt_raw,
            output_srt=scene_srt,
            min_duration=scene_cfg.min_duration,
            max_duration=scene_cfg.max_duration,
            artifact_store=artifact_store,
        )
        log_file("Repaired Scene SRT", scene_srt)
        repaired_scenes = self.parse_scene_srt(scene_srt)
        duration_errors = self.validate_scene_durations(
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
            f"[green]OK[/green] Scene duration range: "
            f"[yellow]{shortest_scene:.2f}s[/yellow].."
            f"[yellow]{longest_scene:.2f}s[/yellow] "
            f"from [yellow]{len(repaired_scenes)}[/yellow] scenes"
        )

        log_step("5. Stage 1 Segment Mapping")
        self.build_stage1_segment_json(
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
            f"[green]OK[/green] Stage 1 segments: [yellow]{len(stage1_segments)}[/yellow] "
            f"{type_counts}"
        )

        log_step("6. LTX Prompt Relay Skeleton")
        self.build_scene_prompt_relay(
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
