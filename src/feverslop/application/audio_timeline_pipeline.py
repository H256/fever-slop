from __future__ import annotations

from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.ports.generate_pipeline import (
    BeatImpactAnalyzerFactory,
    LyricAlignerFactory,
    StemSeparatorFactory,
    VocalTimelineAnalyzerFactory,
)


class AudioTimelinePipeline:
    """Application service boundary for stem, vocal timeline, and beat analysis."""

    required_keys = {"config", "paths", "song_id", "video_settings"}
    produced_keys = {"stem_files", "timeline", "timeline_json", "beat_json", "beat_data"}

    def __init__(
        self,
        *,
        separator_factory: StemSeparatorFactory,
        vocal_analyzer_factory: VocalTimelineAnalyzerFactory,
        beat_analyzer_factory: BeatImpactAnalyzerFactory,
        lyric_aligner_factory: LyricAlignerFactory | None = None,
        normalize_empty_vocals: Callable[[list[Any]], list[Any]],
        merge_same_kind_segments: Callable[..., list[Any]],
        save_timeline_json: Callable[..., Any],
    ):
        self.separator_factory = separator_factory
        self.vocal_analyzer_factory = vocal_analyzer_factory
        self.beat_analyzer_factory = beat_analyzer_factory
        self.lyric_aligner_factory = lyric_aligner_factory
        self.normalize_empty_vocals = normalize_empty_vocals
        self.merge_same_kind_segments = merge_same_kind_segments
        self.save_timeline_json = save_timeline_json

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        config = context["config"]
        paths = context["paths"]
        timeline_json = context["timeline_json"]
        beat_json = context["beat_json"]
        log_step = context["log_step"]
        log_file = context["log_file"]
        run_spinner = context["run_spinner"]
        reporter = context["reporter"]

        log_step("1. Demucs Stem Separation")
        separator = self.separator_factory(config)
        files = run_spinner(
            "Separating audio into vocals/drums/bass/other...",
            lambda: separator.separate(config.input_audio, paths.stems_dir),
        )

        reporter.table(
            "Generated Stems",
            ["Stem", "Path"],
            [[stem_name, str(files[stem_name])] for stem_name in ("vocals", "drums", "bass", "other")],
        )

        log_step("2. Vocal Timeline Analysis")
        vocal_cfg = config.vocal_detection
        analyzer = self.vocal_analyzer_factory(config)
        timeline = run_spinner(
            "Detecting vocal activity and transcribing lyrics...",
            lambda: analyzer.analyze(files["vocals"]),
        )
        timeline = self.normalize_empty_vocals(timeline)
        timeline = self.merge_same_kind_segments(timeline, merge_gap=vocal_cfg.merge_gap)
        reference_lyrics = str(getattr(config, "lyrics", "") or "").strip()
        if reference_lyrics and self.lyric_aligner_factory is not None:
            aligner = self.lyric_aligner_factory(context)
            timeline = run_spinner(
                "Correcting Whisper lyrics against project lyrics...",
                lambda: aligner.align(timeline, reference_lyrics),
            )
        self.save_timeline_json(timeline, timeline_json)
        log_file("Timeline JSON", timeline_json)

        vocal_count = sum(1 for seg in timeline if seg.kind == "vocals")
        instrumental_count = sum(1 for seg in timeline if seg.kind == "instrumental")
        reporter.message(
            f"[green]OK[/green] Timeline segments: "
            f"[yellow]{len(timeline)}[/yellow] total, "
            f"[yellow]{vocal_count}[/yellow] vocals, "
            f"[yellow]{instrumental_count}[/yellow] instrumental"
        )

        log_step("3. Beat / Impact Analysis")
        beat_analyzer = self.beat_analyzer_factory()
        run_spinner(
            "Analyzing beats and impact values...",
            lambda: beat_analyzer.analyze_to_json_file(
                final_mix_path=config.input_audio,
                output_json_path=beat_json,
                drums_path=files["drums"],
                bass_path=files["bass"],
                vocals_path=files["vocals"],
                other_path=files["other"],
            ),
        )
        log_file("Beat Data JSON", beat_json)
        beat_data = context["artifact_store"].read_json(beat_json)
        reporter.message(
            f"[green]OK[/green] BPM: [yellow]{beat_data.get('bpm')}[/yellow], "
            f"beats: [yellow]{len(beat_data.get('beats', []))}[/yellow], "
            f"source: [yellow]{beat_data.get('source_used_for_beats')}[/yellow]"
        )

        context.update(
            {
                "stem_files": files,
                "timeline": timeline,
                "beat_data": beat_data,
            }
        )
        return context
