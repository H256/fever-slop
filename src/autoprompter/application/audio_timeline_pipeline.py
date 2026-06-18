from __future__ import annotations

from typing import Any

from autoprompter.application.pipeline_context import GenerateRenderPlanContext
from autoprompter.ports.generate_pipeline import (
    BeatImpactAnalyzerFactory,
    StemSeparatorFactory,
    VocalTimelineAnalyzerFactory,
)
from rich.table import Table

from autoprompter.audio.beat_analysis import BeatImpactAnalyzer
from autoprompter.audio.demucs_separator import DemucsSeparator
from autoprompter.pipeline.utils import save_timeline_json
from autoprompter.audio.vocal_timeline_analyzer import (
    VocalTimelineAnalyzer,
    merge_same_kind_segments,
    normalize_empty_vocals,
)


class AudioTimelinePipeline:
    """Application service boundary for stem, vocal timeline, and beat analysis."""

    required_keys = {"config", "paths", "song_id", "video_settings"}
    produced_keys = {"stem_files", "timeline", "timeline_json", "beat_json", "beat_data"}

    def __init__(
        self,
        *,
        separator_factory: StemSeparatorFactory | None = None,
        vocal_analyzer_factory: VocalTimelineAnalyzerFactory | None = None,
        beat_analyzer_factory: BeatImpactAnalyzerFactory | None = None,
    ):
        self.separator_factory = separator_factory or self._default_separator
        self.vocal_analyzer_factory = vocal_analyzer_factory or self._default_vocal_analyzer
        self.beat_analyzer_factory = beat_analyzer_factory or BeatImpactAnalyzer

    def _default_separator(self, config: Any):
        return DemucsSeparator(model_name=config.audio.demucs_model)

    def _default_vocal_analyzer(self, config: Any):
        vocal_cfg = config.vocal_detection
        return VocalTimelineAnalyzer(
            whisper_model=config.audio.whisper_model,
            language=config.audio.language,
            merge_gap=vocal_cfg.merge_gap,
            min_vocal_duration=vocal_cfg.min_vocal_duration,
            min_silence_duration=vocal_cfg.min_silence_duration,
            rms_low_percentile=vocal_cfg.rms_low_percentile,
            rms_high_percentile=vocal_cfg.rms_high_percentile,
            rms_ratio=vocal_cfg.rms_ratio,
            smooth_frames=vocal_cfg.smooth_frames,
        )

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
        console = context["console"]

        log_step("1. Demucs Stem Separation")
        separator = self.separator_factory(config)
        files = run_spinner(
            "Separating audio into vocals/drums/bass/other...",
            lambda: separator.separate(config.input_audio, paths.stems_dir),
        )

        stem_table = Table(title="Generated Stems")
        stem_table.add_column("Stem", style="bold")
        stem_table.add_column("Path", style="cyan")
        for stem_name in ("vocals", "drums", "bass", "other"):
            stem_table.add_row(stem_name, str(files[stem_name]))
        console.print(stem_table)

        log_step("2. Vocal Timeline Analysis")
        vocal_cfg = config.vocal_detection
        analyzer = self.vocal_analyzer_factory(config)
        timeline = run_spinner(
            "Detecting vocal activity and transcribing lyrics...",
            lambda: analyzer.analyze(files["vocals"]),
        )
        timeline = normalize_empty_vocals(timeline)
        timeline = merge_same_kind_segments(timeline, merge_gap=vocal_cfg.merge_gap)
        save_timeline_json(timeline, timeline_json)
        log_file("Timeline JSON", timeline_json)

        vocal_count = sum(1 for seg in timeline if seg.kind == "vocals")
        instrumental_count = sum(1 for seg in timeline if seg.kind == "instrumental")
        console.print(
            f"[green]✓[/green] Timeline segments: "
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
        console.print(
            f"[green]✓[/green] BPM: [yellow]{beat_data.get('bpm')}[/yellow], "
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
