from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.ports.generate_pipeline import (
    BeatImpactAnalyzerFactory,
    LyricAlignerFactory,
    StemSeparatorFactory,
    VocalTimelineAnalyzerFactory,
)
from feverslop.domain.timeline import TimelineSegment

logger = logging.getLogger(__name__)


def _close_audio_component(component: Any, model_name: str, reporter: Any) -> None:
    close = getattr(component, "close", None)
    if not callable(close):
        return

    active_error = sys.exception()
    try:
        close()
    except Exception as exc:
        logger.warning("%s model cleanup failed: %s", model_name, exc)
        if active_error is None:
            raise
        try:
            reporter.message(
                f"[yellow]WARNING[/yellow] {model_name} model cleanup failed: {exc}"
            )
        except Exception:  # noqa: BLE001 - reporting must not replace the pipeline error
            logger.debug("Failed to report %s cleanup warning", model_name, exc_info=True)
        return

    try:
        reporter.message(f"[green]OK[/green] {model_name} model unloaded from memory")
    except Exception:  # noqa: BLE001 - reporting is secondary to resource cleanup
        logger.debug("Failed to report %s model cleanup", model_name, exc_info=True)


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
        request = context["request"]
        skip_stems = bool(getattr(request, "skip_stem_separation", False))
        skip_whisper = bool(getattr(request, "skip_whisper", False))
        skip_beats = bool(getattr(request, "skip_beat_analysis", False))

        log_step("1. Demucs Stem Separation")
        if skip_stems:
            files = self._load_existing_stems(paths.stems_dir, config.input_audio)
            reporter.message("[yellow]Skipping stem separation; using existing stems.[/yellow]")
        else:
            separator = self.separator_factory(config)
            try:
                files = run_spinner(
                    "Separating audio into vocals/drums/bass/other...",
                    lambda: separator.separate(config.input_audio, paths.stems_dir),
                )
            finally:
                _close_audio_component(separator, "Demucs", reporter)

        reporter.table(
            "Generated Stems",
            ["Stem", "Path"],
            [[stem_name, str(files[stem_name])] for stem_name in ("vocals", "drums", "bass", "other")],
        )

        log_step("2. Vocal Timeline Analysis")
        vocal_cfg = config.vocal_detection
        if skip_whisper:
            timeline = self._load_existing_timeline(timeline_json, context["artifact_store"])
            reporter.message("[yellow]Skipping Whisper analysis; using existing timeline.[/yellow]")
        else:
            analyzer = self.vocal_analyzer_factory(config)
            try:
                timeline = run_spinner(
                    "Detecting vocal activity and transcribing lyrics...",
                    lambda: analyzer.analyze(files["vocals"]),
                )
            finally:
                _close_audio_component(analyzer, "Whisper", reporter)
        timeline = self.normalize_empty_vocals(timeline)
        timeline = self.merge_same_kind_segments(timeline, merge_gap=vocal_cfg.merge_gap)
        reference_lyrics = str(getattr(config, "lyrics", "") or "").strip()
        if reference_lyrics and self.lyric_aligner_factory is not None:
            aligner = self.lyric_aligner_factory(context)
            vocal_segments = sum(1 for seg in timeline if seg.kind == "vocals")
            reporter.message(
                f"[cyan]LLM lyric alignment: correcting "
                f"{vocal_segments} vocal segments against project lyrics...[/cyan]"
            )
            timeline = run_spinner(
                "Correcting Whisper lyrics against project lyrics...",
                lambda: aligner.align(timeline, reference_lyrics),
            )
            reporter.message(
                f"[green]OK[/green] LLM lyric alignment finished: "
                f"{vocal_segments} vocal segments checked"
            )
        raw_whisper = getattr(analyzer, "raw_whisper_segments", None) if not skip_whisper else None
        if raw_whisper is None:
            self.save_timeline_json(timeline, timeline_json)
        else:
            self.save_timeline_json(timeline, timeline_json, whisper_raw=raw_whisper)
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
        if skip_beats:
            if not beat_json.is_file():
                raise FileNotFoundError(f"Cannot skip beat analysis; missing existing beat data: {beat_json}")
            reporter.message("[yellow]Skipping beat analysis; using existing beat data.[/yellow]")
        else:
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

    @staticmethod
    def _load_existing_stems(stems_dir, input_audio):
        suffixes = {".wav", ".mp3", ".flac"}
        stem_prefix = input_audio.stem
        files = {}
        for name in ("vocals", "drums", "bass", "other"):
            matches = sorted(
                path for path in stems_dir.glob(f"{name}_{stem_prefix}.*")
                if path.is_file() and path.suffix.lower() in suffixes
            )
            if matches:
                files[name] = next((path for path in matches if path.suffix.lower() == ".wav"), matches[0])
        missing = sorted({"vocals", "drums", "bass", "other"} - files.keys())
        if missing:
            raise FileNotFoundError(f"Cannot skip stem separation; missing stems: {', '.join(missing)} in {stems_dir}")
        return files

    @staticmethod
    def _load_existing_timeline(path, artifact_store):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot skip Whisper analysis; missing existing timeline: {path}")
        return [
            TimelineSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                kind=str(item.get("type") or item.get("kind") or "instrumental"),
                text=str(item.get("lyrics") or item.get("text") or ""),
                word_timestamps=tuple(item.get("word_timestamps") or ()),
            )
            for item in artifact_store.read_json(path)
        ]
