from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from feverslop.domain.timeline import TimelineSegment

if TYPE_CHECKING:
    from feverslop.application.pipeline_context import GenerateRenderPlanContext
    from feverslop.config.app_config import AppConfig
    from feverslop.config.project_config import ProjectConfig
    from feverslop.prompting.concept_prompt_batcher import ConceptPromptBatcher
    from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder
    from feverslop.prompting.h3_prompt_builder import H3PromptBuilder
    from feverslop.prompting.prompt_pipeline import MusicVideoPromptPipeline
    from feverslop.prompting.scene_prompt_builder import ScenePromptBuilder
    from feverslop.ports.llm import LLMPort


class StemSeparatorPort(Protocol):
    def separate(self, input_audio: Path, output_dir: Path) -> dict[str, Path]:
        """Separate the input audio into named stem files."""


class VocalTimelineAnalyzerPort(Protocol):
    def analyze(self, vocals_path: Path) -> list[TimelineSegment]:
        """Analyze the vocal stem into timeline segments."""


class BeatImpactAnalyzerPort(Protocol):
    def analyze_to_json_file(
        self,
        final_mix_path: str | Path,
        output_json_path: str | Path,
        drums_path: str | Path | None = None,
        bass_path: str | Path | None = None,
        vocals_path: str | Path | None = None,
        other_path: str | Path | None = None,
    ) -> Path | None:
        """Analyze beat/impact data and write JSON output."""


class LyricAlignerPort(Protocol):
    def align(self, timeline: list[TimelineSegment], reference_lyrics: str) -> list[TimelineSegment]:
        """Correct vocal segment text using complete reference lyrics."""


StemSeparatorFactory = Callable[["ProjectConfig"], StemSeparatorPort]
VocalTimelineAnalyzerFactory = Callable[["ProjectConfig"], VocalTimelineAnalyzerPort]
BeatImpactAnalyzerFactory = Callable[[], BeatImpactAnalyzerPort]
LyricAlignerFactory = Callable[["GenerateRenderPlanContext"], LyricAlignerPort]
LLMFactory = Callable[["AppConfig"], "LLMPort"]
PromptPipelineFactory = Callable[["LLMPort"], "MusicVideoPromptPipeline"]
ConceptBatcherFactory = Callable[..., "ConceptPromptBatcher"]
ScenePromptBuilderFactory = Callable[["LLMPort"], "ScenePromptBuilder"]
H3PromptBuilderFactory = Callable[["LLMPort"], "H3PromptBuilder | DspyH3PromptBuilder"]
