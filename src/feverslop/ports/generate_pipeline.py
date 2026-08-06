from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from feverslop.ports.llm import LLMPort


class StemSeparatorPort(Protocol):
    def separate(self, input_audio: Path, output_dir: Path) -> dict[str, Path]:
        """Separate the input audio into named stem files."""


class VocalTimelineAnalyzerPort(Protocol):
    def analyze(self, vocals_path: Path) -> list[Any]:
        """Analyze the vocal stem into timeline segments."""


class BeatImpactAnalyzerPort(Protocol):
    def analyze_to_json_file(self, **kwargs: Any) -> Path | None:
        """Analyze beat/impact data and write JSON output."""


class LyricAlignerPort(Protocol):
    def align(self, timeline: list[Any], reference_lyrics: str) -> list[Any]:
        """Correct vocal segment text using complete reference lyrics."""


StemSeparatorFactory = Callable[[Any], StemSeparatorPort]
VocalTimelineAnalyzerFactory = Callable[[Any], VocalTimelineAnalyzerPort]
BeatImpactAnalyzerFactory = Callable[[], BeatImpactAnalyzerPort]
LyricAlignerFactory = Callable[[Any], LyricAlignerPort]
LLMFactory = Callable[[Any], LLMPort]
PromptPipelineFactory = Callable[[LLMPort], Any]
ConceptBatcherFactory = Callable[..., Any]
ScenePromptBuilderFactory = Callable[[LLMPort], Any]
H3PromptBuilderFactory = Callable[[LLMPort], Any]
