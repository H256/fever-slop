from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from feverslop.ports.llm import LLMPort


class StemSeparatorPort(Protocol):
    def separate(self, input_audio: Path, output_dir: Path) -> dict[str, Path]:
        """Separate the input audio into named stem files."""


class VocalTimelineAnalyzerPort(Protocol):
    def analyze(self, vocals_path: Path) -> list[Any]:
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
    def align(self, timeline: list[Any], reference_lyrics: str) -> list[Any]:
        """Correct vocal segment text using complete reference lyrics."""


StemSeparatorFactory = Callable[[dict[str, Any]], StemSeparatorPort]
VocalTimelineAnalyzerFactory = Callable[[dict[str, Any]], VocalTimelineAnalyzerPort]
BeatImpactAnalyzerFactory = Callable[[], BeatImpactAnalyzerPort]
LyricAlignerFactory = Callable[[dict[str, Any]], LyricAlignerPort]
LLMFactory = Callable[[dict[str, Any]], "LLMPort"]
# Returns a prompt pipeline object (e.g. MusicVideoPromptPipeline)
PromptPipelineFactory = Callable[["LLMPort"], Any]
# Returns a concept batcher object (e.g. ConceptPromptBatcher)
# Takes llm, batch_size, and optional keyword args
ConceptBatcherFactory = Callable[..., Any]
# Returns a scene prompt builder (e.g. ScenePromptBuilder)
ScenePromptBuilderFactory = Callable[["LLMPort"], Any]
# Returns an H3 prompt builder (e.g. H3PromptBuilder or DspyH3PromptBuilder)
H3PromptBuilderFactory = Callable[["LLMPort"], Any]
