# ruff: noqa: F401

from __future__ import annotations

from feverslop.prompting.dspy_h3_analyzer import LocalImageAnalyzer as LocalImageAnalyzer
from feverslop.prompting.dspy_h3_generator_core import VideoPromptGenerator as _VideoPromptGenerator
from feverslop.prompting.dspy_h3_models import (
    BaseVideoPrompt as BaseVideoPrompt,
    GeneratedVideoPrompt as GeneratedVideoPrompt,
    ImageAnalysis as ImageAnalysis,
    ImageAnalysisMode as ImageAnalysisMode,
    MusicIntent as MusicIntent,
    PlannedShot as PlannedShot,
    PlannedSubject as PlannedSubject,
    PromptMode as PromptMode,
    PromptPlan as PromptPlan,
    ReferenceAsset as ReferenceAsset,
    ReferenceKind as ReferenceKind,
    ReferenceLimits as ReferenceLimits,
    ReferenceRole as ReferenceRole,
    ReferenceUsage as ReferenceUsage,
    ReferenceVideoPrompt as ReferenceVideoPrompt,
    ResolvedPromptPlan as ResolvedPromptPlan,
    ResolvedReference as ResolvedReference,
    RetentionAnalysis as RetentionAnalysis,
    RetentionMode as RetentionMode,
    SubjectDefinition as SubjectDefinition,
    VideoPromptRequest as VideoPromptRequest,
)
from feverslop.prompting.dspy_h3_signatures import (
    build_dspy_signatures as build_dspy_signatures,
    build_h3_signature_bundle as build_h3_signature_bundle,
)


class VideoPromptGenerator(_VideoPromptGenerator):
    """Backward-compatible public generator name."""
