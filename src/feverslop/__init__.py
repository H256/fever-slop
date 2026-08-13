"""FeverSlop package namespace."""

from __future__ import annotations

import importlib.metadata
import warnings

__version__: str = importlib.metadata.version(__name__)

_LAZY_MAP: dict[str, tuple[str, str]] = {
    # -- Errors (feverslop.errors) --
    "FeverSlopAdaptationError": ("feverslop.errors", "FeverSlopAdaptationError"),
    "FeverSlopConfigError": ("feverslop.errors", "FeverSlopConfigError"),
    "FeverSlopDataError": ("feverslop.errors", "FeverSlopDataError"),
    "FeverSlopError": ("feverslop.errors", "FeverSlopError"),
    "FeverSlopLMLError": ("feverslop.errors", "FeverSlopLMLError"),
    "FeverSlopRenderError": ("feverslop.errors", "FeverSlopRenderError"),
    "FeverSlopValidationError": ("feverslop.errors", "FeverSlopValidationError"),
    "FeverSlopWorkflowError": ("feverslop.errors", "FeverSlopWorkflowError"),
    # -- Use cases & application types (feverslop.application.movie) --
    "AutoProduceMovieUseCase": ("feverslop.application.movie", "AutoProduceMovieUseCase"),
    "ScaffoldMovieUseCase": ("feverslop.application.movie", "ScaffoldMovieUseCase"),
    "MovieInput": ("feverslop.application.movie", "MovieInput"),
    "MovieProductionResult": ("feverslop.application.movie", "MovieProductionResult"),
    "MovieScaffoldResult": ("feverslop.application.movie", "MovieScaffoldResult"),
    # -- Domain models (feverslop.domain.movie) --
    "CinematicShot": ("feverslop.domain.movie", "CinematicShot"),
    "MovieActor": ("feverslop.domain.movie", "MovieActor"),
    "MovieAct": ("feverslop.domain.movie", "MovieAct"),
    "MovieBible": ("feverslop.domain.movie", "MovieBible"),
    "MovieCharacterArc": ("feverslop.domain.movie", "MovieCharacterArc"),
    "MovieContinuityCharacterState": ("feverslop.domain.movie", "MovieContinuityCharacterState"),
    "MovieContinuityLedger": ("feverslop.domain.movie", "MovieContinuityLedger"),
    "MovieContinuityLocationState": ("feverslop.domain.movie", "MovieContinuityLocationState"),
    "MovieContinuityPlan": ("feverslop.domain.movie", "MovieContinuityPlan"),
    "MovieContinuityRule": ("feverslop.domain.movie", "MovieContinuityRule"),
    "MovieContinuityStyleBible": ("feverslop.domain.movie", "MovieContinuityStyleBible"),
    "MovieLocation": ("feverslop.domain.movie", "MovieLocation"),
    "MovieNarrativeBeat": ("feverslop.domain.movie", "MovieNarrativeBeat"),
    "MovieNarrativePlan": ("feverslop.domain.movie", "MovieNarrativePlan"),
    "MovieProject": ("feverslop.domain.movie", "MovieProject"),
    "MovieSceneBlueprint": ("feverslop.domain.movie", "MovieSceneBlueprint"),
    "MovieSceneCard": ("feverslop.domain.movie", "MovieSceneCard"),
    "MovieSceneContinuityPacket": ("feverslop.domain.movie", "MovieSceneContinuityPacket"),
    "MovieScreenplayArtifact": ("feverslop.domain.movie", "MovieScreenplayArtifact"),
    "MovieScreenplayScene": ("feverslop.domain.movie", "MovieScreenplayScene"),
    "MovieSetupPayoff": ("feverslop.domain.movie", "MovieSetupPayoff"),
    "MovieShotCard": ("feverslop.domain.movie", "MovieShotCard"),
    "MovieStoryDesign": ("feverslop.domain.movie", "MovieStoryDesign"),
    "MovieTurningPoint": ("feverslop.domain.movie", "MovieTurningPoint"),
    "Screenplay": ("feverslop.domain.movie", "Screenplay"),
    "StoryArch": ("feverslop.domain.movie", "StoryArch"),
    # -- Reporter (feverslop.ports.reporting) --
    "ConsoleReporter": ("feverslop.adapters.reporting", "ConsoleReporter"),
    "NullReporter": ("feverslop.adapters.reporting", "NullReporter"),
    "Reporter": ("feverslop.ports.reporting", "Reporter"),
    # -- LLM ports (feverslop.ports.llm) --
    "LLMPort": ("feverslop.ports.llm", "LLMPort"),
    "StoryboardPromptTransformerPort": ("feverslop.ports.llm", "StoryboardPromptTransformerPort"),
    "VisionLLMPort": ("feverslop.ports.llm", "VisionLLMPort"),
    # -- Movie pipeline ports (feverslop.ports.movie) --
    "MovieArtifactWriter": ("feverslop.ports.movie", "MovieArtifactWriter"),
    "ReferenceGenerationPort": ("feverslop.ports.movie", "ReferenceGenerationPort"),
    "ScenePlanningPort": ("feverslop.ports.movie", "ScenePlanningPort"),
    "StoryGenerationPort": ("feverslop.ports.movie", "StoryGenerationPort"),
    "VisualGenerationPort": ("feverslop.ports.movie", "VisualGenerationPort"),
    # -- Artifact ports (feverslop.ports.artifacts) --
    "ArtifactStore": ("feverslop.ports.artifacts", "ArtifactStore"),
    "JsonArtifactStore": ("feverslop.ports.artifacts", "JsonArtifactStore"),
    "RenderPlanStore": ("feverslop.ports.artifacts", "RenderPlanStore"),
    "TextArtifactReaderWriter": ("feverslop.ports.artifacts", "TextArtifactReaderWriter"),
    # -- Full-auto ports (feverslop.ports.full_auto) --
    "PipelineRunnerPort": ("feverslop.ports.full_auto", "PipelineRunnerPort"),
    "ProjectScaffoldPort": ("feverslop.ports.full_auto", "ProjectScaffoldPort"),
    "SongAudioGeneratorPort": ("feverslop.ports.full_auto", "SongAudioGeneratorPort"),
    "SongBriefGeneratorPort": ("feverslop.ports.full_auto", "SongBriefGeneratorPort"),
    # -- Generate-pipeline ports (feverslop.ports.generate_pipeline) --
    "BeatImpactAnalyzerPort": ("feverslop.ports.generate_pipeline", "BeatImpactAnalyzerPort"),
    "LyricAlignerPort": ("feverslop.ports.generate_pipeline", "LyricAlignerPort"),
    "StemSeparatorPort": ("feverslop.ports.generate_pipeline", "StemSeparatorPort"),
    "VocalTimelineAnalyzerPort": ("feverslop.ports.generate_pipeline", "VocalTimelineAnalyzerPort"),
    # -- Post-processing ports (feverslop.ports.postprocessing) --
    "PostProcessorPort": ("feverslop.ports.postprocessing", "PostProcessorPort"),
    # -- Rendering ports (feverslop.ports.rendering) --
    "ImageRenderBackend": ("feverslop.ports.rendering", "ImageRenderBackend"),
    "ImageRenderRequest": ("feverslop.ports.rendering", "ImageRenderRequest"),
    "RenderBackendConfig": ("feverslop.ports.rendering", "RenderBackendConfig"),
    "VideoRenderBackend": ("feverslop.ports.rendering", "VideoRenderBackend"),
    "VideoRenderRequest": ("feverslop.ports.rendering", "VideoRenderRequest"),
    "WorkflowAnchorConfig": ("feverslop.ports.rendering", "WorkflowAnchorConfig"),
    # -- Scene-document ports (feverslop.ports.scene_documents) --
    "SceneDocumentConflict": ("feverslop.ports.scene_documents", "SceneDocumentConflict"),
    "SceneDocumentPort": ("feverslop.ports.scene_documents", "SceneDocumentPort"),
    "SceneDocumentSnapshot": ("feverslop.ports.scene_documents", "SceneDocumentSnapshot"),
    "SceneLtxPromptField": ("feverslop.ports.scene_documents", "SceneLtxPromptField"),
    "SceneMediaPort": ("feverslop.ports.scene_documents", "SceneMediaPort"),
    # -- Workflow ports (feverslop.ports.workflow) --
    "PreparedWorkflowRendererPort": ("feverslop.ports.workflow", "PreparedWorkflowRendererPort"),
    "WorkflowBackendPort": ("feverslop.ports.workflow", "WorkflowBackendPort"),
    "WorkflowMaterializationRequest": ("feverslop.ports.workflow", "WorkflowMaterializationRequest"),
    "WorkflowMaterializerPort": ("feverslop.ports.workflow", "WorkflowMaterializerPort"),
    # -- Domain utilities (feverslop.domain.movie_utils and slug_utils) --
    "clean_visual_description": ("feverslop.domain.movie_utils", "clean_visual_description"),
    "configured_actors": ("feverslop.domain.movie_utils", "configured_actors"),
    "configured_locations": ("feverslop.domain.movie_utils", "configured_locations"),
    "display_name": ("feverslop.domain.movie_utils", "display_name"),
    "safe_id": ("feverslop.domain.movie_utils", "safe_id"),
    "safe_id_list": ("feverslop.domain.movie_utils", "safe_id_list"),
    "string_list": ("feverslop.domain.movie_utils", "string_list"),
    "transition_from_previous": ("feverslop.domain.movie_utils", "transition_from_previous"),
    "slugify_project_name": ("feverslop.domain.slug_utils", "slugify_project_name"),
}

_DEPRECATED: dict[str, str] = {}  # name -> deprecation message (populated when symbols are deprecated)


_INTERNAL_SUBMODULES = frozenset(["config", "application", "path_utils", "domain", "ports", "errors"])


def __getattr__(name: str):
    """PEP 562 lazy loading of public API symbols."""
    if name in _INTERNAL_SUBMODULES:
        raise ImportError(
            f"cannot import name {name!r} from {__name__!r} "
            f"(internal submodule; use from feverslop.{name} if needed)",
            name=__name__ + "." + name,
        )
    if name in _LAZY_MAP:
        import importlib as _importlib

        mod_name, attr_name = _LAZY_MAP[name]
        mod = _importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val  # cache for subsequent access

        if name in _DEPRECATED:
            warnings.warn(_DEPRECATED[name], DeprecationWarning, stacklevel=2)

        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return all public API symbols."""
    return list(_LAZY_MAP.keys()) + [
        "__version__",
        "__all__",
        "__doc__",
    ] + [k for k in globals() if not k.startswith("_") or k == "__version__"]


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # -- Errors --
    "FeverSlopAdaptationError",
    "FeverSlopConfigError",
    "FeverSlopDataError",
    "FeverSlopError",
    "FeverSlopLMLError",
    "FeverSlopRenderError",
    "FeverSlopValidationError",
    "FeverSlopWorkflowError",
    # -- Use cases --
    "AutoProduceMovieUseCase",
    "ScaffoldMovieUseCase",
    # -- Application types --
    "MovieInput",
    "MovieProductionResult",
    "MovieScaffoldResult",
    # -- Domain models --
    "CinematicShot",
    "MovieActor",
    "MovieAct",
    "MovieBible",
    "MovieCharacterArc",
    "MovieContinuityCharacterState",
    "MovieContinuityLedger",
    "MovieContinuityLocationState",
    "MovieContinuityPlan",
    "MovieContinuityRule",
    "MovieContinuityStyleBible",
    "MovieLocation",
    "MovieNarrativeBeat",
    "MovieNarrativePlan",
    "MovieProject",
    "MovieSceneBlueprint",
    "MovieSceneCard",
    "MovieSceneContinuityPacket",
    "MovieScreenplayArtifact",
    "MovieScreenplayScene",
    "MovieSetupPayoff",
    "MovieShotCard",
    "MovieStoryDesign",
    "MovieTurningPoint",
    "Screenplay",
    "StoryArch",
    # -- Reporter --
    "ConsoleReporter",
    "NullReporter",
    "Reporter",
    # -- LLM ports --
    "LLMPort",
    "StoryboardPromptTransformerPort",
    "VisionLLMPort",
    # -- Movie pipeline ports --
    "MovieArtifactWriter",
    "ReferenceGenerationPort",
    "ScenePlanningPort",
    "StoryGenerationPort",
    "VisualGenerationPort",
    # -- Artifact ports --
    "ArtifactStore",
    "JsonArtifactStore",
    "RenderPlanStore",
    "TextArtifactReaderWriter",
    # -- Full-auto ports --
    "PipelineRunnerPort",
    "ProjectScaffoldPort",
    "SongAudioGeneratorPort",
    "SongBriefGeneratorPort",
    # -- Generate-pipeline ports --
    "BeatImpactAnalyzerPort",
    "LyricAlignerPort",
    "StemSeparatorPort",
    "VocalTimelineAnalyzerPort",
    # -- Post-processing ports --
    "PostProcessorPort",
    # -- Rendering ports and types --
    "ImageRenderBackend",
    "ImageRenderRequest",
    "RenderBackendConfig",
    "VideoRenderBackend",
    "VideoRenderRequest",
    "WorkflowAnchorConfig",
    # -- Scene-document ports and types --
    "SceneDocumentConflict",
    "SceneDocumentPort",
    "SceneDocumentSnapshot",
    "SceneLtxPromptField",
    "SceneMediaPort",
    # -- Workflow ports and types --
    "PreparedWorkflowRendererPort",
    "WorkflowBackendPort",
    "WorkflowMaterializationRequest",
    "WorkflowMaterializerPort",
    # -- Domain utilities --
    "clean_visual_description",
    "configured_actors",
    "configured_locations",
    "display_name",
    "safe_id",
    "safe_id_list",
    "slugify_project_name",
    "string_list",
    "transition_from_previous",
]
