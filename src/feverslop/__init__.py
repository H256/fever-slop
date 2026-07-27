"""FeverSlop package namespace."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Use cases (application layer)
# ---------------------------------------------------------------------------

from feverslop.application.movie import (
    AutoProduceMovieUseCase,
    ScaffoldMovieUseCase,
)

# ---------------------------------------------------------------------------
# Application types
# ---------------------------------------------------------------------------

from feverslop.application.movie import (
    MovieInput,
    MovieProductionResult,
    MovieScaffoldResult,
    validate_movie_input,
)

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

from feverslop.domain.movie import (
    CinematicShot,
    MovieActor,
    MovieAct,
    MovieBible,
    MovieCharacterArc,
    MovieContinuityCharacterState,
    MovieContinuityLedger,
    MovieContinuityLocationState,
    MovieContinuityPlan,
    MovieContinuityRule,
    MovieContinuityStyleBible,
    MovieLocation,
    MovieNarrativeBeat,
    MovieNarrativePlan,
    MovieProject,
    MovieSceneBlueprint,
    MovieSceneCard,
    MovieSceneContinuityPacket,
    MovieScreenplayArtifact,
    MovieScreenplayScene,
    MovieSetupPayoff,
    MovieShotCard,
    MovieStoryDesign,
    MovieTurningPoint,
    Screenplay,
    StoryArch,
)

# ---------------------------------------------------------------------------
# Port protocols — reporting
# ---------------------------------------------------------------------------

from feverslop.ports.reporting import (
    ConsoleReporter,
    NullReporter,
    Reporter,
)

# ---------------------------------------------------------------------------
# Port protocols — LLM
# ---------------------------------------------------------------------------

from feverslop.ports.llm import (
    LLMPort,
    StoryboardPromptTransformerPort,
    VisionLLMPort,
)

# ---------------------------------------------------------------------------
# Port protocols — movie pipeline
# ---------------------------------------------------------------------------

from feverslop.ports.movie import (
    MovieArtifactWriter,
    ReferenceGenerationPort,
    ScenePlanningPort,
    StoryGenerationPort,
    VisualGenerationPort,
)

# ---------------------------------------------------------------------------
# Port protocols — artifacts
# ---------------------------------------------------------------------------

from feverslop.ports.artifacts import (
    ArtifactStore,
    JsonArtifactStore,
    RenderPlanStore,
    TextArtifactReaderWriter,
)

# ---------------------------------------------------------------------------
# Port protocols — audio
# ---------------------------------------------------------------------------

from feverslop.ports.audio import AudioAnalyzerPort

# ---------------------------------------------------------------------------
# Port protocols — full auto
# ---------------------------------------------------------------------------

from feverslop.ports.full_auto import (
    PipelineRunnerPort,
    ProjectScaffoldPort,
    SongAudioGeneratorPort,
    SongBriefGeneratorPort,
)

# ---------------------------------------------------------------------------
# Port protocols — generate pipeline
# ---------------------------------------------------------------------------

from feverslop.ports.generate_pipeline import (
    BeatImpactAnalyzerPort,
    LyricAlignerPort,
    StemSeparatorPort,
    VocalTimelineAnalyzerPort,
)

# ---------------------------------------------------------------------------
# Port protocols — postprocessing
# ---------------------------------------------------------------------------

from feverslop.ports.postprocessing import PostProcessorPort

# ---------------------------------------------------------------------------
# Port protocols — rendering
# ---------------------------------------------------------------------------

from feverslop.ports.rendering import (
    ImageRenderBackend,
    ImageRenderRequest,
    RenderBackendConfig,
    VideoRenderBackend,
    VideoRenderRequest,
    WorkflowAnchorConfig,
)

# ---------------------------------------------------------------------------
# Port protocols — scene documents
# ---------------------------------------------------------------------------

from feverslop.ports.scene_documents import (
    SceneDocumentConflict,
    SceneDocumentPort,
    SceneDocumentSnapshot,
    SceneLtxPromptField,
    SceneMediaPort,
)

# ---------------------------------------------------------------------------
# Port protocols — workflow
# ---------------------------------------------------------------------------

from feverslop.ports.workflow import (
    PreparedWorkflowRendererPort,
    WorkflowBackendPort,
    WorkflowMaterializationRequest,
    WorkflowMaterializerPort,
)

# ---------------------------------------------------------------------------
# Domain utilities
# ---------------------------------------------------------------------------

from feverslop.domain.movie_utils import (
    clean_visual_description,
    configured_actors,
    configured_locations,
    display_name,
    safe_id,
    safe_id_list,
    string_list,
    transition_from_previous,
)

from feverslop.domain.slug_utils import slugify_project_name

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # -- Use cases --
    "AutoProduceMovieUseCase",
    "ScaffoldMovieUseCase",
    # -- Application types --
    "MovieInput",
    "MovieProductionResult",
    "MovieScaffoldResult",
    "validate_movie_input",
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
    # -- Audio ports --
    "AudioAnalyzerPort",
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
