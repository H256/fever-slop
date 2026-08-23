
from __future__ import annotations

from feverslop.prompting.dspy_h3_analyzer import (
    LocalImageAnalyzer as LocalImageAnalyzer,
)
from feverslop.prompting.dspy_h3_generator_core import (
    VideoPromptGenerator as _VideoPromptGenerator,
)
from feverslop.prompting.dspy_h3_models import (
    BaseVideoPrompt as BaseVideoPrompt,
)
from feverslop.prompting.dspy_h3_models import (
    GeneratedVideoPrompt as GeneratedVideoPrompt,
)
from feverslop.prompting.dspy_h3_models import (
    ImageAnalysis as ImageAnalysis,
)
from feverslop.prompting.dspy_h3_models import (
    ImageAnalysisMode as ImageAnalysisMode,
)
from feverslop.prompting.dspy_h3_models import (
    MusicIntent as MusicIntent,
)
from feverslop.prompting.dspy_h3_models import (
    PlannedShot as PlannedShot,
)
from feverslop.prompting.dspy_h3_models import (
    PlannedSubject as PlannedSubject,
)
from feverslop.prompting.dspy_h3_models import (
    PromptMode as PromptMode,
)
from feverslop.prompting.dspy_h3_models import (
    PromptPlan as PromptPlan,
)
from feverslop.prompting.dspy_h3_models import (
    ReferenceAsset as ReferenceAsset,
)
from feverslop.prompting.dspy_h3_models import (
    ReferenceKind as ReferenceKind,
)
from feverslop.prompting.dspy_h3_models import (
    ReferenceLimits as ReferenceLimits,
)
from feverslop.prompting.dspy_h3_models import (
    ReferenceRole as ReferenceRole,
)
from feverslop.prompting.dspy_h3_models import (
    ReferenceUsage as ReferenceUsage,
)
from feverslop.prompting.dspy_h3_models import (
    ReferenceVideoPrompt as ReferenceVideoPrompt,
)
from feverslop.prompting.dspy_h3_models import (
    ResolvedPromptPlan as ResolvedPromptPlan,
)
from feverslop.prompting.dspy_h3_models import (
    ResolvedReference as ResolvedReference,
)
from feverslop.prompting.dspy_h3_models import (
    RetentionAnalysis as RetentionAnalysis,
)
from feverslop.prompting.dspy_h3_models import (
    RetentionMode as RetentionMode,
)
from feverslop.prompting.dspy_h3_models import (
    SubjectDefinition as SubjectDefinition,
)
from feverslop.prompting.dspy_h3_models import (
    VideoPromptRequest as VideoPromptRequest,
)
from feverslop.prompting.dspy_h3_signatures import (
    build_dspy_signatures as build_dspy_signatures,
)
from feverslop.prompting.dspy_h3_signatures import (
    build_h3_signature_bundle as build_h3_signature_bundle,
)


class VideoPromptGenerator(_VideoPromptGenerator):
    """Backward-compatible public generator name."""
