# ruff: noqa: E402, F401, F811

from __future__ import annotations

import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PromptMode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    FL2V = "fl2v"
    L2V = "l2v"
    R2V = "r2v"


class ReferenceKind(str, Enum):
    PICTURE = "picture"
    AUDIO = "audio"
    VIDEO = "video"


class ReferenceRole(str, Enum):
    SUBJECT = "subject"
    STYLE = "style"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    KEYFRAME = "keyframe"
    STORYBOARD = "storyboard"
    COMPOSITION = "composition"
    ENVIRONMENT = "environment"
    MOTION = "motion"
    CAMERA = "camera"
    EDIT_SOURCE = "edit_source"
    CONTINUATION = "continuation"
    TEMPORAL_STRUCTURE = "temporal_structure"
    AUDIO_REUSE = "audio_reuse"
    VOICE = "voice"
    MUSIC_STYLE = "music_style"
    RHYTHM = "rhythm"
    SOUND_STYLE = "sound_style"
    GENERAL = "general"


class ImageAnalysisMode(str, Enum):
    OFF = "off"
    MISSING_ONLY = "missing_only"
    ALWAYS = "always"


class RetentionMode(str, Enum):
    FULLY_PRESERVED = "fully_preserved"
    PARTIALLY_PRESERVED = "partially_preserved"
    ATTRIBUTE_TRANSFER = "attribute_transfer"
    STYLE_TRANSFER = "style_transfer"
    ENVIRONMENT_TRANSFER = "environment_transfer"
    MOTION_TRANSFER = "motion_transfer"
    AUDIO_TRANSFER = "audio_transfer"
    TRANSFORMED = "transformed"


class MusicIntent(str, Enum):
    NONE = "none"
    GENERATE = "generate"
    REFERENCE = "reference"


class ReferenceAsset(BaseModel):
    kind: ReferenceKind
    source: str
    role: ReferenceRole = ReferenceRole.GENERAL
    description: str | None = None
    name: str | None = None
    use_audio: bool = False


class ReferenceLimits(BaseModel):
    max_pictures: int = 10
    max_audio: int = 5
    max_videos: int = 3


class VideoPromptRequest(BaseModel):
    mode: PromptMode
    user_prompt: str = Field(min_length=1)
    duration_seconds: float | None = Field(default=None, gt=0)
    references: list[ReferenceAsset] = Field(default_factory=list)
    notes: str | None = None
    strict_fidelity: bool = True
    music_intent: MusicIntent | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "VideoPromptRequest":
        required = {
            PromptMode.I2V: (ReferenceRole.FIRST_FRAME,),
            PromptMode.FL2V: (ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME),
            PromptMode.L2V: (ReferenceRole.LAST_FRAME,),
        }.get(self.mode)
        if required:
            for role in required:
                if sum(ref.role == role and ref.kind == ReferenceKind.PICTURE for ref in self.references) != 1:
                    raise ValueError(f"{self.mode.value} requires exactly one picture with role={role.value!r}")
        return self


class ResolvedReference(BaseModel):
    label: str
    kind: ReferenceKind
    source: str
    role: ReferenceRole
    description: str
    name: str | None = None
    use_audio: bool = False


class ImageAnalysis(BaseModel):
    objective_description: str
    visible_subjects: list[str] = Field(default_factory=list)
    environment: str | None = None
    visual_style: str | None = None
    composition: str | None = None
    lighting: str | None = None
    visible_text: list[str] = Field(default_factory=list)


class LocalImageAnalyzer:
    """Analyze local picture references only when the configured mode allows it."""

    def __init__(self, predictor: Any, mode: ImageAnalysisMode = ImageAnalysisMode.MISSING_ONLY):
        self.predictor = predictor
        self.mode = mode

    def should_analyze(self, reference: ReferenceAsset) -> bool:
        return (
            reference.kind == ReferenceKind.PICTURE
            and self.mode != ImageAnalysisMode.OFF
            and self._local_file(reference.source) is not None
            and (self.mode == ImageAnalysisMode.ALWAYS or not reference.description)
        )

    @staticmethod
    def _local_file(source: str) -> Path | None:
        path = Path(source).expanduser()
        return path.resolve() if path.is_file() else None

    def analyze(self, reference: ReferenceAsset) -> str:
        path = self._local_file(reference.source)
        if path is None:
            raise ValueError(f"Image is not a local file: {reference.source}")
        analysis = self.predictor(
            image=__import__("dspy").Image.from_path(str(path)),
            intended_role=reference.role.value,
            user_hint=reference.description or "",
        ).analysis
        return "\n".join(filter(None, [
            analysis.objective_description,
            "Visible subjects: " + "; ".join(analysis.visible_subjects) if analysis.visible_subjects else "",
            f"Environment: {analysis.environment}" if analysis.environment else "",
            f"Visual style: {analysis.visual_style}" if analysis.visual_style else "",
            f"Composition: {analysis.composition}" if analysis.composition else "",
            f"Lighting: {analysis.lighting}" if analysis.lighting else "",
            "Visible text: " + "; ".join(analysis.visible_text) if analysis.visible_text else "",
        ]))


class PlannedSubject(BaseModel):
    name: str
    description: str
    source_references: list[str] = Field(default_factory=list)


class ReferenceUsage(BaseModel):
    reference_label: str
    purpose: str
    details: str


class PlannedShot(BaseModel):
    shot_number: int = Field(ge=1)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    description: str
    involved_subjects: list[str] = Field(default_factory=list)
    reference_labels: list[str] = Field(default_factory=list)


class PromptPlan(BaseModel):
    creative_intent: str
    subjects: list[PlannedSubject] = Field(default_factory=list)
    reference_usage: list[ReferenceUsage] = Field(default_factory=list)
    shots: list[PlannedShot] = Field(default_factory=list)
    overall_soundscape: str
    music_intent: MusicIntent
    non_diegetic_music: str | None = None
    alignment_instruction: str | None = None

    @model_validator(mode="after")
    def validate_music(self) -> "PromptPlan":
        if self.music_intent == MusicIntent.NONE:
            self.non_diegetic_music = None
        elif not self.non_diegetic_music:
            raise ValueError("A non_diegetic_music description is required when music is enabled")
        return self


class SubjectDefinition(BaseModel):
    label: str
    name: str
    description: str
    source_references: list[str] = Field(default_factory=list)

    def render(self) -> str:
        text = f"{self.label} ({self.name}): {self.description}"
        if self.source_references:
            text += " Source references: " + ", ".join(self.source_references) + "."
        return text


class ResolvedPromptPlan(BaseModel):
    creative_intent: str
    subjects: list[SubjectDefinition] = Field(default_factory=list)
    reference_usage: list[ReferenceUsage] = Field(default_factory=list)
    shots: list[PlannedShot] = Field(default_factory=list)
    overall_soundscape: str
    music_intent: MusicIntent
    non_diegetic_music: str | None = None
    alignment_instruction: str | None = None


class RetentionAnalysis(BaseModel):
    target_label: str
    mode: str
    details: str
    shots: list[int] = Field(default_factory=list)

    def render(self) -> str:
        shots = f" (appears in {', '.join(f'[Shot {n}]' for n in self.shots)})" if self.shots else ""
        return f"{self.target_label}{shots}: {self.mode} - {self.details}"


class BaseVideoPrompt(BaseModel):
    integrated_multimodal_description: str
    overall_soundscape: str
    non_diegetic_music: str | None = None
    alignment_instruction: str | None = None

    def render(self) -> str:
        parts = [self.alignment_instruction.strip()] if self.alignment_instruction else []
        parts.extend([
            "integrated_multimodal_description: " + self.integrated_multimodal_description.strip(),
            "overall_soundscape: " + self.overall_soundscape.strip(),
            "non_diegetic_music: " + (self.non_diegetic_music.strip() if self.non_diegetic_music else "N/A"),
        ])
        return "\n\n".join(parts)


class ReferenceVideoPrompt(BaseModel):
    subject_definitions: list[SubjectDefinition]
    summary: str
    retention_analysis: list[RetentionAnalysis]
    detailed_description: str
    overall_soundscape: str
    non_diegetic_music: str | None = None

    def render(self) -> str:
        subjects = "\n".join(item.render() for item in self.subject_definitions)
        retention = "\n".join(item.render() for item in self.retention_analysis)
        return "\n\n".join([
            f"subject_definitions:\n{subjects}",
            f"summary: {self.summary.strip()}",
            f"retention_analysis:\n{retention}",
            "detailed_description: " + self.detailed_description.strip(),
            "overall_soundscape: " + self.overall_soundscape.strip(),
            "non_diegetic_music: " + (self.non_diegetic_music.strip() if self.non_diegetic_music else "N/A"),
        ])


class GeneratedVideoPrompt(BaseModel):
    mode: PromptMode
    prompt: BaseVideoPrompt | ReferenceVideoPrompt
    plan: ResolvedPromptPlan
    references: list[ResolvedReference]

    @property
    def rendered_prompt(self) -> str:
        return self.prompt.render()


def _dspy_signatures():
    import dspy

    class AnalyzeImage(dspy.Signature):
        """Analyze only observable information in a reference image for video generation."""
        image: dspy.Image = dspy.InputField()
        intended_role: str = dspy.InputField()
        user_hint: str = dspy.InputField()
        analysis: ImageAnalysis = dspy.OutputField()

    class BuildPromptPlan(dspy.Signature):
        """Create a strict, structured production plan using only supplied references."""
        mode: str = dspy.InputField()
        user_prompt: str = dspy.InputField()
        duration_seconds: float | None = dspy.InputField()
        references_json: str = dspy.InputField()
        notes: str = dspy.InputField()
        strict_fidelity: bool = dspy.InputField()
        requested_music_intent: str = dspy.InputField()
        plan: PromptPlan = dspy.OutputField()

    class RenderBasePrompt(dspy.Signature):
        """Render a production-ready MiniMax base prompt; the guide is authoritative."""
        guide: str = dspy.InputField()
        mode: str = dspy.InputField()
        user_prompt: str = dspy.InputField()
        plan_json: str = dspy.InputField()
        references_json: str = dspy.InputField()
        strict_fidelity: bool = dspy.InputField()
        music_intent: str = dspy.InputField()
        result: BaseVideoPrompt = dspy.OutputField()

    class RenderReferencePrompt(dspy.Signature):
        """Render all generated portions of a MiniMax full-reference prompt."""
        guide: str = dspy.InputField()
        user_prompt: str = dspy.InputField()
        plan_json: str = dspy.InputField()
        references_json: str = dspy.InputField()
        strict_fidelity: bool = dspy.InputField()
        music_intent: str = dspy.InputField()
        summary: str = dspy.OutputField()
        retention_analysis: list[RetentionAnalysis] = dspy.OutputField()
        detailed_description: str = dspy.OutputField()
        overall_soundscape: str = dspy.OutputField()
        non_diegetic_music: str | None = dspy.OutputField()

    return AnalyzeImage, BuildPromptPlan, RenderBasePrompt, RenderReferencePrompt


class VideoPromptGenerator:
    """Integrated DSPy planner/analyzer/renderer from the reference generator."""

    def __init__(self, *, base_guide_path: str | Path, reference_guide_path: str | Path,
                 llm: Any, image_analysis_mode: ImageAnalysisMode = ImageAnalysisMode.MISSING_ONLY,
                 limits: ReferenceLimits | None = None):
        import dspy

        self.base_guide_path = Path(base_guide_path)
        self.reference_guide_path = Path(reference_guide_path)
        self.limits = limits or ReferenceLimits()
        AnalyzeImage, BuildPromptPlan, RenderBasePrompt, RenderReferencePrompt = _dspy_signatures()
        self.image_analyzer = LocalImageAnalyzer(
            dspy.Predict(AnalyzeImage), image_analysis_mode
        )
        self.planner = dspy.Predict(BuildPromptPlan)
        self.base_renderer = dspy.Predict(RenderBasePrompt)
        self.reference_renderer = dspy.Predict(RenderReferencePrompt)
        client = getattr(llm, "client", None)
        api_base = getattr(client, "base_url", None)
        if api_base is not None and not isinstance(api_base, str):
            api_base = str(api_base)
        self.lm = dspy.LM(
            f"openai/{llm.model}",
            api_base=api_base,
            api_key=getattr(client, "api_key", None),
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
        )

    @staticmethod
    def _read(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Guide not found: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _local_file(source: str) -> Path | None:
        path = Path(source).expanduser()
        return path.resolve() if path.is_file() else None

    @staticmethod
    def _label(kind: ReferenceKind, number: int) -> str:
        return f"<{kind.value.title()} {number}>"

    def _resolve_references(self, refs: list[ReferenceAsset]) -> list[ResolvedReference]:
        counts = defaultdict(int)
        result = []
        for ref in refs:
            counts[ref.kind] += 1
            if counts[ref.kind] > getattr(self.limits, f"max_{ref.kind.value}"):
                raise ValueError(f"Too many {ref.kind.value} references")
            description = ref.description
            if self.image_analyzer.should_analyze(ref):
                description = self.image_analyzer.analyze(ref)
            if not description:
                raise ValueError(f"{self._label(ref.kind, counts[ref.kind])} has no usable description")
            result.append(ResolvedReference(label=self._label(ref.kind, counts[ref.kind]), kind=ref.kind,
                                             source=ref.source, role=ref.role, description=description,
                                             name=ref.name, use_audio=ref.use_audio))
        return result

    @staticmethod
    def _json(value: BaseModel | list[BaseModel]) -> str:
        data = value.model_dump(mode="json") if isinstance(value, BaseModel) else [item.model_dump(mode="json") for item in value]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _plan(self, request: VideoPromptRequest, refs: list[ResolvedReference]) -> ResolvedPromptPlan:
        prediction = self.planner(
            mode=request.mode.value, user_prompt=request.user_prompt, duration_seconds=request.duration_seconds,
            references_json=self._json(refs), notes=request.notes or "", strict_fidelity=request.strict_fidelity,
            requested_music_intent=request.music_intent.value if request.music_intent else "",
        )
        plan = prediction.plan
        if request.music_intent is not None:
            plan.music_intent = request.music_intent
        if plan.music_intent == MusicIntent.NONE:
            plan.non_diegetic_music = None
        allowed = {ref.label for ref in refs}
        for subject in plan.subjects:
            if any(label not in allowed for label in subject.source_references):
                raise ValueError("Planner invented an unknown reference")
        for usage in plan.reference_usage:
            if usage.reference_label not in allowed:
                raise ValueError("Planner invented an unknown reference")
        subjects = []
        seen = set()
        for index, subject in enumerate(plan.subjects, 1):
            key = subject.name.strip().casefold()
            if key in seen:
                raise ValueError(f"Planner created duplicate subject name: {subject.name!r}")
            seen.add(key)
            subjects.append(SubjectDefinition(label=f"<Subject {index}>", name=subject.name,
                                               description=subject.description, source_references=subject.source_references))
        return ResolvedPromptPlan(creative_intent=plan.creative_intent, subjects=subjects,
                                  reference_usage=plan.reference_usage, shots=plan.shots,
                                  overall_soundscape=plan.overall_soundscape, music_intent=plan.music_intent,
                                  non_diegetic_music=plan.non_diegetic_music, alignment_instruction=plan.alignment_instruction)

    def __call__(self, request_data: dict[str, Any]) -> GeneratedVideoPrompt:
        if request_data.get("mode") == "ref":
            request_data = {**request_data, "mode": PromptMode.R2V.value}
        request = VideoPromptRequest.model_validate(request_data)
        refs = self._resolve_references(request.references)
        plan = self._plan(request, refs)
        with __import__("dspy").context(lm=self.lm):
            plan_json = self._json(plan)
            references_json = self._json(refs)
            if request.mode == PromptMode.R2V:
                output = self.reference_renderer(guide=self._read(self.reference_guide_path), user_prompt=request.user_prompt,
                    plan_json=plan_json, references_json=references_json, strict_fidelity=request.strict_fidelity,
                    music_intent=plan.music_intent.value)
                prompt = ReferenceVideoPrompt(subject_definitions=plan.subjects, summary=output.summary,
                    retention_analysis=output.retention_analysis, detailed_description=output.detailed_description,
                    overall_soundscape=output.overall_soundscape, non_diegetic_music=output.non_diegetic_music)
            else:
                output = self.base_renderer(guide=self._read(self.base_guide_path), mode=request.mode.value,
                    user_prompt=request.user_prompt, plan_json=plan_json, references_json=references_json,
                    strict_fidelity=request.strict_fidelity, music_intent=plan.music_intent.value)
                prompt = output.result
        if plan.music_intent == MusicIntent.NONE:
            prompt.non_diegetic_music = None
        return GeneratedVideoPrompt(mode=request.mode, prompt=prompt, plan=plan, references=refs)


# Compatibility exports.  The implementation is split into focused modules,
# while this historical import path remains stable for existing callers.
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
from feverslop.prompting.dspy_h3_signatures import build_dspy_signatures


class VideoPromptGenerator(_VideoPromptGenerator):
    """Backward-compatible public generator name."""