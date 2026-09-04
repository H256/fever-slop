from __future__ import annotations

from enum import Enum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feverslop.domain.continuation_intent import ContinuationIntent


class PromptMode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    FL2V = "fl2v"
    L2V = "l2v"
    R2V = "r2v"


class PromptJudgeResult(BaseModel):
    verdict: Literal["good", "bad"]
    issues: list[str] = Field(default_factory=list)
    repair_instruction: str = ""
    suggested_prompt: str = ""
    field_issues: list["CreativeFieldIssue"] = Field(default_factory=list)


class CreativeFieldIssue(BaseModel):
    """Addressable judge feedback for one mutable creative shot field."""

    model_config = ConfigDict(extra="forbid")

    shot_id: str = Field(min_length=1)
    field: Literal[
        "visible_action",
        "performance",
        "camera_behavior",
        "environmental_motion",
        "transition_intent",
    ]
    issue_code: str = Field(min_length=1)
    repair_instruction: str = Field(min_length=1)


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
    relay_segments: list[dict] = Field(default_factory=list)
    audio_subject_bindings: list[AudioSubjectBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mode(self) -> VideoPromptRequest:
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


class AudioSubjectBinding(BaseModel):
    audio_label: str
    stem: str
    subject_label: str
    speaker_id: str | None = None


class ImageAnalysis(BaseModel):
    objective_description: str
    visible_subjects: list[str] = Field(default_factory=list)
    environment: str | None = None
    visual_style: str | None = None
    composition: str | None = None
    lighting: str | None = None
    visible_text: list[str] = Field(default_factory=list)


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
    visible_action: str | None = None
    performance: str | None = None
    camera_behavior: str | None = None
    environmental_motion: str | None = None
    transition_intent: str | None = None
    involved_subjects: list[str] = Field(default_factory=list)
    reference_labels: list[str] = Field(default_factory=list)
    hard_cut_after: bool = False


class H3CreativeShot(BaseModel):
    """Creative shot prose returned by the H3 LLM; labels and timing syntax are compiler-owned."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    visible_action: str | None = None
    performance: str | None = None
    camera_behavior: str | None = None
    environmental_motion: str | None = None
    transition_intent: str | None = None

    @field_validator(
        "description",
        "visible_action",
        "performance",
        "camera_behavior",
        "environmental_motion",
        "transition_intent",
    )
    @classmethod
    def reject_compiler_syntax(cls, value: str | None) -> str | None:
        return _reject_h3_compiler_syntax(value)


class H3CreativePlan(BaseModel):
    """Only the creative fields needed to enrich an existing scene plan for H3."""

    model_config = ConfigDict(extra="forbid")

    creative_intent: str
    style_opening: str | None = None
    shots: list[H3CreativeShot] = Field(default_factory=list)
    overall_soundscape: str
    music_intent: MusicIntent
    non_diegetic_music: str | None = None

    @field_validator(
        "creative_intent",
        "style_opening",
        "overall_soundscape",
        "non_diegetic_music",
    )
    @classmethod
    def reject_compiler_syntax(cls, value: str | None) -> str | None:
        return _reject_h3_compiler_syntax(value)

    @model_validator(mode="after")
    def validate_music(self) -> "H3CreativePlan":
        if self.music_intent == MusicIntent.NONE:
            self.non_diegetic_music = None
        elif not self.non_diegetic_music:
            raise ValueError("A non_diegetic_music description is required when music is enabled")
        return self


def _reject_h3_compiler_syntax(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if re.search(r"<(?:Subject|Picture|Video|Audio)\s+\d+>", text, re.IGNORECASE):
        raise ValueError("H3 creative fields must not contain compiler-owned reference labels")
    if re.search(r"\[Shot\s+\d+\]|\b\d{2}:\d{2}(?:[:.]\d{2,3})\b", text, re.IGNORECASE):
        raise ValueError("H3 creative fields must not contain compiler-owned shot syntax or timecodes")
    return value


class CreativeShotPayload(BaseModel):
    """Backend-neutral creative decisions consumed by a deterministic compiler."""

    model_config = ConfigDict(extra="forbid")

    shot_id: str = Field(min_length=1)
    visible_action: str = Field(min_length=1)
    performance: str = Field(min_length=1)
    camera_behavior: str | None = None
    environmental_motion: str | None = None
    transition_intent: str | None = None

    @model_validator(mode="after")
    def reject_backend_syntax(self) -> "CreativeShotPayload":
        fields = (
            self.visible_action,
            self.performance,
            self.camera_behavior,
            self.environmental_motion,
            self.transition_intent,
        )
        for value in fields:
            text = str(value or "")
            if "<picture " in text.lower() or "<audio " in text.lower() or "<video " in text.lower():
                raise ValueError("creative shot fields must not contain backend reference labels")
            if re.search(r"\b\d{2}:\d{2}(?:[:.]\d{2,3})\b", text):
                raise ValueError("creative shot fields must not contain timecodes")
        return self


class PromptPlan(BaseModel):
    creative_intent: str
    style_opening: str | None = None
    subjects: list[PlannedSubject] = Field(default_factory=list)
    reference_usage: list[ReferenceUsage] = Field(default_factory=list)
    shots: list[PlannedShot] = Field(default_factory=list)
    continuation_intents: list[ContinuationIntent] = Field(default_factory=list)
    overall_soundscape: str
    music_intent: MusicIntent
    non_diegetic_music: str | None = None
    alignment_instruction: str | None = None

    @model_validator(mode="after")
    def validate_music(self) -> PromptPlan:
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
        text = f"{self.label} is {self.name}, {self.description}"
        if self.source_references:
            text += " in " + ", ".join(self.source_references) + "."
        return text


class ResolvedPromptPlan(BaseModel):
    creative_intent: str
    style_opening: str | None = None
    subjects: list[SubjectDefinition] = Field(default_factory=list)
    reference_usage: list[ReferenceUsage] = Field(default_factory=list)
    shots: list[PlannedShot] = Field(default_factory=list)
    continuation_intents: list[ContinuationIntent] = Field(default_factory=list)
    overall_soundscape: str
    music_intent: MusicIntent
    non_diegetic_music: str | None = None
    alignment_instruction: str | None = None


class H3PromptSections(BaseModel):
    """Backend-neutral section contract returned by the H3 planning pass.

    This object deliberately contains content fields, not rendered H3 syntax.
    Reference labels and timing remain structured values until the deterministic
    compiler assembles the final prompt.
    """

    model_config = ConfigDict(extra="forbid")

    creative_intent: str
    style_opening: str | None = None
    subjects: list[SubjectDefinition] = Field(default_factory=list)
    reference_usage: list[ReferenceUsage] = Field(default_factory=list)
    shots: list[PlannedShot] = Field(default_factory=list)
    continuation_intents: list[ContinuationIntent] = Field(default_factory=list)
    overall_soundscape: str
    music_intent: MusicIntent
    non_diegetic_music: str | None = None
    alignment_instruction: str | None = None

    @classmethod
    def from_plan(cls, plan: ResolvedPromptPlan) -> "H3PromptSections":
        return cls.model_validate(plan.model_dump())

    def to_plan(self) -> ResolvedPromptPlan:
        return ResolvedPromptPlan.model_validate(self.model_dump())


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
    reference_definitions: list[str] = []
    summary: str
    retention_analysis: list[RetentionAnalysis]
    detailed_description: str
    overall_soundscape: str
    non_diegetic_music: str | None = None
    audio_subject_bindings: list[AudioSubjectBinding] = Field(default_factory=list)

    def render(self) -> str:
        subjects = "\n".join(item.render() for item in self.subject_definitions)
        reference_defs = "\n".join(
            str(item).strip() for item in self.reference_definitions if str(item).strip()
        )
        definitions = "\n".join(part for part in (subjects, reference_defs) if part)
        retention = "\n".join(item.render() for item in self.retention_analysis)
        bindings = "\n".join(
            f"{item.audio_label} ({item.stem}) -> {item.subject_label}"
            f"{f' ({item.speaker_id})' if item.speaker_id else ''}"
            for item in self.audio_subject_bindings
        )
        detailed = self.detailed_description.strip()
        if bindings:
            detailed += "\nAudio subject bindings:\n" + bindings
        return "\n\n".join([
            f"subject_definitions:\n{definitions}",
            f"summary: {self.summary.strip()}",
            f"retention_analysis:\n{retention}",
            "detailed_description: " + detailed,
            "overall_soundscape: " + self.overall_soundscape.strip(),
            "non_diegetic_music: " + (self.non_diegetic_music.strip() if self.non_diegetic_music else "N/A"),
        ])


class GeneratedVideoPrompt(BaseModel):
    mode: PromptMode
    prompt: BaseVideoPrompt | ReferenceVideoPrompt
    plan: ResolvedPromptPlan
    references: list[ResolvedReference]
    judge: PromptJudgeResult | None = None
    judge_attempts: list[PromptJudgeResult] = Field(default_factory=list)

    @property
    def rendered_prompt(self) -> str:
        return self.prompt.render()
