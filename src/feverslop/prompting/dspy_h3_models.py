from __future__ import annotations

from enum import Enum

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
    relay_segments: list[dict] = Field(default_factory=list)

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