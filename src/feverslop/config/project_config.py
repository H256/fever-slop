from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json

from feverslop.config.video_settings import VideoSettings
from feverslop.path_utils import coerce_local_path


@dataclass(frozen=True)
class VideoConfig:
    fps: int = 24
    width: int = 1280
    height: int = 704


@dataclass(frozen=True)
class AudioConfig:
    demucs_model: str = "htdemucs_ft"
    whisper_model: str = "large"
    language: str = "en"


@dataclass(frozen=True)
class SceneGenerationConfig:
    min_duration: float = 2.0
    max_duration: float = 10.0
    bias: float = 0.7
    duration_preset: str = "impact_weighted"
    seed: int = -1


@dataclass(frozen=True)
class VocalDetectionConfig:
    merge_gap: float = 0.5
    min_vocal_duration: float = 0.4
    min_silence_duration: float = 0.8
    rms_low_percentile: float = 20.0
    rms_high_percentile: float = 85.0
    rms_ratio: float = 0.35
    smooth_frames: int = 10


@dataclass(frozen=True)
class SteeringConfig:
    global_: str = ""
    story_idea: str = ""
    style: str = ""
    subject: str = ""
    locations: str = ""
    concepts: str = ""
    zimage: str = ""
    ltx: str = ""
    final_prompts: str = ""


@dataclass(frozen=True)
class PromptGuidanceConfig:
    character_visibility: str = ""
    shot_types: str = ""
    environments: str = ""
    lighting: str = ""
    camera_motion: str = ""
    physical_interaction: str = ""
    facial_expression: str = ""
    outfit_rules: str = ""
    prompt_structure: str = ""
    list_handling: str = ""
    word_count_min: int = 40
    word_count_max: int = 50

    def as_prompt_context(self) -> dict:
        return {
            "character_visibility": self.character_visibility,
            "shot_types": self.shot_types,
            "environments": self.environments,
            "lighting": self.lighting,
            "camera_motion": self.camera_motion,
            "physical_interaction": self.physical_interaction,
            "facial_expression": self.facial_expression,
            "outfit_rules": self.outfit_rules,
            "prompt_structure": self.prompt_structure,
            "list_handling": self.list_handling,
            "word_count_min": self.word_count_min,
            "word_count_max": self.word_count_max,
        }


@dataclass(frozen=True)
class ActorConfig:
    id: str
    name: str
    role: str = ""
    visual_description: str = ""
    image_prompt: str = ""


@dataclass(frozen=True)
class StructuredLocationConfig:
    id: str
    name: str
    visual_description: str = ""
    image_prompt: str = ""


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool = False
    name: str = ""
    strength_model: float = 1.0
    strength_clip: float = 1.0
    name_explicit: bool = False
    strength_model_explicit: bool = False
    strength_clip_explicit: bool = False


def _load_lora_config(raw: dict) -> LoraConfig:
    return LoraConfig(
        enabled=bool(raw.get("enabled", False)),
        name=raw.get("name", ""),
        strength_model=float(raw.get("strength_model", 1.0)),
        strength_clip=float(raw.get("strength_clip", 1.0)),
        name_explicit="name" in raw and bool(str(raw.get("name", "")).strip()),
        strength_model_explicit="strength_model" in raw,
        strength_clip_explicit="strength_clip" in raw,
    )


def _load_multiline_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip()).strip()
    return str(value or "").strip()


def _safe_id(value: str, fallback: str) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    raw = "_".join(part for part in raw.split("_") if part)
    return raw or fallback


def _load_actor(raw: dict, index: int) -> ActorConfig:
    name = str(raw.get("name") or raw.get("id") or f"Actor {index}").strip()
    return ActorConfig(
        id=str(raw.get("id") or _safe_id(name, f"actor_{index}")).strip(),
        name=name,
        role=str(raw.get("role", "") or "").strip(),
        visual_description=str(raw.get("visual_description", "") or "").strip(),
        image_prompt=str(raw.get("image_prompt", "") or "").strip(),
    )


def _load_structured_location(raw, index: int) -> StructuredLocationConfig:
    if isinstance(raw, dict):
        name = str(raw.get("name") or raw.get("id") or f"Location {index}").strip()
        return StructuredLocationConfig(
            id=str(raw.get("id") or _safe_id(name, f"location_{index}")).strip(),
            name=name,
            visual_description=str(raw.get("visual_description", "") or "").strip(),
            image_prompt=str(raw.get("image_prompt", "") or "").strip(),
        )

    name = str(raw or f"Location {index}").strip()
    return StructuredLocationConfig(
        id=_safe_id(name, f"location_{index}"),
        name=name,
        visual_description=name,
        image_prompt=name,
    )


def _validate_numeric_fields(raw: dict, fields: tuple[str, ...]) -> None:
    for field_name in fields:
        if field_name in raw:
            try:
                val = int(raw[field_name])
                if val <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid '{field_name}': {raw[field_name]!r} ({exc})") from exc


@dataclass(frozen=True)
class ProjectConfig:
    project_dir: Path
    project_name: str
    input_audio: Path
    silent_mode: bool = False
    lyrics: str = ""

    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    scene_generation: SceneGenerationConfig = field(default_factory=SceneGenerationConfig)
    vocal_detection: VocalDetectionConfig = field(default_factory=VocalDetectionConfig)

    story_idea: str = ""
    style: str = ""
    subject: str = ""
    locations: list[str] = field(default_factory=list)
    actors: tuple[ActorConfig, ...] = field(default_factory=tuple)
    structured_locations: tuple[StructuredLocationConfig, ...] = field(default_factory=tuple)
    subject_mode: str = "multi"
    max_scene_actors: int = 4

    steering: SteeringConfig = field(default_factory=SteeringConfig)
    prompt_guidance: PromptGuidanceConfig = field(default_factory=PromptGuidanceConfig)
    lora_1: LoraConfig = field(default_factory=LoraConfig)
    loras: tuple[LoraConfig, ...] = field(default_factory=tuple)
    lora_split_enabled: bool = False

    @classmethod
    def load(cls, config_path: str | Path) -> "ProjectConfig":
        config_path = coerce_local_path(config_path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))

        project_dir = config_path.parent
        video_raw = raw.get("video", {})
        audio_raw = raw.get("audio", {})
        scene_raw = raw.get("scene_generation", {})
        vocal_raw = raw.get("vocal_detection", {})
        steering_raw = raw.get("steering", {})
        guidance_raw = raw.get("prompt_guidance", {})
        lora_1_raw = raw.get("lora_1", {})
        loras_raw = raw.get("loras")
        actors_raw = raw.get("actors", [])
        locations_raw = raw.get("locations", [])

        input_audio = coerce_local_path(raw["input_audio"], base_dir=project_dir)
        _validate_numeric_fields(video_raw, ("fps", "width", "height"))
        silent_mode = raw.get("silent_mode", False)
        if silent_mode is None:
            silent_mode = False
        if not isinstance(silent_mode, bool):
            raise ValueError("silent_mode must be a boolean")

        lora_1 = _load_lora_config(lora_1_raw)
        if isinstance(loras_raw, list):
            loras = tuple(
                _load_lora_config(item)
                for item in loras_raw
                if isinstance(item, dict)
            )
        else:
            loras = (lora_1,)

        subject_mode = str(raw.get("subject_mode", "multi") or "multi").strip().lower()
        if subject_mode not in {"single", "multi"}:
            raise ValueError("subject_mode must be 'single' or 'multi'")
        max_scene_actors = int(raw.get("max_scene_actors", 1 if subject_mode == "single" else 4))
        if max_scene_actors < 1 or max_scene_actors > 4:
            raise ValueError("max_scene_actors must be between 1 and 4")
        if subject_mode == "single":
            max_scene_actors = 1

        return cls(
            project_dir=project_dir,
            project_name=raw.get("project_name") or input_audio.stem,
            input_audio=input_audio,
            silent_mode=silent_mode,
            lyrics=_load_multiline_text(raw.get("lyrics", "")),

            video=VideoConfig(
                fps=int(video_raw.get("fps", 24)),
                width=int(video_raw.get("width", 1280)),
                height=int(video_raw.get("height", 704)),
            ),

            audio=AudioConfig(
                demucs_model=audio_raw.get("demucs_model", "htdemucs_ft"),
                whisper_model=audio_raw.get("whisper_model", "large"),
                language=audio_raw.get("language", "en"),
            ),

            scene_generation=SceneGenerationConfig(
                min_duration=float(scene_raw.get("min_duration", 2.0)),
                max_duration=float(scene_raw.get("max_duration", 10.0)),
                bias=float(scene_raw.get("bias", 0.7)),
                duration_preset=scene_raw.get("duration_preset", "impact_weighted"),
                seed=int(scene_raw.get("seed", -1)),
            ),

            vocal_detection=VocalDetectionConfig(
                merge_gap=float(vocal_raw.get("merge_gap", 0.5)),
                min_vocal_duration=float(vocal_raw.get("min_vocal_duration", 0.4)),
                min_silence_duration=float(vocal_raw.get("min_silence_duration", 0.8)),
                rms_low_percentile=float(vocal_raw.get("rms_low_percentile", 20.0)),
                rms_high_percentile=float(vocal_raw.get("rms_high_percentile", 85.0)),
                rms_ratio=float(vocal_raw.get("rms_ratio", 0.35)),
                smooth_frames=int(vocal_raw.get("smooth_frames", 10)),
            ),

            story_idea=raw.get("story_idea", ""),
            style=raw.get("style", ""),
            subject=raw.get("subject", ""),
            locations=[
                str(item.get("name") if isinstance(item, dict) else item)
                for item in locations_raw
            ],
            actors=tuple(
                _load_actor(item, index)
                for index, item in enumerate(actors_raw, start=1)
                if isinstance(item, dict)
            ),
            structured_locations=tuple(
                _load_structured_location(item, index)
                for index, item in enumerate(locations_raw, start=1)
            ),
            subject_mode=subject_mode,
            max_scene_actors=max_scene_actors,

            steering=SteeringConfig(
                global_=steering_raw.get("global", ""),
                story_idea=steering_raw.get("story_idea", ""),
                style=steering_raw.get("style", ""),
                subject=steering_raw.get("subject", ""),
                locations=steering_raw.get("locations", ""),
                concepts=steering_raw.get("concepts", ""),
                zimage=steering_raw.get("zimage", ""),
                ltx=steering_raw.get("ltx", ""),
                final_prompts=steering_raw.get("final_prompts", ""),
            ),
            prompt_guidance=PromptGuidanceConfig(
                character_visibility=guidance_raw.get("character_visibility", ""),
                shot_types=guidance_raw.get("shot_types", ""),
                environments=guidance_raw.get("environments", ""),
                lighting=guidance_raw.get("lighting", ""),
                camera_motion=guidance_raw.get("camera_motion", ""),
                physical_interaction=guidance_raw.get("physical_interaction", ""),
                facial_expression=guidance_raw.get("facial_expression", ""),
                outfit_rules=guidance_raw.get("outfit_rules", ""),
                prompt_structure=guidance_raw.get("prompt_structure", ""),
                list_handling=guidance_raw.get("list_handling", ""),
                word_count_min=int(guidance_raw.get("word_count_min", 40)),
                word_count_max=int(guidance_raw.get("word_count_max", 50)),
            ),
            lora_1=lora_1,
            loras=loras,
            lora_split_enabled=bool(raw.get("lora_split_enabled", False)),
        )

    def to_video_settings(self) -> VideoSettings:
        return VideoSettings(
            fps=self.video.fps,
            width=self.video.width,
            height=self.video.height,
        )

    def apply_resolution_override(
        self, *, width: int | None = None, height: int | None = None,
    ) -> "ProjectConfig":
        """Return a new ProjectConfig with overridden video resolution."""
        if width is None and height is None:
            return self
        return replace(
            self,
            video=replace(
                self.video,
                width=width if width is not None else self.video.width,
                height=height if height is not None else self.video.height,
            ),
        )

    @property
    def song_id(self) -> str:
        return self.input_audio.stem

    @property
    def output_dir(self) -> Path:
        return self.paths.output_dir

    @property
    def stems_dir(self) -> Path:
        return self.paths.stems_dir

    @property
    def timeline_dir(self) -> Path:
        return self.paths.timeline_dir

    @property
    def prompts_dir(self) -> Path:
        return self.paths.prompts_dir

    @property
    def render_dir(self) -> Path:
        return self.paths.render_dir

    @property
    def paths(self) -> "ProjectPaths":
        return ProjectPaths.from_config(self)


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path
    output_dir: Path
    stems_dir: Path
    timeline_dir: Path
    prompts_dir: Path
    render_dir: Path

    @property
    def artifact_layout(self):
        from feverslop.scene_artifacts import SceneArtifactLayout

        return SceneArtifactLayout(self.project_dir)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "ProjectPaths":
        output_dir = config.project_dir / "output"
        return cls(
            project_dir=config.project_dir,
            output_dir=output_dir,
            stems_dir=output_dir / "stems",
            timeline_dir=output_dir / "timeline",
            prompts_dir=output_dir / "prompts",
            render_dir=output_dir / "render",
        )

    def ensure_output_dirs(self) -> None:
        for directory in (
            self.output_dir,
            self.stems_dir,
            self.timeline_dir,
            self.prompts_dir,
            self.render_dir,
            self.artifact_layout.plans_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
