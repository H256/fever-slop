from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from video_settings import VideoSettings


@dataclass(frozen=True)
class VideoConfig:
    fps: int = 24
    width: int = 1280
    height: int = 704


@dataclass(frozen=True)
class AudioConfig:
    demucs_model: str = "htdemucs_ft"
    whisper_model: str = "large"
    language: str = "de"


@dataclass(frozen=True)
class SceneGenerationConfig:
    min_duration: float = 2.0
    max_duration: float = 10.0
    bias: float = 0.7
    duration_preset: str = "impact_weighted"
    seed: int = 42


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
class LoraConfig:
    enabled: bool = False
    name: str = ""
    strength_model: float = 1.0
    strength_clip: float = 1.0


@dataclass(frozen=True)
class ProjectConfig:
    project_dir: Path
    project_name: str
    input_audio: Path

    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    scene_generation: SceneGenerationConfig = field(default_factory=SceneGenerationConfig)
    vocal_detection: VocalDetectionConfig = field(default_factory=VocalDetectionConfig)

    story_idea: str = ""
    style: str = ""
    subject: str = ""
    locations: list[str] = field(default_factory=list)

    steering: SteeringConfig = field(default_factory=SteeringConfig)
    prompt_guidance: PromptGuidanceConfig = field(default_factory=PromptGuidanceConfig)
    lora_1: LoraConfig = field(default_factory=LoraConfig)

    @classmethod
    def load(cls, config_path: str | Path) -> "ProjectConfig":
        config_path = Path(config_path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))

        project_dir = config_path.parent
        video_raw = raw.get("video", {})
        audio_raw = raw.get("audio", {})
        scene_raw = raw.get("scene_generation", {})
        vocal_raw = raw.get("vocal_detection", {})
        steering_raw = raw.get("steering", {})
        guidance_raw = raw.get("prompt_guidance", {})
        lora_1_raw = raw.get("lora_1", {})

        input_audio = Path(raw["input_audio"])
        if not input_audio.is_absolute():
            input_audio = project_dir / input_audio

        return cls(
            project_dir=project_dir,
            project_name=raw.get("project_name") or input_audio.stem,
            input_audio=input_audio,

            video=VideoConfig(
                fps=int(video_raw.get("fps", 24)),
                width=int(video_raw.get("width", 1280)),
                height=int(video_raw.get("height", 704)),
            ),

            audio=AudioConfig(
                demucs_model=audio_raw.get("demucs_model", "htdemucs_ft"),
                whisper_model=audio_raw.get("whisper_model", "large"),
                language=audio_raw.get("language", "de"),
            ),

            scene_generation=SceneGenerationConfig(
                min_duration=float(scene_raw.get("min_duration", 2.0)),
                max_duration=float(scene_raw.get("max_duration", 10.0)),
                bias=float(scene_raw.get("bias", 0.7)),
                duration_preset=scene_raw.get("duration_preset", "impact_weighted"),
                seed=int(scene_raw.get("seed", 42)),
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
            locations=list(raw.get("locations", [])),

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
            lora_1=LoraConfig(
                enabled=bool(lora_1_raw.get("enabled", False)),
                name=lora_1_raw.get("name", ""),
                strength_model=float(lora_1_raw.get("strength_model", 1.0)),
                strength_clip=float(lora_1_raw.get("strength_clip", 1.0)),
            ),
        )

    def to_video_settings(self) -> VideoSettings:
        return VideoSettings(
            fps=self.video.fps,
            width=self.video.width,
            height=self.video.height,
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
        ):
            directory.mkdir(parents=True, exist_ok=True)
