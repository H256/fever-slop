from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from feverslop.config.video_settings import VideoSettings
from feverslop.path_utils import coerce_local_path

SCENE_PROMPT_WORD_COUNT_MIN = 40
SCENE_PROMPT_WORD_COUNT_MAX = 50


@dataclass(frozen=True)
class VideoConfig:
    fps: int = 24
    width: int = 1280
    height: int = 704


@dataclass(frozen=True)
class ProjectWorkflowConfig:
    video: str | None = None
    reference_hero: str | None = None
    reference_edit: str | None = None
    reference_sequence: str | None = None


@dataclass(frozen=True)
class UpscaleConfig:
    enabled: bool = False
    workflow_path: str | None = "workflows/video_seedvr2_3b_api.json"
    model: str | None = "seedvr2_3b_int8_convrot.safetensors"
    vae: str | None = "seedvr2_ema_vae_fp16.safetensors"
    target_width: int | None = None
    target_height: int | None = None
    default_scale: float = 2.0
    strategy: str = "auto"
    max_pass_scale: float = 2.0
    max_ai_passes: int = 3
    denoise: float = 0.35
    temporal_overlap: int = 4
    split_latent: bool = True
    vae_temporal_size: int = 64
    vae_temporal_overlap: int = 8
    segment_duration_seconds: float = 4.0
    color_correction: str = "none"
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("upscale enabled must be a boolean")
        if self.enabled:
            self.validate_resources()
        if self.strategy not in {"auto", "single"}:
            raise ValueError("upscale strategy must be 'auto' or 'single'")
        if self.max_pass_scale <= 1.0:
            raise ValueError("upscale max_pass_scale must be greater than 1")
        if self.max_ai_passes < 1:
            raise ValueError("upscale max_ai_passes must be positive")
        if not 0.0 <= self.denoise <= 1.0:
            raise ValueError("upscale denoise must be between 0 and 1")
        if self.temporal_overlap < 0:
            raise ValueError("upscale temporal_overlap must not be negative")
        if self.vae_temporal_size < 8:
            raise ValueError("upscale vae_temporal_size must be at least 8")
        if self.vae_temporal_overlap < 4 or self.vae_temporal_overlap > self.vae_temporal_size:
            raise ValueError("upscale vae_temporal_overlap must be between 4 and vae_temporal_size")
        if self.segment_duration_seconds <= 0:
            raise ValueError("upscale segment_duration_seconds must be positive")
        if self.color_correction not in {"lab", "wavelet", "adain", "none"}:
            raise ValueError("upscale color_correction must be lab, wavelet, adain, or none")

    def validate_resources(self) -> None:
        for key in ("workflow_path", "model", "vae"):
            if not getattr(self, key):
                raise ValueError(f"upscale {key} is required when enabled")


@dataclass(frozen=True)
class ReferenceImagesConfig:
    width: int | None = None
    height: int | None = None

    def resolve(self, video: VideoConfig) -> tuple[int, int]:
        return (
            self.width if self.width is not None else video.width,
            self.height if self.height is not None else video.height,
        )


@dataclass(frozen=True)
class AudioConfig:
    demucs_model: str = "htdemucs_6s"
    whisper_model: str = "large-v3"
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
    word_count_min: int = SCENE_PROMPT_WORD_COUNT_MIN
    word_count_max: int = SCENE_PROMPT_WORD_COUNT_MAX

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
    reference_mode: str = "empty_environment"


@dataclass(frozen=True)
class GlobalAssetConfig:
    asset_id: str
    look_id: str = "default"
    role: str = ""

    def __post_init__(self) -> None:
        if not str(self.asset_id).strip():
            raise ValueError("global asset_id is required")
        if not str(self.look_id).strip():
            raise ValueError("global look_id is required")


def _load_global_assets(raw, field_name: str) -> tuple[GlobalAssetConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array")
    result = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be objects")
        if item.get("asset_id") is None:
            raise ValueError(f"{field_name} asset_id is required")
        result.append(GlobalAssetConfig(
            asset_id=str(item.get("asset_id", "")).strip(),
            look_id=str(item.get("look_id", "default") or "default").strip(),
            role=str(item.get("role", "") or "").strip(),
        ))
    return tuple(result)


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool = False
    name: str = ""
    strength_model: float = 1.0
    strength_clip: float = 1.0
    name_explicit: bool = False
    strength_model_explicit: bool = False
    strength_clip_explicit: bool = False


VALID_AUDIO_REF_STEMS = frozenset({"vocals", "drums", "bass", "other", "full_mix"})


@dataclass(frozen=True)
class AudioRefsConfig:
    stems: list[str] = field(default_factory=lambda: ["vocals", "full_mix"])


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
            reference_mode=str(raw.get("reference_mode", "empty_environment") or "empty_environment").strip(),
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


def _ensure_dict(value, key: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _ensure_list(value, key: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _optional_workflow_path(raw: dict, key: str) -> str | None:
    if key not in raw:
        return None
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"workflows.{key} must be a non-empty string")
    return value.strip()


def _optional_upscale_string(raw: dict, key: str, default: str) -> str | None:
    if key not in raw:
        return default
    value = raw[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"upscale.{key} must be a non-empty string or null")
    return value.strip()


@dataclass(frozen=True)
class ProjectConfig:
    project_dir: Path
    project_name: str
    input_audio: Path
    silent_mode: bool = False
    lyrics: str = ""

    video: VideoConfig = field(default_factory=VideoConfig)
    workflows: ProjectWorkflowConfig = field(default_factory=ProjectWorkflowConfig)
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    reference_images: ReferenceImagesConfig = field(default_factory=ReferenceImagesConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    scene_generation: SceneGenerationConfig = field(default_factory=SceneGenerationConfig)
    vocal_detection: VocalDetectionConfig = field(default_factory=VocalDetectionConfig)

    story_idea: str = ""
    style: str = ""
    music_style: str = ""
    subject: str = ""
    locations: list[str] = field(default_factory=list)
    actors: tuple[ActorConfig, ...] = field(default_factory=tuple)
    structured_locations: tuple[StructuredLocationConfig, ...] = field(default_factory=tuple)
    global_cast: tuple[GlobalAssetConfig, ...] = field(default_factory=tuple)
    global_locations: tuple[GlobalAssetConfig, ...] = field(default_factory=tuple)
    global_styles: tuple[GlobalAssetConfig, ...] = field(default_factory=tuple)
    global_props: tuple[GlobalAssetConfig, ...] = field(default_factory=tuple)
    subject_mode: str = "multi"
    max_scene_actors: int = 4

    steering: SteeringConfig = field(default_factory=SteeringConfig)
    prompt_guidance: PromptGuidanceConfig = field(default_factory=PromptGuidanceConfig)
    lora_1: LoraConfig = field(default_factory=LoraConfig)
    loras: tuple[LoraConfig, ...] = field(default_factory=tuple)
    lora_split_enabled: bool = False
    video_pipeline: str = "ltx_i2v"
    reference_generation: str = "image_views"
    minimax_h3_audio_refs: AudioRefsConfig = field(default_factory=AudioRefsConfig)

    @classmethod
    def load(cls, config_path: str | Path) -> ProjectConfig:
        config_path = coerce_local_path(config_path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))

        project_dir = config_path.parent
        video_raw = _ensure_dict(raw.get("video", {}), "video")
        workflows_raw = _ensure_dict(raw.get("workflows", {}), "workflows")
        upscale_raw = _ensure_dict(raw.get("upscale", {}), "upscale")
        reference_images_raw = _ensure_dict(raw.get("reference_images", {}), "reference_images")
        audio_raw = _ensure_dict(raw.get("audio", {}), "audio")
        scene_raw = _ensure_dict(raw.get("scene_generation", {}), "scene_generation")
        vocal_raw = _ensure_dict(raw.get("vocal_detection", {}), "vocal_detection")
        steering_raw = _ensure_dict(raw.get("steering", {}), "steering")
        guidance_raw = _ensure_dict(raw.get("prompt_guidance", {}), "prompt_guidance")
        lora_1_raw = _ensure_dict(raw.get("lora_1", {}), "lora_1")
        loras_raw = raw.get("loras")
        actors_raw = _ensure_list(raw.get("actors", []), "actors")
        locations_raw = _ensure_list(raw.get("locations", []), "locations")
        audio_refs_raw = _ensure_dict(raw.get("minimax_h3_audio_refs", {}), "minimax_h3_audio_refs")
        global_raw = _ensure_dict(raw.get("global_assets", {}), "global_assets")
        unknown_global_keys = set(global_raw) - {"cast", "locations", "styles", "props"}
        if unknown_global_keys:
            valid_global_keys = "cast, locations, props, styles"
            unknown_keys = ", ".join(sorted(unknown_global_keys))
            raise ValueError(
                f"global_assets contains unknown key(s): {unknown_keys}; "
                f"valid keys are: {valid_global_keys}"
            )

        input_audio_raw = raw.get("input_audio")
        if not isinstance(input_audio_raw, str):
            raise ValueError("Project config requires an 'input_audio' string field")
        # Blank is the established "no audio" sentinel for movie projects
        # (studio/project_validation.py exempts it; movie_pipeline.py treats it as absent).
        input_audio = coerce_local_path(input_audio_raw, base_dir=project_dir)
        _validate_numeric_fields(video_raw, ("fps", "width", "height"))
        valid_upscale_keys = {
            "enabled",
            "workflow_path",
            "model",
            "vae",
            "target_width",
            "target_height",
            "default_scale",
            "strategy",
            "max_pass_scale",
            "max_ai_passes",
            "denoise",
            "temporal_overlap",
            "split_latent",
            "vae_temporal_size",
            "vae_temporal_overlap",
            "segment_duration_seconds",
            "color_correction",
            "seed",
        }
        unknown_upscale_keys = set(upscale_raw) - valid_upscale_keys
        if unknown_upscale_keys:
            unknown_keys = ", ".join(sorted(unknown_upscale_keys))
            valid_keys = ", ".join(sorted(valid_upscale_keys))
            raise ValueError(
                f"upscale contains unknown key(s): {unknown_keys}; valid keys are: {valid_keys}"
            )
        enabled_raw = upscale_raw.get("enabled", False)
        if type(enabled_raw) is not bool:
            raise ValueError("upscale enabled must be a boolean")
        _validate_numeric_fields(reference_images_raw, ("width", "height"))
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
        video_pipeline = str(raw.get("video_pipeline", "ltx_i2v") or "ltx_i2v").strip()
        reference_generation = str(raw.get("reference_generation", "image_views") or "image_views").strip()
        if reference_generation not in {"image_views", "sequence_sheet"}:
            raise ValueError("reference_generation must be 'image_views' or 'sequence_sheet'")
        max_actor_limit = 8 if video_pipeline in {"minimax-h3-r2v", "minimax-h3-i2v"} else 4
        max_scene_actors = int(raw.get("max_scene_actors", 1 if subject_mode == "single" else max_actor_limit))
        if max_scene_actors < 1 or max_scene_actors > max_actor_limit:
            raise ValueError(f"max_scene_actors must be between 1 and {max_actor_limit}")
        if subject_mode == "single":
            max_scene_actors = 1
        word_count_min = int(guidance_raw.get("word_count_min", SCENE_PROMPT_WORD_COUNT_MIN))
        word_count_max = int(guidance_raw.get("word_count_max", SCENE_PROMPT_WORD_COUNT_MAX))
        if word_count_min < 1:
            raise ValueError(f"prompt_guidance.word_count_min must be >= 1, got {word_count_min}")
        if word_count_max < 1:
            raise ValueError(f"prompt_guidance.word_count_max must be >= 1, got {word_count_max}")
        if word_count_min > word_count_max:
            raise ValueError(
                f"prompt_guidance.word_count_min ({word_count_min}) "
                f"must be <= prompt_guidance.word_count_max ({word_count_max})",
            )
        for index, item in enumerate(actors_raw):
            if not isinstance(item, dict):
                raise ValueError(f"actors[{index}] must be an object")
        actors = tuple(
            _load_actor(item, index)
            for index, item in enumerate(actors_raw, start=1)
        )

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
            workflows=ProjectWorkflowConfig(
                video=_optional_workflow_path(workflows_raw, "video"),
                reference_hero=_optional_workflow_path(workflows_raw, "reference_hero"),
                reference_edit=_optional_workflow_path(workflows_raw, "reference_edit"),
                reference_sequence=_optional_workflow_path(workflows_raw, "reference_sequence"),
            ),
            upscale=UpscaleConfig(
                enabled=enabled_raw,
                workflow_path=_optional_upscale_string(
                    upscale_raw, "workflow_path", "workflows/video_seedvr2_3b_api.json"
                ),
                model=_optional_upscale_string(
                    upscale_raw, "model", "seedvr2_3b_int8_convrot.safetensors"
                ),
                vae=_optional_upscale_string(
                    upscale_raw, "vae", "seedvr2_ema_vae_fp16.safetensors"
                ),
                target_width=(int(upscale_raw["target_width"]) if upscale_raw.get("target_width") is not None else None),
                target_height=(int(upscale_raw["target_height"]) if upscale_raw.get("target_height") is not None else None),
                default_scale=float(upscale_raw.get("default_scale", 2.0)),
                strategy=str(upscale_raw.get("strategy", "auto")),
                max_pass_scale=float(upscale_raw.get("max_pass_scale", 2.0)),
                max_ai_passes=int(upscale_raw.get("max_ai_passes", 3)),
                denoise=float(upscale_raw.get("denoise", 0.35)),
                temporal_overlap=int(upscale_raw.get("temporal_overlap", 4)),
                split_latent=bool(upscale_raw.get("split_latent", True)),
                vae_temporal_size=int(upscale_raw.get("vae_temporal_size", 64)),
                vae_temporal_overlap=int(upscale_raw.get("vae_temporal_overlap", 8)),
                segment_duration_seconds=float(upscale_raw.get("segment_duration_seconds", 4.0)),
                color_correction=str(upscale_raw.get("color_correction", "none")),
                seed=int(upscale_raw.get("seed", 0)),
            ),
            reference_images=ReferenceImagesConfig(
                width=(int(reference_images_raw["width"]) if "width" in reference_images_raw else None),
                height=(int(reference_images_raw["height"]) if "height" in reference_images_raw else None),
            ),

            audio=AudioConfig(
                demucs_model=audio_raw.get("demucs_model", "htdemucs_6s"),
                whisper_model=audio_raw.get("whisper_model", "large-v3"),
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
            music_style=raw.get("music_style", ""),
            subject=raw.get("subject", ""),
            locations=[
                str(item.get("name") if isinstance(item, dict) else item)
                for item in locations_raw
            ],
            actors=actors,
            structured_locations=tuple(
                _load_structured_location(item, index)
                for index, item in enumerate(locations_raw, start=1)
            ),
            global_cast=_load_global_assets(raw.get("global_cast", global_raw.get("cast")), "global_cast"),
            global_locations=_load_global_assets(raw.get("global_locations", global_raw.get("locations")), "global_locations"),
            global_styles=_load_global_assets(raw.get("global_styles", global_raw.get("styles")), "global_styles"),
            global_props=_load_global_assets(raw.get("global_props", global_raw.get("props")), "global_props"),
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
                word_count_min=word_count_min,
                word_count_max=word_count_max,
            ),
            video_pipeline=video_pipeline or "ltx_i2v",
            reference_generation=reference_generation,
            lora_1=lora_1,
            loras=loras,
            lora_split_enabled=bool(raw.get("lora_split_enabled", False)),
            minimax_h3_audio_refs=AudioRefsConfig(
                stems=cls._validate_stems(list(audio_refs_raw.get("stems", ["vocals", "full_mix"]))),
            ),
        )

    @staticmethod
    def _validate_stems(stems_list: list[str]) -> list[str]:
        """Validate audio ref stems against allowed set."""
        invalid = [s for s in stems_list if s not in VALID_AUDIO_REF_STEMS]
        if invalid:
            raise ValueError(
                f"Invalid audio ref stem(s): {', '.join(repr(s) for s in invalid)}. "
                f"Valid options: {', '.join(sorted(VALID_AUDIO_REF_STEMS))}",
            )
        return stems_list

    def to_video_settings(self) -> VideoSettings:
        return VideoSettings(
            fps=self.video.fps,
            width=self.video.width,
            height=self.video.height,
        )

    def apply_resolution_override(
        self, *, width: int | None = None, height: int | None = None,
    ) -> ProjectConfig:
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

    @staticmethod
    def set_resolution_on_disk(
        config_path: str | Path,
        *,
        width: int,
        height: int,
    ) -> None:
        """Patch config.json with new resolution and write it back to disk.

        Operates on the raw JSON so it does not disturb other fields.
        """
        config_path = coerce_local_path(config_path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if "video" not in raw:
            raw["video"] = {}
        raw["video"]["width"] = width
        raw["video"]["height"] = height
        config_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
    def paths(self) -> ProjectPaths:
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
    def from_config(cls, config: ProjectConfig) -> ProjectPaths:
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
