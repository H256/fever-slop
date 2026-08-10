from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path

from feverslop.config.comfyui import ComfyUIModelOverride
from feverslop.domain.video_workflow_profile import VideoWorkflowProfile
from feverslop.path_utils import coerce_local_path


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    request_timeout_seconds: float = 180.0
    dspy_cache: bool = False
    _local_api_key: str | None = field(default=None, repr=False)

    @property
    def api_key(self) -> str | None:
        """Resolve the process environment before persistent local config."""
        if "LLM_API_KEY" in os.environ:
            return os.environ["LLM_API_KEY"]
        return self._local_api_key


@dataclass(frozen=True)
class VideoWorkflowLimitConfig:
    workflow: str
    max_render_duration_seconds: float

    @classmethod
    def from_dict(cls, raw: dict) -> "VideoWorkflowLimitConfig":
        raw_workflow = raw.get("workflow")
        if not isinstance(raw_workflow, str):
            raise ValueError("Video workflow limit workflow must be a string")
        workflow = raw_workflow.strip()
        duration = float(raw["max_render_duration_seconds"])
        if not workflow:
            raise ValueError("Video workflow limit requires a non-empty workflow")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("max_render_duration_seconds must be greater than zero")
        return cls(workflow=workflow, max_render_duration_seconds=duration)


@dataclass
class ComfyUIConfig:
    base_url: str = "http://127.0.0.1:8188"
    prompt_timeout_seconds: float = 1800.0
    model_overrides: list[ComfyUIModelOverride] = field(default_factory=list)
    default_max_render_duration_seconds: float | None = None
    video_workflow_limits: tuple[VideoWorkflowLimitConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StoryboardPromptTransformConfig:
    workflow: str
    kind: str = "template"
    template: str = ""
    positive_prompt_input: str = "text"
    debug_dir: str = "storyboard_prompt_debug"

    @classmethod
    def from_dict(cls, raw: dict) -> "StoryboardPromptTransformConfig":
        raw_workflow = raw.get("workflow")
        if not isinstance(raw_workflow, str):
            raise ValueError("StoryboardPromptTransformConfig requires a non-empty workflow")
        raw_workflow = raw_workflow.strip()
        if not raw_workflow:
            raise ValueError("StoryboardPromptTransformConfig requires a non-empty workflow")
        return cls(
            workflow=raw_workflow,
            kind=str(raw.get("kind", "template")),
            template=str(raw.get("template", "")),
            positive_prompt_input=str(raw.get("positive_prompt_input", "text")),
            debug_dir=str(raw.get("debug_dir", "storyboard_prompt_debug")),
        )


@dataclass
class AppConfig:
    llm: LLMConfig
    comfyui: ComfyUIConfig
    storyboard_prompt_transforms: list[StoryboardPromptTransformConfig] = field(default_factory=list)
    video_workflow_profiles: tuple[VideoWorkflowProfile, ...] = field(default_factory=tuple)
    _video_workflow_profile_defaults: tuple[tuple[str, str, str], ...] = field(
        default_factory=tuple,
        repr=False,
    )

    def resolve_video_workflow_profile(
        self,
        *,
        pipeline: str,
        purpose: str,
        name: str | None = None,
    ) -> VideoWorkflowProfile | None:
        if name is not None:
            for profile in self.video_workflow_profiles:
                if profile.name == name:
                    if profile.pipeline != pipeline or profile.purpose != purpose:
                        raise ValueError(
                            f"Video workflow profile '{name}' does not match pipeline/purpose "
                            f"{pipeline}/{purpose}"
                        )
                    return profile
            raise ValueError(f"Unknown video workflow profile: {name}")

        default_name = next(
            (
                profile_name
                for default_pipeline, default_purpose, profile_name
                in self._video_workflow_profile_defaults
                if default_pipeline == pipeline and default_purpose == purpose
            ),
            None,
        )
        if default_name is None:
            return None
        return next(
            profile
            for profile in self.video_workflow_profiles
            if profile.name == default_name
        )

    @classmethod
    def load(cls, path: str | Path, *, required_keys: list[str] | None = None) -> "AppConfig":
        path = coerce_local_path(path)
        dotenv_api_key = _read_dotenv_value(path.parent / ".env", "LLM_API_KEY")

        if not path.exists():
            if required_keys:
                missing = ", ".join(f"'{key}'" for key in required_keys)
                raise ValueError(f"App config not found at {path}; required keys absent: {missing}")
            return cls(
                llm=LLMConfig(_local_api_key=dotenv_api_key),
                comfyui=ComfyUIConfig(),
            )

        raw = json.loads(path.read_text(encoding="utf-8"))

        if required_keys:
            missing = [key for key in required_keys if key not in raw or raw[key] is None]
            if missing:
                details = "; ".join(f"'{key}'" for key in missing)
                raise ValueError(f"Missing required config keys: {details}")

        return cls._build_config(raw, dotenv_api_key=dotenv_api_key)

    @classmethod
    def _build_config(cls, raw: dict, *, dotenv_api_key: str | None = None) -> "AppConfig":
        llm_raw = raw.get("llm", {})
        comfyui_raw = raw.get("comfyui", {})
        default_max_render_duration_raw = comfyui_raw.get("default_max_render_duration_seconds")
        default_max_render_duration = (
            None
            if default_max_render_duration_raw is None
            else float(default_max_render_duration_raw)
        )
        if default_max_render_duration is not None and (
            not math.isfinite(default_max_render_duration) or default_max_render_duration <= 0
        ):
            raise ValueError("default_max_render_duration_seconds must be greater than zero")

        video_workflow_limits = tuple(
            VideoWorkflowLimitConfig.from_dict(item)
            for item in comfyui_raw.get("video_workflow_limits") or []
        )
        workflow_basenames: set[str] = set()
        for item in video_workflow_limits:
            workflow_basename = Path(item.workflow).name.casefold()
            if workflow_basename in workflow_basenames:
                raise ValueError(f"Duplicate video workflow limit: {Path(item.workflow).name}")
            workflow_basenames.add(workflow_basename)

        video_workflow_profiles, video_workflow_profile_defaults = (
            _parse_video_workflow_profiles(raw.get("video_workflow_profiles", []))
        )

        return cls(
            llm=LLMConfig(
                base_url=llm_raw.get("base_url", "http://localhost:8080/v1"),
                model=llm_raw.get("model", "default"),
                temperature=float(llm_raw.get("temperature", 0.7)),
                max_tokens=int(llm_raw.get("max_tokens", 4096)),
                request_timeout_seconds=float(llm_raw.get("request_timeout_seconds", 180.0)),
                dspy_cache=_parse_bool(llm_raw.get("dspy_cache", False), "llm.dspy_cache"),
                _local_api_key=_optional_secret(llm_raw.get("api_key")) or dotenv_api_key,
            ),
            comfyui=ComfyUIConfig(
                base_url=comfyui_raw.get("base_url", "http://127.0.0.1:8188"),
                prompt_timeout_seconds=float(comfyui_raw.get("prompt_timeout_seconds", 1800.0)),
                model_overrides=[
                    ComfyUIModelOverride.from_dict(item)
                    for item in comfyui_raw.get("model_overrides") or []
                ],
                default_max_render_duration_seconds=default_max_render_duration,
                video_workflow_limits=video_workflow_limits,
            ),
            storyboard_prompt_transforms=[
                StoryboardPromptTransformConfig.from_dict(item)
                for item in raw.get("storyboard_prompt_transforms", [])
            ],
            video_workflow_profiles=video_workflow_profiles,
            _video_workflow_profile_defaults=video_workflow_profile_defaults,
        )


def _optional_secret(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("llm.api_key must be a string")
    return value.strip() or None


def _parse_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _read_dotenv_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        if not separator or key.strip() != name:
            continue
        value = _strip_dotenv_comment(raw_value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value or None
    return None


def _strip_dotenv_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            continue
        if character == "#" and quote is None and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value


_VIDEO_WORKFLOW_PROFILE_FIELDS = frozenset({
    "name",
    "pipeline",
    "workflow",
    "purpose",
    "stages",
    "output_scale",
    "supports_per_pass_loras",
    "supports_start_frame",
    "satisfies_final_output",
    "default",
})


def _parse_video_workflow_profiles(
    raw_profiles,
) -> tuple[tuple[VideoWorkflowProfile, ...], tuple[tuple[str, str, str], ...]]:
    if not isinstance(raw_profiles, list):
        raise ValueError("video_workflow_profiles must be a list")

    profiles: list[VideoWorkflowProfile] = []
    names: set[str] = set()
    defaults: dict[tuple[str, str], str] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise ValueError("Each video workflow profile must be an object")
        unknown_fields = sorted(set(raw_profile) - _VIDEO_WORKFLOW_PROFILE_FIELDS)
        if unknown_fields:
            raise ValueError(
                "Unknown video workflow profile fields: " + ", ".join(unknown_fields)
            )
        missing_fields = sorted(
            _VIDEO_WORKFLOW_PROFILE_FIELDS
            - {"satisfies_final_output", "supports_start_frame", "default"}
            - set(raw_profile)
        )
        if missing_fields:
            raise ValueError(
                "Missing video workflow profile fields: " + ", ".join(missing_fields)
            )

        is_default = raw_profile.get("default", False)
        if type(is_default) is not bool:
            raise ValueError("Video workflow profile default must be a boolean")
        profile = VideoWorkflowProfile.create(
            name=raw_profile["name"],
            pipeline=raw_profile["pipeline"],
            workflow_path=raw_profile["workflow"],
            purpose=raw_profile["purpose"],
            stages=raw_profile["stages"],
            output_scale=raw_profile["output_scale"],
            supports_per_pass_loras=raw_profile["supports_per_pass_loras"],
            satisfies_final_output=raw_profile.get("satisfies_final_output"),
            supports_start_frame=raw_profile.get("supports_start_frame", False),
        )
        if profile.name in names:
            raise ValueError(f"Duplicate video workflow profile name: {profile.name}")
        names.add(profile.name)
        profiles.append(profile)

        if is_default:
            key = (profile.pipeline, profile.purpose)
            if key in defaults:
                raise ValueError(
                    "Multiple default video workflow profiles for "
                    f"{profile.pipeline}/{profile.purpose}"
                )
            defaults[key] = profile.name

    return tuple(profiles), tuple(
        (pipeline, purpose, name)
        for (pipeline, purpose), name in defaults.items()
    )
