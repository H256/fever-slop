from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math

from feverslop.config.comfyui import ComfyUIModelOverride
from feverslop.path_utils import coerce_local_path


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096


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
        return cls(
            workflow=str(raw["workflow"]),
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

    @classmethod
    def load(cls, path: str | Path, *, required_keys: list[str] | None = None) -> "AppConfig":
        path = coerce_local_path(path)

        if not path.exists():
            if required_keys:
                missing = ", ".join(f"'{key}'" for key in required_keys)
                raise ValueError(f"App config not found at {path}; required keys absent: {missing}")
            return cls(
                llm=LLMConfig(),
                comfyui=ComfyUIConfig(),
            )

        raw = json.loads(path.read_text(encoding="utf-8"))

        if required_keys:
            missing = [key for key in required_keys if key not in raw or raw[key] is None]
            if missing:
                details = "; ".join(f"'{key}'" for key in missing)
                raise ValueError(f"Missing required config keys: {details}")

        return cls._build_config(raw)

    @classmethod
    def _build_config(cls, raw: dict) -> "AppConfig":
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

        return cls(
            llm=LLMConfig(
                base_url=llm_raw.get("base_url", "http://localhost:8080/v1"),
                model=llm_raw.get("model", "default"),
                temperature=float(llm_raw.get("temperature", 0.7)),
                max_tokens=int(llm_raw.get("max_tokens", 4096)),
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
        )
