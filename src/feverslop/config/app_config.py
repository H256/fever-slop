from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from feverslop.config.comfyui import ComfyUIModelOverride
from feverslop.domain.postprocessing import FFMPEG_TIMEOUT_SECONDS
from feverslop.domain.story_plan import PLANNER_REVISION
from feverslop.domain.video_workflow_profile import VideoWorkflowProfile
from feverslop.path_utils import coerce_local_path
from feverslop.ports.reporting import parse_log_level
from feverslop.prompting.dspy_runtime import DEFAULT_TASK_TEMPERATURES
from feverslop.utils.io import read_json

_logger = logging.getLogger(__name__)


@dataclass
class StoryPlanningConfig:
    """Story-plan build/approval settings (issue #1386).

    ``require_approval`` gates a freshly built plan behind an explicit
    ``--story-plan-approve`` before concept generation. ``planner_revision``
    names the planner that produced the plan so a revision change invalidates
    cached planning state on resume.
    """

    require_approval: bool = False
    planner_revision: str = PLANNER_REVISION
    failure_policy: str = "warn"
    # Four short scene briefs keep the creative request feasible for local
    # 7B--35B models while still amortizing prompt overhead.
    acting_batch_size: int = 4


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "default"
    temperature: float = 0.7
    dspy_temperature: float = 0.4
    max_tokens: int = 4096
    request_timeout_seconds: float = 180.0
    dspy_cache: bool = False
    max_concurrent_requests: int = 1
    prompt_judge_attempts: int = 3
    prompt_judge_max_tokens: int = 8192
    prompt_judge_enabled: bool = False
    # Kept for config compatibility; H3 judges are advisory in all modes.
    prompt_judge_blocking: bool = False
    # H3 planner output budget.  0 = auto-scale by project shot count (default).
    # When set, this explicit ceiling overrides the auto-scaler.
    prompt_planner_max_tokens: int = 0
    # Enforcement policy for inferred narrative-contract violations after
    # bounded repair: "warn" (default) preserves structurally valid concepts
    # and continues with a diagnostic warning; "block" raises and aborts.
    narrative_contract_enforcement: str = "warn"
    story_planning: StoryPlanningConfig = field(default_factory=StoryPlanningConfig)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    # Per-task DSPy temperatures (planner, renderer, judge, analyzer). Always
    # resolved against DEFAULT_TASK_TEMPERATURES at parse time, so feasible
    # defaults apply when the operator leaves a task unset.
    task_temperatures: dict[str, float] = field(default_factory=dict)
    _local_api_key: str | None = field(default=None, repr=False)

    def model_for(self, task_type: str | None = None) -> str:
        """Return an optional task-profile model, falling back to the legacy model."""
        profile = str(task_type or "").strip().lower()
        return self.models.get(profile, self.model) if profile else self.model

    @property
    def api_key(self) -> str | None:
        """Resolve the process environment before persistent local config."""
        if "LLM_API_KEY" in os.environ:
            return os.environ["LLM_API_KEY"]
        return self._local_api_key


def _require_float(raw: dict, key: str) -> float:
    """Extract a required numeric config field; raises ValueError if missing or non-numeric."""
    value = raw.get(key)
    if value is None:
        raise ValueError(f"Video workflow limit requires '{key}' field")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Video workflow limit '{key}' must be a number") from exc


@dataclass(frozen=True)
class VideoWorkflowLimitConfig:
    workflow: str
    max_render_duration_seconds: float

    @classmethod
    def from_dict(cls, raw: dict) -> VideoWorkflowLimitConfig:
        raw_workflow = raw.get("workflow")
        if not isinstance(raw_workflow, str):
            raise ValueError("Video workflow limit workflow must be a string")
        workflow = raw_workflow.strip()
        duration = _require_float(raw, "max_render_duration_seconds")
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
    latent_upscaler_device: str | None = None
    # Wall-clock budget (seconds) for each FFmpeg operation in the post-processing
    # paths (scene re-encode, concat, mux). Overridable per project; missing
    # config falls back to this default (canonical value in the domain layer).
    ffmpeg_timeout_seconds: float = FFMPEG_TIMEOUT_SECONDS


class VramHandoffMode(str, Enum):
    CONTINUOUS = "continuous"
    MANUAL = "manual"


@dataclass(frozen=True)
class ExecutionConfig:
    vram_handoff: VramHandoffMode = VramHandoffMode.CONTINUOUS
    log_level: int = logging.INFO


@dataclass(frozen=True)
class StoryboardPromptTransformConfig:
    workflow: str
    kind: str = "template"
    template: str = ""
    positive_prompt_input: str = "text"
    debug_dir: str = "storyboard_prompt_debug"
    max_words: int = 150

    @classmethod
    def from_dict(cls, raw: dict) -> StoryboardPromptTransformConfig:
        raw_workflow = raw.get("workflow")
        if not isinstance(raw_workflow, str):
            raise ValueError("StoryboardPromptTransformConfig requires a non-empty workflow")
        raw_workflow = raw_workflow.strip()
        if not raw_workflow:
            raise ValueError("StoryboardPromptTransformConfig requires a non-empty workflow")
        max_words = int(raw.get("max_words", 150))
        if max_words < 1:
            raise ValueError("StoryboardPromptTransformConfig max_words must be >= 1")
        return cls(
            workflow=raw_workflow,
            kind=str(raw.get("kind", "template")),
            template=str(raw.get("template", "")),
            positive_prompt_input=str(raw.get("positive_prompt_input", "text")),
            debug_dir=str(raw.get("debug_dir", "storyboard_prompt_debug")),
            max_words=max_words,
        )



def _check_required_keys(raw: dict, required_keys: list[str]) -> None:
    """Validate required keys with dot-notation nested path support."""
    missing = []
    for key_path in required_keys:
        parts = key_path.split(".")
        node = raw
        found = True
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                found = False
                break
            node = node[part]
        if not found or node is None:
            missing.append(key_path)
    if missing:
        details = "; ".join(f"'{key}'" for key in missing)
        raise ValueError(f"Missing required config keys: {details}")


@dataclass
class AppConfig:
    llm: LLMConfig
    comfyui: ComfyUIConfig
    global_library_path: Path = field(default_factory=lambda: (Path.home() / ".feverslop" / "library").expanduser())
    storyboard_prompt_transforms: list[StoryboardPromptTransformConfig] = field(default_factory=list)
    video_workflow_profiles: tuple[VideoWorkflowProfile, ...] = field(default_factory=tuple)
    _video_workflow_profile_defaults: tuple[tuple[str, str, str], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    import_store: Any = field(default=None, repr=False)

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
                            f"{pipeline}/{purpose}",
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
        default_profile = next(
            profile
            for profile in self.video_workflow_profiles
            if profile.name == default_name
        )
        override_path = self._imported_workflow_path(pipeline, purpose)
        if override_path is None:
            return default_profile
        return replace(default_profile, workflow_path=override_path)

    def _imported_workflow_path(self, pipeline: str, purpose: str) -> str | None:
        """Return the active import's snapshot path for pipeline/purpose, if any."""
        store = self.import_store
        if store is None:
            return None
        active = store.find_active(pipeline=pipeline, purpose=purpose)
        if active is None:
            return None
        return str(store.snapshot_path(active.profile_id))

    def _attach_import_store(self, project_dir: Path | None) -> None:
        """Attach the per-project import store for precedence.

        Layout: ``projects_root/<project>/workflows/``. The store is a no-op
        (returns None) when no active import exists, so this is safe to run
        on every load. Re-anchoring is allowed (overwrites the store) so
        render-path call sites can correct the anchor from the config-file
        directory to the actual project directory.
        """
        if project_dir is None:
            return
        project_dir = project_dir.resolve()
        workflows = project_dir / "workflows"
        if not workflows.is_dir():
            return
        from feverslop.domain.workflow_import_store import WorkflowImportStore

        self.import_store = WorkflowImportStore(
            projects_root=project_dir.parent,
        ).for_project(project_dir.name)

    def attach_import_store(self, project_dir: Path | None) -> None:
        """Re-anchor the import store to the actual project directory.

        Called from render-path call sites where the project dir is known
        but the config file lives at repo root (the common case).
        """
        self._attach_import_store(project_dir)

    @classmethod
    def load(cls, path: str | Path, *, required_keys: list[str] | None = None) -> AppConfig:
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

        raw = read_json(path)

        if required_keys:
            _check_required_keys(raw, required_keys)

        config = cls._build_config(raw, dotenv_api_key=dotenv_api_key, base_dir=path.parent)
        config._attach_import_store(path.parent)
        return config

    @classmethod
    def _build_config(
        cls,
        raw: dict,
        *,
        dotenv_api_key: str | None = None,
        base_dir: Path | None = None,
    ) -> AppConfig:
        llm_raw = raw.get("llm", {})
        comfyui_raw = raw.get("comfyui", {})
        execution_raw = raw.get("execution", {})
        if not isinstance(execution_raw, dict):
            raise ValueError("execution must be an object")

        raw_vram_handoff = execution_raw.get("vram_handoff", "continuous")
        try:
            vram_handoff = VramHandoffMode(raw_vram_handoff)
        except (TypeError, ValueError):
            raise ValueError(
                "execution.vram_handoff must be 'continuous' or 'manual'",
            ) from None
        raw_log_level = execution_raw.get("log_level", "info")
        if not isinstance(raw_log_level, (str, int)):
            raise ValueError("execution.log_level must be a string or integer")
        try:
            log_level = parse_log_level(raw_log_level)
        except ValueError as exc:
            raise ValueError(f"execution.log_level: {exc}") from None
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
        prompt_timeout = float(comfyui_raw.get("prompt_timeout_seconds", 1800.0))
        if not math.isfinite(prompt_timeout) or prompt_timeout <= 0:
            raise ValueError("comfyui.prompt_timeout_seconds must be greater than zero")

        raw_latent_upscaler_device = comfyui_raw.get("latent_upscaler_device")
        if raw_latent_upscaler_device is None:
            latent_upscaler_device: str | None = None
        elif not isinstance(raw_latent_upscaler_device, str):
            raise ValueError("comfyui.latent_upscaler_device must be a string")
        else:
            latent_upscaler_device = raw_latent_upscaler_device.strip()
            if latent_upscaler_device not in ("cuda", "rocm", "cpu", "auto"):
                raise ValueError(
                    "comfyui.latent_upscaler_device must be 'cuda', 'rocm', 'cpu', or 'auto'"
                )

        raw_ffmpeg_timeout = comfyui_raw.get("ffmpeg_timeout_seconds")
        if raw_ffmpeg_timeout is None:
            ffmpeg_timeout_seconds = FFMPEG_TIMEOUT_SECONDS
        else:
            if isinstance(raw_ffmpeg_timeout, bool) or not isinstance(raw_ffmpeg_timeout, (int, float)):
                raise ValueError("comfyui.ffmpeg_timeout_seconds must be a number")
            ffmpeg_timeout_seconds = float(raw_ffmpeg_timeout)
            if not math.isfinite(ffmpeg_timeout_seconds) or ffmpeg_timeout_seconds <= 0:
                raise ValueError("comfyui.ffmpeg_timeout_seconds must be greater than zero")

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

        llm_temperature = float(llm_raw.get("temperature", 0.7))
        llm_max_tokens = int(llm_raw.get("max_tokens", 4096))
        llm_max_concurrent_requests = int(llm_raw.get("max_concurrent_requests", 1))
        llm_prompt_judge_attempts = int(llm_raw.get("prompt_judge_attempts", 3))
        llm_prompt_judge_max_tokens = int(llm_raw.get("prompt_judge_max_tokens", 8192))
        llm_prompt_judge_enabled = _parse_bool(
            llm_raw.get("prompt_judge_enabled", False),
            "llm.prompt_judge_enabled",
        )
        llm_prompt_judge_blocking = _parse_bool(
            llm_raw.get("prompt_judge_blocking", False),
            "llm.prompt_judge_blocking",
        )
        llm_prompt_planner_max_tokens = int(llm_raw.get("prompt_planner_max_tokens", 0))
        if llm_prompt_planner_max_tokens < 0:
            raise ValueError("llm.prompt_planner_max_tokens must be >= 0")
        llm_narrative_enforcement = str(
            llm_raw.get("narrative_contract_enforcement", "warn")
        ).strip().lower()
        if llm_narrative_enforcement not in ("warn", "block"):
            raise ValueError(
                "llm.narrative_contract_enforcement must be 'warn' or 'block'"
            )
        story_planning_raw = llm_raw.get("story_planning")
        if story_planning_raw is None:
            story_planning_raw = raw.get("story_planning", {})
        if story_planning_raw is None:
            story_planning_raw = {}
        if not isinstance(story_planning_raw, dict):
            raise ValueError("story_planning must be an object")
        story_planning_require_approval = _parse_bool(
            story_planning_raw.get("require_approval", False),
            "story_planning.require_approval",
        )
        story_planning_planner_revision = str(
            story_planning_raw.get("planner_revision", PLANNER_REVISION)
        ).strip()
        if not story_planning_planner_revision:
            raise ValueError("story_planning.planner_revision must be a non-empty string")
        story_planning_failure_policy = str(
            story_planning_raw.get("failure_policy", "warn")
        ).strip().lower()
        if story_planning_failure_policy not in ("warn", "block"):
            raise ValueError("story_planning.failure_policy must be 'warn' or 'block'")
        try:
            story_planning_acting_batch_size = int(
                story_planning_raw.get("acting_batch_size", 4)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("story_planning.acting_batch_size must be an integer") from exc
        if story_planning_acting_batch_size < 1:
            raise ValueError("story_planning.acting_batch_size must be >= 1")
        llm_chat_template_kwargs_raw = llm_raw.get("chat_template_kwargs", {})
        if not isinstance(llm_chat_template_kwargs_raw, dict):
            raise ValueError("llm.chat_template_kwargs must be an object")
        llm_chat_template_kwargs: dict[str, Any] = dict(llm_chat_template_kwargs_raw)
        llm_models_raw = llm_raw.get("models", {})
        if not isinstance(llm_models_raw, dict):
            raise ValueError("llm.models must be an object")
        llm_models: dict[str, str] = {}
        for raw_profile, raw_model in llm_models_raw.items():
            profile = str(raw_profile).strip().lower()
            model = str(raw_model).strip()
            if not profile or not model:
                raise ValueError("llm.models requires non-empty profile and model names")
            if profile in llm_models:
                raise ValueError(f"Duplicate llm.models profile: {profile}")
            llm_models[profile] = model
        llm_task_temperatures_raw = llm_raw.get("task_temperatures", {})
        if not isinstance(llm_task_temperatures_raw, dict):
            raise ValueError("llm.task_temperatures must be an object")
        # Start from the feasible defaults so an unset task still resolves;
        # operator overrides win per task.
        llm_task_temperatures: dict[str, float] = dict(DEFAULT_TASK_TEMPERATURES)
        for raw_task, raw_value in llm_task_temperatures_raw.items():
            task = str(raw_task).strip().lower()
            if not task:
                raise ValueError("llm.task_temperatures requires non-empty task names")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"llm.task_temperatures.{task} must be a number") from exc
            if value < 0:
                raise ValueError(f"llm.task_temperatures.{task} must be >= 0, got {value}")
            llm_task_temperatures[task] = value
            if task not in DEFAULT_TASK_TEMPERATURES:
                # Unknown task names are stored but never looked up by make_lm;
                # warn so a typo (e.g. "planer") does not silently no-op.
                _logger.warning(
                    "llm.task_temperatures.%s is not a recognized H3 task "
                    "(expected one of %s); the override will be ignored",
                    task,
                    ", ".join(sorted(DEFAULT_TASK_TEMPERATURES)),
                )
        if llm_temperature < 0:
            raise ValueError(f"llm.temperature must be >= 0, got {llm_temperature}")
        if llm_max_tokens <= 0:
            raise ValueError(f"llm.max_tokens must be > 0, got {llm_max_tokens}")
        if llm_max_concurrent_requests <= 0:
            raise ValueError("llm.max_concurrent_requests must be > 0")
        if llm_prompt_judge_attempts <= 0:
            raise ValueError("llm.prompt_judge_attempts must be > 0")
        if llm_prompt_judge_max_tokens <= 0:
            raise ValueError("llm.prompt_judge_max_tokens must be > 0")
        library_raw = raw.get("global_library_path")
        if library_raw is None and isinstance(raw.get("global_library"), dict):
            library_raw = raw["global_library"].get("path")
        library_path = Path(str(library_raw or (Path.home() / ".feverslop" / "library"))).expanduser()
        if not library_path.is_absolute() and base_dir is not None:
            library_path = base_dir / library_path
        return cls(
            llm=LLMConfig(
                base_url=llm_raw.get("base_url", "http://localhost:8080/v1"),
                model=llm_raw.get("model", "default"),
                temperature=llm_temperature,
                dspy_temperature=float(llm_raw.get("dspy_temperature", 0.4)),
                max_tokens=llm_max_tokens,
                request_timeout_seconds=float(llm_raw.get("request_timeout_seconds", 180.0)),
                dspy_cache=_parse_bool(llm_raw.get("dspy_cache", False), "llm.dspy_cache"),
                max_concurrent_requests=llm_max_concurrent_requests,
                prompt_judge_attempts=llm_prompt_judge_attempts,
                prompt_judge_max_tokens=llm_prompt_judge_max_tokens,
                prompt_judge_enabled=llm_prompt_judge_enabled,
                prompt_judge_blocking=llm_prompt_judge_blocking,
                prompt_planner_max_tokens=llm_prompt_planner_max_tokens,
                narrative_contract_enforcement=llm_narrative_enforcement,
                story_planning=StoryPlanningConfig(
                    require_approval=story_planning_require_approval,
                    planner_revision=story_planning_planner_revision,
                    failure_policy=story_planning_failure_policy,
                    acting_batch_size=story_planning_acting_batch_size,
                ),
                chat_template_kwargs=llm_chat_template_kwargs,
                models=llm_models,
                task_temperatures=llm_task_temperatures,
                _local_api_key=_optional_secret(llm_raw.get("api_key")) or dotenv_api_key,
            ),
            comfyui=ComfyUIConfig(
                base_url=comfyui_raw.get("base_url", "http://127.0.0.1:8188"),
                prompt_timeout_seconds=prompt_timeout,
                model_overrides=[
                    ComfyUIModelOverride.from_dict(item)
                    for item in comfyui_raw.get("model_overrides") or []
                ],
                default_max_render_duration_seconds=default_max_render_duration,
                video_workflow_limits=video_workflow_limits,
                latent_upscaler_device=latent_upscaler_device,
                ffmpeg_timeout_seconds=ffmpeg_timeout_seconds,
            ),
            execution=ExecutionConfig(vram_handoff=vram_handoff, log_level=log_level),
            global_library_path=library_path.resolve(),
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
    "duration_capability",
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
                "Unknown video workflow profile fields: " + ", ".join(unknown_fields),
            )
        missing_fields = sorted(
            _VIDEO_WORKFLOW_PROFILE_FIELDS
            - {
                "satisfies_final_output",
                "supports_start_frame",
                "default",
                "duration_capability",
            }
            - set(raw_profile),
        )
        if missing_fields:
            raise ValueError(
                "Missing video workflow profile fields: " + ", ".join(missing_fields),
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
            duration_capability=raw_profile.get("duration_capability"),
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
                    f"{profile.pipeline}/{profile.purpose}",
                )
            defaults[key] = profile.name

    return tuple(profiles), tuple(
        (pipeline, purpose, name)
        for (pipeline, purpose), name in defaults.items()
    )
