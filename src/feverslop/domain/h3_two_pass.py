from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


class H3TwoPassSchemaError(ValueError):
    """Raised when an H3 two-pass contract is invalid."""


class I2VFrameMode(str, Enum):
    """Anchor modes supported by the H3 I2V two-pass workflow (issue #761)."""

    START_ONLY = "start_only"
    START_END = "start_end"


def validate_i2v_frames(
    start_frame: str | Path | None,
    end_frame: str | Path | None = None,
) -> I2VFrameMode:
    """Validate I2V frame inputs before ComfyUI submission.

    I2V always requires a start frame (the source image). An end frame is
    optional: its presence selects ``START_END``, its absence ``START_ONLY``.
    """
    if start_frame is None:
        raise H3TwoPassSchemaError("i2v requires a start frame")
    if end_frame is None:
        return I2VFrameMode.START_ONLY
    return I2VFrameMode.START_END


# Calibrated two-pass budgets for each quality profile. This is the single
# source of truth consumed by workflow preparation (see _patch_two_pass_budget)
# and by the machine-readable snapshot in tests/fixtures/h3_quality_profiles.json.
# Resolution budgets are targets recorded for each profile; the concrete
# resolution is user-driven via --resolution, while the sampling budget
# (steps/denoise) and the latent-upscale scale are applied per selected quality.
H3_QUALITY_BUDGETS: Mapping[str, Mapping[str, float]] = {
    "draft": {
        "pass1_steps": 12, "pass1_denoise": 1.0,
        "pass2_steps": 4, "pass2_denoise": 0.55,
        "pass1_megapixels": 0.4, "pass2_max_megapixels": 1.6,
    },
    "standard": {
        "pass1_steps": 20, "pass1_denoise": 1.0,
        "pass2_steps": 8, "pass2_denoise": 0.40,
        "pass1_megapixels": 0.6, "pass2_max_megapixels": 2.4,
    },
    "final": {
        "pass1_steps": 28, "pass1_denoise": 1.0,
        "pass2_steps": 12, "pass2_denoise": 0.30,
        "pass1_megapixels": 0.8, "pass2_max_megapixels": 3.2,
    },
}


def h3_quality_budgets() -> dict[str, dict[str, float]]:
    """Return a copy of the calibrated per-quality two-pass budgets."""
    return {name: dict(budget) for name, budget in H3_QUALITY_BUDGETS.items()}


def quality_from_render_profile(render_profile: Any) -> str:
    """Extract the quality tier from a render profile string.

    Render profiles are shaped like ``ltx25-r2v-draft``; the final hyphen
    segment is the quality tier. Unknown or missing tiers fall back to
    ``draft`` so a malformed profile never breaks workflow preparation.
    """
    value = str(render_profile or "").strip().lower()
    if not value:
        return "draft"
    tier = value.rsplit("-", 1)[-1]
    return tier if tier in H3_QUALITY_BUDGETS else "draft"


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise H3TwoPassSchemaError(f"{field} must be a positive integer")
    return value


def _denoise(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise H3TwoPassSchemaError(f"{field} must be between 0 and 1")
    resolved = float(value)
    if not isfinite(resolved) or not 0 < resolved <= 1:
        raise H3TwoPassSchemaError(f"{field} must be between 0 and 1")
    return resolved


def _names(values: Any, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise H3TwoPassSchemaError(f"{field} must be an iterable of names")
    try:
        result = tuple(sorted({str(value).strip() for value in values}))
    except TypeError as exc:
        raise H3TwoPassSchemaError(f"{field} must be an iterable of names") from exc
    if not result or any(not value for value in result):
        raise H3TwoPassSchemaError(f"{field} cannot contain blank names")
    return result


@dataclass(frozen=True)
class H3TwoPassSpec:
    model_assets: tuple[str, ...]
    pass1_sampler: str
    pass1_scheduler: str
    pass1_steps: int
    pass1_denoise: float
    pass2_sampler: str
    pass2_scheduler: str
    pass2_steps: int
    pass2_denoise: float
    preserve_audio_latent: bool
    required_anchors: tuple[str, ...]

    @classmethod
    def create(cls, **values: Any) -> H3TwoPassSpec:
        model_assets = _names(values.get("model_assets"), "model_assets")
        required_anchors = _names(values.get("required_anchors"), "required_anchors")
        if "#PASS1" not in required_anchors or "#PASS2" not in required_anchors:
            raise H3TwoPassSchemaError("required_anchors must include #PASS1 and #PASS2")
        if "#PASS3" in required_anchors:
            raise H3TwoPassSchemaError("three-pass workflows are not supported")
        samplers = {}
        for field in ("pass1_sampler", "pass1_scheduler", "pass2_sampler", "pass2_scheduler"):
            value = str(values.get(field) or "").strip().lower()
            if not value:
                raise H3TwoPassSchemaError(f"{field} is required")
            samplers[field] = value
        preserve = values.get("preserve_audio_latent")
        if type(preserve) is not bool:
            raise H3TwoPassSchemaError("preserve_audio_latent must be a boolean")
        return cls(
            model_assets=model_assets,
            pass1_sampler=samplers["pass1_sampler"],
            pass1_scheduler=samplers["pass1_scheduler"],
            pass1_steps=_positive_int(values.get("pass1_steps"), "pass1_steps"),
            pass1_denoise=_denoise(values.get("pass1_denoise"), "pass1_denoise"),
            pass2_sampler=samplers["pass2_sampler"],
            pass2_scheduler=samplers["pass2_scheduler"],
            pass2_steps=_positive_int(values.get("pass2_steps"), "pass2_steps"),
            pass2_denoise=_denoise(values.get("pass2_denoise"), "pass2_denoise"),
            preserve_audio_latent=preserve,
            required_anchors=required_anchors,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> H3TwoPassSpec:
        if not isinstance(payload, Mapping):
            raise H3TwoPassSchemaError("H3 two-pass spec must be a mapping")
        try:
            return cls.create(**dict(payload))
        except TypeError as exc:
            raise H3TwoPassSchemaError("H3 two-pass spec has missing fields") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_assets": list(self.model_assets),
            "pass1_sampler": self.pass1_sampler,
            "pass1_scheduler": self.pass1_scheduler,
            "pass1_steps": self.pass1_steps,
            "pass1_denoise": self.pass1_denoise,
            "pass2_sampler": self.pass2_sampler,
            "pass2_scheduler": self.pass2_scheduler,
            "pass2_steps": self.pass2_steps,
            "pass2_denoise": self.pass2_denoise,
            "preserve_audio_latent": self.preserve_audio_latent,
            "required_anchors": list(self.required_anchors),
        }

    def validate_workflow_anchors(self, available_anchors: Any) -> None:
        available = {str(anchor).strip() for anchor in available_anchors}
        missing = sorted(set(self.required_anchors) - available)
        if missing:
            raise H3TwoPassSchemaError(
                "workflow is missing required H3 two-pass anchors: " + ", ".join(missing)
            )


def default_h3_two_pass_spec(quality: str, *, audio: bool = False) -> H3TwoPassSpec:
    """Return the calibrated two-pass budget for draft, standard, or final."""
    level = str(quality).strip().lower()
    budget = H3_QUALITY_BUDGETS.get(level)
    if budget is None:
        raise H3TwoPassSchemaError("quality must be draft, standard, or final")
    anchors = ["#PROMPT", "#FRAMECOUNT", "#PASS1", "#PASS2"]
    if audio:
        anchors.append("#AUDIO_LATENT")
    return H3TwoPassSpec.create(
        model_assets=["minimax_h3", "minimax_h3_video_vae"],
        pass1_sampler="res_multistep",
        pass1_scheduler="simple",
        pass1_steps=int(budget["pass1_steps"]),
        pass1_denoise=float(budget["pass1_denoise"]),
        pass2_sampler="res_multistep",
        pass2_scheduler="simple",
        pass2_steps=int(budget["pass2_steps"]),
        pass2_denoise=float(budget["pass2_denoise"]),
        preserve_audio_latent=bool(audio),
        required_anchors=anchors,
    )


def validate_h3_two_pass_topology(
    workflow: Mapping[str, Any], spec: H3TwoPassSpec,
) -> None:
    """Validate the structural nodes required by native H3 latent refinement."""
    if not isinstance(spec, H3TwoPassSpec):
        raise TypeError("spec must be an H3TwoPassSpec")
    nodes = list(workflow.values())
    titles = {
        str(node.get("_meta", {}).get("title"))
        for node in nodes
        if node.get("_meta", {}).get("title")
    }
    spec.validate_workflow_anchors(titles)
    classes = {str(node.get("class_type", "")) for node in nodes}
    required_groups = {
        "AV latent separation": {
            "MiniMaxH3AVLatentSeparateT8",
            "MiniMaxH3AVLatentSeparate",
            "LTXVSeparateAVLatent",
        },
        "learned video latent upscale": {
            "VRGDG_MiniMaxH3LearnedLatentUpscale",
            "MiniMaxH3LatentUpscale",
            "MinimaxH3LatentUpscaler3D",
        },
        "AV latent recombination": {
            "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent",
            "MiniMaxH3ReplaceUpscaledVideoLatent",
            "LTXVConcatAVLatent",
        },
    }
    missing = [name for name, aliases in required_groups.items() if not classes.intersection(aliases)]
    if missing:
        raise H3TwoPassSchemaError("workflow is missing H3 two-pass topology: " + ", ".join(missing))
    sampler_count = sum(node.get("class_type") == "SamplerCustomAdvanced" for node in nodes)
    if sampler_count < 2:
        raise H3TwoPassSchemaError("workflow must contain separate sampler nodes for pass 1 and pass 2")
