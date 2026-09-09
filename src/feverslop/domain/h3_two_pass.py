from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Mapping


class H3TwoPassSchemaError(ValueError):
    """Raised when an H3 two-pass contract is invalid."""


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
    budgets = {
        "draft": (12, 4, 0.55),
        "standard": (20, 8, 0.40),
        "final": (28, 12, 0.30),
    }
    try:
        pass1_steps, pass2_steps, pass2_denoise = budgets[level]
    except KeyError as exc:
        raise H3TwoPassSchemaError("quality must be draft, standard, or final") from exc
    anchors = ["#PROMPT", "#FRAMECOUNT", "#PASS1", "#PASS2"]
    if audio:
        anchors.append("#AUDIO_LATENT")
    return H3TwoPassSpec.create(
        model_assets=["minimax_h3", "minimax_h3_video_vae"],
        pass1_sampler="res_multistep",
        pass1_scheduler="simple",
        pass1_steps=pass1_steps,
        pass1_denoise=1.0,
        pass2_sampler="res_multistep",
        pass2_scheduler="simple",
        pass2_steps=pass2_steps,
        pass2_denoise=pass2_denoise,
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
