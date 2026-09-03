"""Pure detection helpers for the MiniMax H3 latent upscaler device.

Kept dependency-free (stdlib only) so the helpers stay unit-testable without
any ComfyUI or PySide6 context.
"""

from __future__ import annotations

import math

LATENT_UPSCALER_NODE_CLASS = "MinimaxH3LatentUpscaler3D"

_GPU_DEVICE_TYPES = ("cuda", "rocm")
_AMD_NAME_MARKERS = ("amd", "radeon")


def primary_gpu_device(stats: dict) -> dict | None:
    """Return the first GPU entry (type cuda/rocm) from a /system_stats payload.

    PyTorch ROCm builds report their GPU under ``type: "cuda"``, so the type
    filter only excludes non-GPU entries (e.g. CPUs); the AMD/NVIDIA decision
    is made from the device name. Returns None for malformed payloads.
    """
    if not isinstance(stats, dict):
        return None
    devices = stats.get("devices")
    if not isinstance(devices, list):
        return None
    for entry in devices:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type", "")).casefold() in _GPU_DEVICE_TYPES:
            return entry
    return None


def detect_upscaler_device(stats: dict) -> str | None:
    """Return ``"rocm"`` when the primary GPU name matches AMD/Radeon, else None."""
    device = primary_gpu_device(stats)
    if device is None:
        return None
    name = device.get("name")
    if not isinstance(name, str):
        return None
    lowered = name.casefold()
    if any(marker in lowered for marker in _AMD_NAME_MARKERS):
        return "rocm"
    return None


def _dropdown_values(descriptor: object) -> list[str] | None:
    """Extract option lists from the two ComfyUI ``/object_info`` input shapes:
    a wrapped list of strings (``[[...]]``) or an explicit
    ``["COMBO", {"options": [...]}]`` descriptor.

    Mirrors :meth:`ComfyUIModelResolver._dropdown_values` so the safety gate
    recognizes exactly the shapes the running server actually emits.
    """
    if (
        isinstance(descriptor, list)
        and descriptor
        and isinstance(descriptor[0], list)
        and all(isinstance(item, str) for item in descriptor[0])
    ):
        return list(descriptor[0])
    if (
        isinstance(descriptor, list)
        and len(descriptor) >= 2
        and descriptor[0] == "COMBO"
        and isinstance(descriptor[1], dict)
        and isinstance(descriptor[1].get("options"), list)
        and all(isinstance(item, str) for item in descriptor[1]["options"])
    ):
        return list(descriptor[1]["options"])
    return None


def device_input_candidates(object_info: dict | None) -> list[str] | None:
    """Allowed values for the upscaler node's ``device`` input.

    None means "cannot be determined" (missing server payload, missing node
    class, or an input shape we do not recognize); a list (possibly empty)
    means "determined".
    """
    if not isinstance(object_info, dict):
        return None
    spec = object_info.get(LATENT_UPSCALER_NODE_CLASS)
    if not isinstance(spec, dict):
        return None
    input_spec = spec.get("input")
    if not isinstance(input_spec, dict):
        return None
    for section in ("required", "optional"):
        fields = input_spec.get(section)
        if not isinstance(fields, dict):
            continue
        candidates = _dropdown_values(fields.get("device"))
        if candidates is not None:
            return candidates
    return None


def format_vram_gib(value: object) -> str | None:
    """Format a byte count as GiB (e.g. ``"24.0 GiB"``); None when the value
    is not a finite, non-negative number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return f"{value / 1024**3:.1f} GiB"
