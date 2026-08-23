"""Compatibility facade; prefer feverslop.adapters.comfyui_video_backend for new imports."""

from __future__ import annotations

from feverslop.adapters.comfyui_video_backend import (
    AudioWindowSpec,
    ComfyUIVideoBackend,
)

__all__ = ["AudioWindowSpec", "LTXVideoRenderer"]


class LTXVideoRenderer(ComfyUIVideoBackend):
    """Compatibility facade for the ComfyUI video backend."""
