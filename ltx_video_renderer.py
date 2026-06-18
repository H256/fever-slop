"""Compatibility facade; prefer autoprompter.adapters.comfyui_video_backend for new imports."""

from __future__ import annotations

from autoprompter.adapters.comfyui_video_backend import AudioWindowSpec, ComfyUIVideoBackend

__all__ = ["AudioWindowSpec", "LTXVideoRenderer"]


class LTXVideoRenderer(ComfyUIVideoBackend):
    """Compatibility facade for the ComfyUI video backend."""
