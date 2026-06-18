"""Compatibility facade; prefer adapters.comfyui_video_backend for new imports."""

from __future__ import annotations

from adapters.comfyui_video_backend import ComfyUIVideoBackend, AudioWindowSpec


class LTXVideoRenderer(ComfyUIVideoBackend):
    """Compatibility facade for the ComfyUI video backend."""
