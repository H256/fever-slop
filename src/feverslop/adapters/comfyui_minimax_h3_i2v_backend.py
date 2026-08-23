from __future__ import annotations

from feverslop.adapters.comfyui_minimax_h3_t2v_backend import ComfyUIMiniMaxH3T2VBackend


class ComfyUIMiniMaxH3I2VBackend(ComfyUIMiniMaxH3T2VBackend):
    """MiniMax H3 image-to-video backend with optional start/end frames."""

    pipeline_name = "minimax-h3-i2v"
