from __future__ import annotations

from feverslop.adapters.comfyui_minimax_h3_t2v_backend import ComfyUIMiniMaxH3T2VBackend
from feverslop.domain.h3_two_pass import H3TwoPassSchemaError, validate_i2v_frames
from feverslop.errors import FeverSlopValidationError


class ComfyUIMiniMaxH3I2VBackend(ComfyUIMiniMaxH3T2VBackend):
    """MiniMax H3 image-to-video backend with optional start/end frames.

    Unlike T2V, I2V always requires the source image (start frame); the
    end frame is optional (issue #761). Frame inputs are validated before
    ComfyUI submission in ``build_workflow``.
    """

    pipeline_name = "minimax-h3-i2v"

    def _validate_i2v_frames(self, scene: dict) -> None:
        """Validate I2V frame inputs before ComfyUI submission (issue #761)."""
        try:
            validate_i2v_frames(
                super()._resolve_start_frame(scene),
                self._resolve_end_frame(scene),
            )
        except H3TwoPassSchemaError as exc:
            raise FeverSlopValidationError(str(exc)) from exc

    def build_workflow(self, scene: dict, *args: object, **kwargs: object) -> dict:
        """Validate I2V frame inputs before ComfyUI submission (issue #761)."""
        self._validate_i2v_frames(scene)
        return super().build_workflow(scene, *args, **kwargs)
