from __future__ import annotations


class FeverSlopError(Exception):
    """Base exception for all FeverSlop errors."""


class FeverSlopLMLError(FeverSlopError):
    """Error from LLM interaction (parsing, HTTP, retry exhausted)."""


class FeverSlopRenderError(FeverSlopError):
    """Error from rendering pipeline (ComfyUI, FFmpeg, workflow)."""


class FeverSlopConfigError(FeverSlopError):
    """Error from project config loading or validation."""


class FeverSlopWorkflowError(FeverSlopRenderError):
    """Error from ComfyUI workflow patching or validation."""


class FeverSlopValidationError(FeverSlopError):
    """Error from input or parameter validation."""


class FeverSlopAdaptationError(FeverSlopError):
    """Error from adapter layer (third-party exception wrapper)."""


class FeverSlopDataError(FeverSlopError):
    """Error from missing or invalid data in pipeline input artifacts."""
