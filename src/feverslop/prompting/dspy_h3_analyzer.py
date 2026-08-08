from __future__ import annotations

from pathlib import Path
from typing import Any

from feverslop.prompting.dspy_h3_models import ImageAnalysisMode, ReferenceAsset, ReferenceKind


class LocalImageAnalyzer:
    """Analyze local picture references only when the configured mode allows it."""

    def __init__(self, predictor: Any, mode: ImageAnalysisMode = ImageAnalysisMode.MISSING_ONLY):
        self.predictor = predictor
        self.mode = mode

    def should_analyze(self, reference: ReferenceAsset) -> bool:
        return (
            reference.kind == ReferenceKind.PICTURE
            and self.mode != ImageAnalysisMode.OFF
            and self._local_file(reference.source) is not None
            and (self.mode == ImageAnalysisMode.ALWAYS or not reference.description)
        )

    @staticmethod
    def _local_file(source: str) -> Path | None:
        path = Path(source).expanduser()
        return path.resolve() if path.is_file() else None

    def analyze(self, reference: ReferenceAsset) -> str:
        path = self._local_file(reference.source)
        if path is None:
            raise ValueError(f"Image is not a local file: {reference.source}")
        analysis = self.predictor(
            image=__import__("dspy").Image.from_path(str(path)),
            intended_role=reference.role.value,
            user_hint=reference.description or "",
        ).analysis
        return "\n".join(filter(None, [
            analysis.objective_description,
            "Visible subjects: " + "; ".join(analysis.visible_subjects) if analysis.visible_subjects else "",
            f"Environment: {analysis.environment}" if analysis.environment else "",
            f"Visual style: {analysis.visual_style}" if analysis.visual_style else "",
            f"Composition: {analysis.composition}" if analysis.composition else "",
            f"Lighting: {analysis.lighting}" if analysis.lighting else "",
            "Visible text: " + "; ".join(analysis.visible_text) if analysis.visible_text else "",
        ]))