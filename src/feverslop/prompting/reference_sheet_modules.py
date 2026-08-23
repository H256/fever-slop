from __future__ import annotations

from typing import Any

from feverslop.domain.reference_sheet import ReferenceSheetPlan
from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.reference_sheet_signatures import (
    build_reference_sheet_signature,
)


def _value(result: Any, name: str) -> Any:
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, result)


class ReferenceSheetPlanningModules:
    """DSPy boundary for structured reference-sheet planning."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy reference-sheet planning requires a configured LLM")
        if dspy_runtime is None:
            import dspy

            dspy_runtime = DspyRuntime.create(dspy)
        else:
            import dspy

        self._lm = dspy_runtime.make_lm(llm)
        self._context = dspy_runtime.context
        self._predictor = dspy_runtime.predict(build_reference_sheet_signature(dspy))

    def plan(self, *, kind: str, description: str, asset_context: dict[str, Any]) -> ReferenceSheetPlan:
        with self._context(lm=self._lm):
            result = self._predictor(kind=kind, description=description, asset_context=asset_context)
        plan = _value(result, "plan")
        if plan is None:
            raise ValueError("DSPy reference-sheet planning returned no plan")
        if hasattr(plan, "model_dump"):
            return ReferenceSheetPlan.model_validate(plan.model_dump())
        if isinstance(plan, dict):
            return ReferenceSheetPlan.model_validate(plan)
        return ReferenceSheetPlan.model_validate(vars(plan))
