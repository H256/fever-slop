from __future__ import annotations

from typing import Any


def build_subject_directive_signature(dspy_module: Any | None = None) -> Any:
    """Return the shared staging signature used by optional DSPy planners."""
    if dspy_module is None:
        import dspy as dspy_module

    class BuildSharedStagingPlan(dspy_module.Signature):
        """Resolve one coherent shot space before generating subject prose.

        The output must retain every visible subject, action, position, relation,
        prop binding/state, visibility/cardinality, and temporal scope from the input.
        Do not make independent subject decisions outside the shared staging plan.
        """

        scene: dict[str, Any] = dspy_module.InputField()
        staging_plan: dict[str, Any] = dspy_module.OutputField()

    return BuildSharedStagingPlan


__all__ = ["build_subject_directive_signature"]
