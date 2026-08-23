from __future__ import annotations

from dataclasses import dataclass

from feverslop.domain.subject_directives import SubjectDirectivePlan
from feverslop.prompting.subject_directive_planning import (
    project_directives_to_prompt,
    validate_projected_prompt,
)


@dataclass(frozen=True)
class DirectiveProjection:
    backend: str
    prompt: str
    supported_fields: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "prompt": self.prompt,
            "supported_fields": list(self.supported_fields),
            "diagnostics": list(self.diagnostics),
        }


_SUPPORTED_FIELDS = (
    "subject_id", "role", "position", "action", "interactions", "gaze_direction",
    "prop_bindings", "prop_state", "visibility", "cardinality", "temporal_scope",
)


def project_subject_directives(
    plan: SubjectDirectivePlan,
    *,
    backend: str,
) -> DirectiveProjection:
    """Project one contract into backend-labelled prompt data without changing facts."""
    normalized = str(backend).strip().lower()
    aliases = {
        "h3": "minimax-h3-r2v",
        "minimax": "minimax-h3-r2v",
        "ltx": "ltx-t2v",
        "msr": "ltx-msr",
        "ingredients": "ltx-ingredients",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"minimax-h3-r2v", "ltx-t2v", "ltx-msr", "ltx-ingredients"}:
        raise ValueError(f"Unsupported subject directive backend: {backend}")
    prompt = project_directives_to_prompt(plan)
    validate_projected_prompt(plan, prompt)
    diagnostics: tuple[str, ...] = ()
    if normalized == "ltx-ingredients":
        diagnostics = ("Ingredients preserves timing as best-effort static prompt guidance.",)
    return DirectiveProjection(
        backend=normalized,
        prompt=prompt,
        supported_fields=_SUPPORTED_FIELDS,
        diagnostics=diagnostics,
    )


__all__ = ["DirectiveProjection", "project_subject_directives"]
