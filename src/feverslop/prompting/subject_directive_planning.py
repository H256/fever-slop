from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import ast
from typing import Any

from feverslop.domain.subject_directives import (
    SCHEMA_VERSION,
    SubjectDirective,
    SubjectDirectivePlan,
    TemporalScope,
    SpatialRelation,
    validate_subject_directive_plan,
)


class SubjectDirectivePlanner:
    """Small adapter that keeps DSPy optional at the application boundary."""

    def __init__(self, predictor: Callable[..., Any] | None = None):
        self.predictor = predictor

    def plan(self, scene: Mapping[str, Any]) -> SubjectDirectivePlan:
        if self.predictor is None:
            return build_shared_staging_plan(scene)
        return build_shared_staging_plan(
            scene,
            generator=lambda payload: self.predictor(payload),
        )


class DspySubjectDirectivePlanner:
    """Production DSPy adapter for the shared staging pass."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        from feverslop.prompting.dspy_runtime import DspyRuntime
        from feverslop.prompting.dspy_subject_directive_signatures import (
            build_subject_directive_signature,
        )

        self.runtime = dspy_runtime or DspyRuntime.create()
        import dspy

        self.predictor = self.runtime.predict(build_subject_directive_signature(dspy))
        # A staging plan is compact. Do not inherit a project-wide generation
        # budget (which may be 65k+) and let one malformed scene consume the
        # whole response or get truncated before the JSON plan is complete.
        self.lm = self.runtime.make_lm(llm, max_tokens=4096)
        self.last_scene: dict[str, Any] | None = None
        self.last_output: Any = None
        self.last_lm_history: Any = None
        self.last_repairs: list[str] = []

    def plan(self, scene: Mapping[str, Any]) -> SubjectDirectivePlan:
        from feverslop.domain.llm_parsing import extract_json_object

        self.last_scene = dict(scene)
        try:
            with self.runtime.context(lm=self.lm):
                output = self.predictor(scene=dict(scene))
        finally:
            self.last_lm_history = list(getattr(self.lm, "history", []) or [])
        self.last_output = output
        payload = output.get("staging_plan") if isinstance(output, Mapping) else getattr(output, "staging_plan", output)
        if isinstance(payload, str):
            payload = extract_json_object(payload)
        if not isinstance(payload, Mapping):
            raise ValueError("DSPy subject planner returned no structured staging plan")
        payload = _decode_nested_json(payload)
        payload, self.last_repairs = _repair_zero_length_scopes(payload, scene)
        return build_shared_staging_plan(scene, generator=lambda _payload: payload)


def _decode_nested_json(value: Any) -> Any:
    """Normalize DSPy JSON fields that LiteLLM may deserialize only partially."""
    if isinstance(value, Mapping):
        return {key: _decode_nested_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nested_json(item) for item in value]
    if isinstance(value, str):
        candidate = value.strip()
        if candidate[:1] in "[{":
            try:
                return _decode_nested_json(json.loads(candidate))
            except json.JSONDecodeError:
                try:
                    return _decode_nested_json(ast.literal_eval(candidate))
                except (SyntaxError, ValueError):
                    pass
    return value


def _repair_zero_length_scopes(
    payload: Mapping[str, Any], scene: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Repair only the unambiguous model error of a zero-length full-shot scope."""
    duration = float(scene.get("duration_seconds") or scene.get("duration") or 1)
    result = dict(payload)
    repairs: list[str] = []

    def repair_scope(scope: Any, label: str) -> dict[str, Any] | Any:
        if not isinstance(scope, Mapping):
            return scope
        try:
            start = float(scope.get("start_seconds"))
            end = float(scope.get("end_seconds"))
        except (TypeError, ValueError):
            return scope
        if start == 0 and end == 0:
            repairs.append(f"{label}: temporal_scope 0..0s -> 0..{duration:g}s")
            return {"start_seconds": 0, "end_seconds": duration}
        return scope

    result["temporal_scope"] = repair_scope(result.get("temporal_scope"), "plan")
    subjects = []
    for item in result.get("subjects") or ():
        if not isinstance(item, Mapping):
            subjects.append(item)
            continue
        subject = dict(item)
        subject_id = str(subject.get("subject_id") or "unknown")
        subject["temporal_scope"] = repair_scope(subject.get("temporal_scope"), subject_id)
        subjects.append(subject)
    if "subjects" in result:
        result["subjects"] = subjects
    return result, repairs


def build_shared_staging_plan(
    scene: Mapping[str, Any],
    *,
    generator: Callable[[dict[str, Any]], Any] | None = None,
) -> SubjectDirectivePlan:
    """Build one shared staging result before any subject-specific prose pass.

    ``generator`` is intentionally passed the complete scene payload. A DSPy adapter
    can therefore replace the deterministic fallback without changing the contract or
    allowing independent subject calls to invent incompatible positions.
    """
    if generator is not None:
        generated = generator({"schema_version": SCHEMA_VERSION, "scene": dict(scene)})
        if isinstance(generated, SubjectDirectivePlan):
            return generated
        if isinstance(generated, Mapping):
            payload = generated.get("subject_directives", generated)
            return SubjectDirectivePlan.from_dict(payload)

    duration = float(scene.get("duration_seconds") or scene.get("duration") or 1)
    scope = TemporalScope(0, duration)
    subjects = []
    for item in scene.get("subjects") or scene.get("directives") or ():
        payload = dict(item)
        payload.setdefault("temporal_scope", scope.to_dict())
        subjects.append(SubjectDirective.from_dict(payload))
    relations = tuple(
        SpatialRelation.from_dict(item)
        for item in scene.get("spatial_relations") or ()
    )
    plan = SubjectDirectivePlan(
        shot_id=str(scene.get("shot_id") or scene.get("segment_id") or scene.get("scene") or "shot").strip(),
        temporal_scope=scope,
        subjects=tuple(subjects),
        spatial_relations=relations,
    )
    issues = validate_subject_directive_plan(plan)
    if issues:
        raise ValueError("Invalid shared staging plan: " + "; ".join(issues))
    return plan


def project_directives_to_prompt(plan: SubjectDirectivePlan) -> str:
    """Serialize the model-neutral facts into explicit, auditable prompt prose."""
    lines = [
        "Subject directives (authoritative; preserve every listed fact):",
        f"Shot {plan.shot_id}, {plan.temporal_scope.start_seconds:.2f}-{plan.temporal_scope.end_seconds:.2f}s.",
    ]
    for subject in plan.subjects:
        props = ", ".join(
            f"{binding.prop_id} ({binding.state}{': ' + binding.detail if binding.detail else ''})"
            for binding in subject.prop_bindings
        ) or "none specified"
        interactions = ", ".join(subject.interactions) or "none specified"
        gaze = subject.gaze_direction or "none specified"
        scope = subject.temporal_scope or plan.temporal_scope
        lines.append(
            f"Subject {subject.subject_id}: role={subject.role}; visibility={subject.visibility}; "
            f"cardinality={subject.cardinality}; position={subject.position}; action={subject.action}; "
            f"interactions={interactions}; gaze_direction={gaze}; props={props}; "
            f"time={scope.start_seconds:.2f}-{scope.end_seconds:.2f}s."
        )
    for relation in plan.spatial_relations:
        detail = f" ({relation.detail})" if relation.detail else ""
        lines.append(f"Relation: {relation.subject_id} {relation.relation} {relation.target_id}{detail}.")
    return "\n".join(lines)


def validate_projected_prompt(plan: SubjectDirectivePlan, prompt: str) -> None:
    """Reject prompt projections that silently lose directive facts."""
    haystack = str(prompt or "").casefold()
    missing: list[str] = []
    for subject in plan.subjects:
        required = ((subject.subject_id, "subject"), (subject.position, "position"), (subject.action, "action"))
        for value, label in required:
            if value.strip() and value.casefold() not in haystack:
                missing.append(f"{subject.subject_id} {label} '{value}'")
        for binding in subject.prop_bindings:
            value = f"{binding.prop_id} ({binding.state})"
            if value.casefold() not in haystack:
                missing.append(f"{subject.subject_id} prop '{value}'")
    for relation in plan.spatial_relations:
        value = f"{relation.subject_id} {relation.relation} {relation.target_id}"
        if value.casefold() not in haystack:
            missing.append(f"relation '{value}'")
    if missing:
        raise ValueError("Projected prompt lost subject directive coverage: " + "; ".join(missing))


def compose_directive_prompt(
    plan: SubjectDirectivePlan,
    *,
    composer: Callable[[dict[str, Any]], Any] | None = None,
) -> str:
    """Compose optional LLM prose while treating the structured plan as authority."""
    authoritative = project_directives_to_prompt(plan)
    if composer is None:
        return authoritative
    result = composer({"subject_directives": plan.to_dict(), "authoritative_prompt": authoritative})
    prompt = str(result.get("prompt") if isinstance(result, Mapping) else result or "").strip()
    validate_projected_prompt(plan, prompt)
    return prompt


def judge_directive_prompt(plan: SubjectDirectivePlan, prompt: str) -> list[str]:
    """Return actionable deterministic diagnostics for a semantic judge boundary."""
    try:
        validate_projected_prompt(plan, prompt)
    except ValueError as exc:
        return [str(exc)]
    return []


def subject_directives_from_scene(scene: Mapping[str, Any]) -> SubjectDirectivePlan | None:
    payload = scene.get("subject_directives")
    if payload is None:
        return None
    if isinstance(payload, SubjectDirectivePlan):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("subject_directives must be an object")
    plan = SubjectDirectivePlan.from_dict(payload)
    issues = validate_subject_directive_plan(plan)
    if issues:
        raise ValueError("Invalid subject directives: " + "; ".join(issues))
    return plan


__all__ = [
    "SubjectDirectivePlanner",
    "DspySubjectDirectivePlanner",
    "build_shared_staging_plan",
    "compose_directive_prompt",
    "judge_directive_prompt",
    "project_directives_to_prompt",
    "subject_directives_from_scene",
    "validate_projected_prompt",
]
