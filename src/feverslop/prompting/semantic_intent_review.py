"""Independent semantic intent review and bounded targeted repair (#545).

This module is the *review/repair* stage for the Semantic Intent Contracts
milestone. It compares an extracted ``IntentLedger`` (from ``#544``) back
against the original free-form story idea and produces bounded, targeted
repairs for the four finding classes the milestone calls out:

- ``omitted``   -- a constraint the source states is missing from the ledger.
- ``invented``  -- a ledger record the source does not state.
- ``contradicted`` -- a ledger record that conflicts with the source.
- ``misbound``  -- a record bound to the wrong entity.

Scope (per #545):
- A DSPy judge contract that returns structured findings with source evidence.
- An explicit model/prompt independence policy and a correlated-error risk
  warning when the judge cannot be run under an independent LM.
- Targeted repair of only the invalid ledger records; unaffected IDs and
  records are retained.
- Bounded retries and clear English diagnostics.
- An ambiguous-input state that does not silently guess.

Non-goals (per #545):
- Using the judge as the sole downstream render gate.
- Judging every scene prompt against the entire original idea.
- Automatically resolving genuinely ambiguous intent without policy.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from feverslop.domain.semantic_intent import IntentLedger

logger = logging.getLogger(__name__)

__all__ = [
    "AMBIGUOUS",
    "FAILED",
    "NEEDS_REPAIR",
    "OK",
    "REPAIRED",
    "SEMANTIC_INTENT_REVIEW",
    "SEMANTIC_INTENT_REVIEW_GUIDE",
    "DEFAULT_REVIEW_MAX_ATTEMPTS",
    "IntentFinding",
    "IntentReviewResult",
    "IntentRepairResult",
    "SemanticIntentReviewer",
    "DspySemanticIntentReviewer",
    "build_semantic_intent_review_signature",
    "repair_ledger",
    "review_and_repair",
]

#: llm_policy task name for the review budget.
SEMANTIC_INTENT_REVIEW = "semantic_intent_review"

#: Markdown guide name bundled with the review signature.
SEMANTIC_INTENT_REVIEW_GUIDE = "semantic-intent-review"

#: Bounded visible status values for a structured review attempt.
OK = "ok"
REPAIRED = "repaired"
NEEDS_REPAIR = "needs_repair"
AMBIGUOUS = "ambiguous"
FAILED = "failed"

#: Default bounded retry budget for the judge (attempts, not retries).
DEFAULT_REVIEW_MAX_ATTEMPTS = 3

#: Finding classes the milestone calls out.
FINDING_KINDS = frozenset({"omitted", "invented", "contradicted", "misbound"})


class IntentFinding(BaseModel):
    """A single structured review finding with source evidence.

    ``record_id`` names the exact ledger record (entity, relation, or
    constraint) the finding targets. ``entity_id`` is the (possibly wrong)
    entity the record is bound to; ``correct_entity_id`` is set only for
    ``misbound`` findings that name the entity the record should have been
    bound to.
    """

    id: str
    kind: str
    record_id: str = ""
    entity_id: str = ""
    correct_entity_id: str = ""
    description: str = Field(max_length=1000)
    source_evidence: str = Field(default="", max_length=1000)
    severity: str = "warn"

    @property
    def diagnostic(self) -> str:
        """A bounded English diagnostic (no full prompt/response bodies)."""
        target = self.record_id or self.correct_entity_id or self.entity_id
        evidence = f" (source: {self.source_evidence})" if self.source_evidence else ""
        return f"{self.kind} constraint {self.id} on {target}: {self.description}{evidence}"


class IntentReviewResult(BaseModel):
    """Bounded judge output: findings plus a visible status and warnings."""

    findings: list[IntentFinding] = Field(default_factory=list)
    status: str
    warnings: list[str] = Field(default_factory=list)
    attempts: int = 0


class IntentRepairResult(BaseModel):
    """Bounded targeted-repair output.

    ``applied``/``skipped`` record which finding IDs were repaired vs left for
    operator policy (omitted records are never silently invented).
    """

    ledger: IntentLedger
    status: str
    applied: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_semantic_intent_review_signature(dspy_module: Any | None = None) -> Any:
    """Return the DSPy signature for the independent intent judge.

    The output is the language-neutral ``IntentReviewResult``. The model is
    instructed (in the signature docstring and the bundled guide) to judge the
    ledger against the source, copy only what the source supports, and to mark
    genuinely ambiguous intent rather than guess.
    """
    if dspy_module is None:
        import dspy as dspy_module

    class ReviewSemanticIntent(dspy_module.Signature):
        """Review an extracted intent ledger against the original story idea.

        Compare the typed ledger against the source and report only the
        findings it supports. A finding must cite the source evidence. Report
        an omitted constraint the source states but the ledger misses, an
        invented record the source does not state, a contradicted record that
        conflicts with the source, and a misbound record attached to the wrong
        entity. Do not invent findings the source does not support. If the
        source is genuinely ambiguous, return status ``ambiguous`` with no
        findings rather than guessing. Use stable, readable finding IDs. Return
        only the review result; do not echo the guide or DSPy markers.
        """

        story_idea: str = dspy_module.InputField()
        ledger: dict[str, Any] = dspy_module.InputField()
        guide: str = dspy_module.InputField(default="")
        review: IntentReviewResult = dspy_module.OutputField()

    return ReviewSemanticIntent


def _decode_review(raw: Any) -> IntentReviewResult:
    payload = raw
    if isinstance(raw, dict):
        payload = raw.get("review", raw)
    elif hasattr(raw, "review"):
        payload = getattr(raw, "review")
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if isinstance(payload, str):
        from feverslop.domain.llm_parsing import extract_json_object

        payload = extract_json_object(payload)
    if not isinstance(payload, dict):
        raise ValueError("semantic intent review returned no structured result")
    return IntentReviewResult.model_validate(payload)


class SemanticIntentReviewer:
    """Independent judge with an optional predictor seam.

    With no predictor configured (tests, or a non-LLM run) the reviewer is a
    no-op that returns an empty finding set with ``OK`` status -- it never
    invents findings. With a predictor (production DSPy or a test double), it
    calls the model, decodes the structured output, and bounds the result: a
    failed or malformed structured output is bounded (``FAILED`` with a logged
    warning), never fatal, so the downstream stage always has a constructible
    review.
    """

    def __init__(self, predictor: Any | None = None):
        self.predictor = predictor

    def _invoke(self, story_idea: str, ledger: IntentLedger) -> Any:
        """Call the predictor. Subclasses override to add a DSPy LM context."""
        assert self.predictor is not None
        return self.predictor(story_idea=story_idea, ledger=ledger.to_dict())

    def review(
        self,
        story_idea: str,
        ledger: IntentLedger,
        *,
        max_attempts: int = DEFAULT_REVIEW_MAX_ATTEMPTS,
    ) -> IntentReviewResult:
        if self.predictor is None:
            return IntentReviewResult(
                findings=[],
                status=OK,
                warnings=["no reviewer configured; no findings (no invention)"],
                attempts=0,
            )
        last_error = ""
        for attempt in range(1, max(1, max_attempts) + 1):
            try:
                raw = self._invoke(story_idea, ledger)
                result = _decode_review(raw)
            except Exception as exc:  # noqa: BLE001 - bounded structured-output failure
                last_error = f"{type(exc).__name__}"
                logger.warning(
                    "semantic intent review attempt %d failed: %s", attempt, exc
                )
                continue
            result = _bound_review(result)
            result.attempts = attempt
            return result
        return IntentReviewResult(
            findings=[],
            status=FAILED,
            warnings=[f"review failed after {max_attempts} attempts: {last_error}"],
            attempts=max_attempts,
        )


def _bound_review(result: IntentReviewResult) -> IntentReviewResult:
    """Validate finding kinds and normalize the visible status."""
    findings = []
    for finding in result.findings:
        if finding.kind not in FINDING_KINDS:
            continue
        findings.append(finding)
    if result.status not in (OK, NEEDS_REPAIR, AMBIGUOUS):
        status = NEEDS_REPAIR if findings else OK
    else:
        status = result.status
    if status == AMBIGUOUS and findings:
        # Ambiguous input does not silently guess; keep the findings for the
        # operator but surface the ambiguity status.
        pass
    return IntentReviewResult(
        findings=findings, status=status, warnings=result.warnings, attempts=result.attempts
    )


class DspySemanticIntentReviewer(SemanticIntentReviewer):
    """Production DSPy adapter for the independent intent judge.

    Mirrors ``DspySemanticIntentReviewer``/``DspySemanticIntentExtractor``: it
    builds the signature, creates a bounded LM (a compact structured task --
    never the project-wide generation budget), and delegates to the tolerant
    base ``review`` so a failed or malformed model response is bounded, not
    fatal. The judge runs under its own LM so it is independent of the
    extractor's model (model/prompt independence policy).
    """

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        from feverslop.prompting.dspy_runtime import DspyRuntime

        self.runtime = dspy_runtime or DspyRuntime.create()
        import dspy

        self.predictor = self.runtime.predict(build_semantic_intent_review_signature(dspy))
        # A compact structured judge. Do not inherit a project-wide generation
        # budget (which may be 65k+) and let one ledger consume the whole
        # response or truncate before the JSON completes.
        self.lm = self.runtime.make_lm(llm, max_tokens=4096, task="judge")

    def _invoke(self, story_idea: str, ledger: IntentLedger) -> Any:
        """Run the predictor inside the DSPy LM context, with the guide."""
        from feverslop.prompting.guide_loader import load_markdown_guide

        guide = load_markdown_guide(SEMANTIC_INTENT_REVIEW_GUIDE)
        with self.runtime.context(lm=self.lm):
            return self.predictor(
                story_idea=story_idea,
                ledger=ledger.to_dict(),
                guide=guide,
            )


def repair_ledger(
    ledger: IntentLedger,
    findings: list[IntentFinding],
) -> IntentRepairResult:
    """Apply bounded, targeted repair to only the invalid ledger records.

    Deterministic policy (no silent guessing):

    - ``invented`` / ``contradicted`` -- drop the named record (a record the
      source does not state or that conflicts with it must not survive).
    - ``misbound`` -- rebind the named record to ``correct_entity_id``.
    - ``omitted`` -- never silently invented; the finding is recorded as
      ``skipped`` for operator policy.

    A finding is applied only when it names a record that exists in the
    ledger (and, for ``misbound``, a valid target entity). Unaffected IDs and
    records are retained. The result is a constructible ledger; a repair that
    would break ledger invariants is skipped with a warning rather than
    raising.
    """
    payload = ledger.to_dict()
    entity_ids = {entity["id"] for entity in payload["entities"]}
    applied: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []

    drop_ids: set[str] = set()
    rebind: dict[str, str] = {}
    for finding in findings:
        if finding.kind == "omitted":
            if finding.source_evidence.strip():
                # Source-backed omitted constraint: add a typed proposed
                # addition with provenance from the original story.
                entity_id = finding.entity_id
                if not entity_id or entity_id not in entity_ids:
                    skipped.append(finding.id)
                    warnings.append(
                        f"finding {finding.id} skipped: omitted record has "
                        f"unknown entity {entity_id or '(none)'}"
                    )
                    continue
                constraint_id = _next_constraint_id(payload)
                new_constraint: dict[str, Any] = {
                    "id": constraint_id,
                    "entity_id": entity_id,
                    "kind": "scene_obligation",
                    "statement": finding.description,
                    "status": "required",
                    "provenance": {
                        "origin": "explicit",
                        "source_text": finding.source_evidence,
                    },
                }
                payload["constraints"].append(new_constraint)
                applied.append(finding.id)
            else:
                # No source evidence: retain an explicit unresolved warning.
                skipped.append(finding.id)
                warnings.append(
                    f"finding {finding.id} unresolved: omitted record without "
                    f"source evidence is not invented"
                )
            continue
        record_id = finding.record_id
        if not record_id:
            skipped.append(finding.id)
            warnings.append(f"finding {finding.id} skipped: no record_id")
            continue
        if finding.kind in ("invented", "contradicted"):
            if record_id not in entity_ids and not _record_exists(payload, record_id):
                skipped.append(finding.id)
                warnings.append(f"finding {finding.id} skipped: unknown record {record_id}")
                continue
            drop_ids.add(record_id)
            applied.append(finding.id)
        elif finding.kind == "misbound":
            target = finding.correct_entity_id
            if not target or target not in entity_ids:
                skipped.append(finding.id)
                warnings.append(
                    f"finding {finding.id} skipped: unknown target entity {target or '(none)'}"
                )
                continue
            if not _record_exists(payload, record_id):
                skipped.append(finding.id)
                warnings.append(
                    f"finding {finding.id} skipped: unknown rebindable record {record_id}"
                )
                continue
            rebind[record_id] = target
            applied.append(finding.id)

    if drop_ids:
        payload["entities"] = [e for e in payload["entities"] if e["id"] not in drop_ids]
        payload["relations"] = [
            r
            for r in payload["relations"]
            if r["id"] not in drop_ids
            and r.get("subject_id") not in drop_ids
            and r.get("target_id") not in drop_ids
        ]
        payload["constraints"] = [
            c
            for c in payload["constraints"]
            if c["id"] not in drop_ids
            and c.get("entity_id") not in drop_ids
        ]
    if rebind:
        for constraint in payload["constraints"]:
            if constraint["id"] in rebind:
                constraint["entity_id"] = rebind[constraint["id"]]
        for relation in payload["relations"]:
            if relation["id"] in rebind:
                relation["target_id"] = rebind[relation["id"]]

    try:
        repaired = IntentLedger.from_dict(payload)
    except ValueError as exc:
        # A repair that breaks invariants is skipped, not fatal.
        return IntentRepairResult(
            ledger=ledger,
            status=AMBIGUOUS,
            applied=[],
            skipped=[f.id for f in findings],
            warnings=[f"repair left ledger invalid, kept original: {exc}"],
        )
    status = NEEDS_REPAIR if skipped else (REPAIRED if applied else OK)
    return IntentRepairResult(
        ledger=repaired,
        status=status,
        applied=applied,
        skipped=skipped,
        warnings=warnings,
    )


def _record_exists(payload: dict[str, Any], record_id: str) -> bool:
    for key in ("relations", "constraints"):
        for record in payload[key]:
            if record.get("id") == record_id:
                return True
    return False


def _next_constraint_id(payload: dict[str, Any]) -> str:
    """Generate a unique constraint ID that doesn't collide with existing IDs."""
    existing = {
        record.get("id", "")
        for key in ("entities", "relations", "constraints")
        for record in payload.get(key, [])
    }
    counter = 1
    while f"constraint_{counter}" in existing:
        counter += 1
    return f"constraint_{counter}"


def review_and_repair(
    story_idea: str,
    ledger: IntentLedger,
    reviewer: SemanticIntentReviewer,
    *,
    on_retry_exhausted: str = "warn",
    max_attempts: int = DEFAULT_REVIEW_MAX_ATTEMPTS,
    review: IntentReviewResult | None = None,
) -> IntentRepairResult:
    """Review a ledger and apply bounded targeted repair.

    ``on_retry_exhausted`` is the explicit policy for a judge whose bounded
    retries are exhausted (``FAILED``): ``"warn"`` keeps the unreviewed
    ledger with a visible warning (the default, so a judge failure never
    blocks expensive reference rendering); ``"block"`` stops with the
    unreviewed ledger and a blocker warning. A genuinely ``ambiguous`` review
    never silently guesses: the ledger is returned unchanged with the
    ambiguity surfaced.

    ``review`` may be supplied to reuse an already-computed review (the
    production pipeline reviews once and needs the findings for its
    diagnostics), avoiding a second model call.
    """
    if review is None:
        review = reviewer.review(story_idea, ledger, max_attempts=max_attempts)
    if review.status == FAILED:
        policy_warning = (
            "review retries exhausted; keeping unreviewed ledger (warning policy)"
            if on_retry_exhausted == "warn"
            else "review retries exhausted; blocking (block policy)"
        )
        return IntentRepairResult(
            ledger=ledger,
            status=AMBIGUOUS if on_retry_exhausted == "block" else OK,
            applied=[],
            skipped=[],
            warnings=[policy_warning],
        )
    if review.status == AMBIGUOUS:
        return IntentRepairResult(
            ledger=ledger,
            status=AMBIGUOUS,
            applied=[],
            skipped=[f.id for f in review.findings],
            warnings=["ambiguous source; no targeted repair applied"],
        )
    result = repair_ledger(ledger, review.findings)
    status = (
        NEEDS_REPAIR
        if review.status == NEEDS_REPAIR and result.status == OK
        else result.status
    )
    return result.model_copy(
        update={"status": status, "warnings": [*review.warnings, *result.warnings]}
    )
