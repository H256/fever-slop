"""DSPy extraction of free-form story ideas into the semantic intent ledger (#544).

This module is the typed extraction stage for the Semantic Intent Contracts
milestone: it translates a free-form project idea (in any language) into the
versioned, language-neutral ``IntentLedger`` from ``domain.semantic_intent``.

Scope (per #544):
- A DSPy signature, module, and Markdown guide (see ``guides/semantic-intent-extraction.md``).
- Extract arbitrary entities, attributes, relations, cardinality, recurrence,
  and scene obligations.
- Preserve source-language evidence for explicit constraints.
- Do not invent gender, performers, or entity properties the source does not
  specify: with no extractor configured the result is an *empty* ledger, never
  a default singer.
- Persist the typed artifact before subject/location generation (wired in
  ``application.prompt_generation_pipeline``).

Non-goals (per #544):
- Judge/review pass.
- Per-scene prompt generation.
- Hard-coded keyword parsing as the primary extractor.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from feverslop.domain.semantic_intent import (
    IntentConstraint,
    IntentEntity,
    IntentLedger,
    IntentRelation,
    ConstraintProvenance,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXTRACTION_EMPTY",
    "EXTRACTION_FAILED",
    "EXTRACTION_OK",
    "SEMANTIC_INTENT_EXTRACTION",
    "SEMANTIC_INTENT_GUIDE",
    "DspySemanticIntentExtractor",
    "IntentExtractionResult",
    "SemanticIntentExtraction",
    "SemanticIntentExtractor",
    "build_semantic_intent_signature",
    "ledger_from_extraction",
]

#: llm_policy task name for the extraction budget.
SEMANTIC_INTENT_EXTRACTION = "semantic_intent_extraction"

#: Markdown guide name bundled with the extraction signature.
SEMANTIC_INTENT_GUIDE = "semantic-intent-extraction"

#: Visible, bounded status values for a structured-output attempt.
EXTRACTION_OK = "ok"
EXTRACTION_EMPTY = "empty"
EXTRACTION_FAILED = "failed"

_ENTITY_KIND_ALIASES = {
    "human": "person",
    "animal": "creature",
    "item": "object",
    "collective": "group",
    "place": "location",
    "concept": "abstract",
}


class IntentExtractionResult(BaseModel):
    """Typed extraction output from the DSPy model (language-neutral)."""

    language: str = Field(default="", max_length=16)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)


class SemanticIntentExtraction(BaseModel):
    """Bounded extraction result: a constructible ledger plus a visible status."""

    ledger: IntentLedger
    status: str
    warnings: list[str] = Field(default_factory=list)


def build_semantic_intent_signature(dspy_module: Any | None = None) -> Any:
    """Return the DSPy signature for multilingual intent extraction.

    The output is the language-neutral ``IntentExtractionResult``. The model is
    instructed (in the signature docstring and the bundled guide) to copy only
    what the source states and to leave fields empty rather than invent them.
    """
    if dspy_module is None:
        import dspy as dspy_module

    class ExtractSemanticIntent(dspy_module.Signature):
        """Extract explicit intent constraints from a free-form story idea.

        Translate the story idea into the typed semantic intent result in the
        idea's own language. Extract only entities, attributes, relations,
        cardinality, recurrence, and scene obligations that the source states.
        Preserve the source-language wording for every explicit constraint.
        Do not invent gender, performers, or entity properties the source does
        not specify: a story without a singer must not gain one. An animated
        stone statue may be extracted as the lead. Keep role-bound attributes
        attached to the correct entity. Use stable, readable IDs. Return only
        the extraction result; do not echo the guide or DSPy markers.
        """

        story_idea: str = dspy_module.InputField()
        notes: str = dspy_module.InputField(default="")
        guide: str = dspy_module.InputField(default="")
        extraction: IntentExtractionResult = dspy_module.OutputField()

    return ExtractSemanticIntent


def ledger_from_extraction(
    extraction: IntentExtractionResult,
    *,
    language: str = "",
) -> tuple[IntentLedger, list[str]]:
    """Build a valid ``IntentLedger`` from a typed extraction result.

    This is the no-invention + ID-repair boundary: it repairs missing or
    duplicate IDs, drops dangling relation/constraint references, and keeps
    only the explicit provenance the model supplied. It never adds entities
    the source did not state, so an empty extraction yields an empty ledger.

    Returns:
        A tuple of the constructible ledger and a list of repair warnings.
    """
    warnings: list[str] = []
    language = (language or extraction.language or "").strip()

    entities: list[IntentEntity] = []
    used_ids: set[str] = set()
    for raw in extraction.entities:
        if not isinstance(raw, dict):
            continue
        entity = _build_entity(raw, used_ids, warnings)
        if entity is not None:
            entities.append(entity)

    entity_ids = {entity.id for entity in entities}
    relations: list[IntentRelation] = []
    for index, raw in enumerate(extraction.relations):
        if not isinstance(raw, dict):
            continue
        relation = _build_relation(raw, index, entity_ids, used_ids, warnings)
        if relation is not None:
            relations.append(relation)

    constraints: list[IntentConstraint] = []
    for index, raw in enumerate(extraction.constraints):
        if not isinstance(raw, dict):
            continue
        constraint = _build_constraint(raw, index, entity_ids, used_ids, warnings)
        if constraint is not None:
            constraints.append(constraint)

    ledger = IntentLedger(entities=entities, relations=relations, constraints=constraints)
    return ledger, warnings


def _next_id(prefix: str, used: set[str]) -> str:
    candidate = prefix
    counter = 1
    while candidate in used:
        counter += 1
        candidate = f"{prefix}-{counter}"
    used.add(candidate)
    return candidate


def _build_entity(
    raw: dict[str, Any], used_ids: set[str], warnings: list[str]
) -> IntentEntity | None:
    raw_id = str(raw.get("id") or "").strip()
    kind = str(raw.get("kind") or "").strip().casefold()
    kind = _ENTITY_KIND_ALIASES.get(kind, kind)
    if not kind:
        warnings.append(f"entity without kind dropped: {raw_id or '(no id)'}")
        return None
    entity_id = raw_id or _next_id("entity", used_ids)
    if not raw_id:
        warnings.append(f"entity without id assigned: {entity_id}")
    elif entity_id in used_ids:
        entity_id = _next_id(raw_id, used_ids)
        warnings.append(f"duplicate entity id '{raw_id}' -> {entity_id}")
    used_ids.add(entity_id)
    identity = {
        str(key): str(value)
        for key, value in (raw.get("identity") or {}).items()
        if value not in (None, "")
    } if isinstance(raw.get("identity"), dict) else {}
    return IntentEntity(
        id=entity_id,
        kind=kind,
        role=str(raw.get("role") or "").strip(),
        identity=identity,
        recurrence=str(raw.get("recurrence") or "once").strip() or "once",
        status=str(raw.get("status") or "required").strip() or "required",
        description=str(raw.get("description") or "").strip(),
    )


def _build_relation(
    raw: dict[str, Any],
    index: int,
    entity_ids: set[str],
    used_ids: set[str],
    warnings: list[str],
) -> IntentRelation | None:
    subject = str(raw.get("subject_id") or "").strip()
    target = str(raw.get("target_id") or "").strip()
    relation = str(raw.get("relation") or "").strip()
    if not relation:
        warnings.append(f"relation {index} without label dropped")
        return None
    if subject not in entity_ids or target not in entity_ids:
        warnings.append(
            f"relation {index} dropped: dangling reference "
            f"({subject!r} -> {target!r})"
        )
        return None
    relation_id = str(raw.get("id") or "").strip() or _next_id("relation", used_ids)
    if relation_id in used_ids:
        relation_id = _next_id(relation_id, used_ids)
    used_ids.add(relation_id)
    try:
        cardinality = int(raw.get("cardinality", 1))
    except (TypeError, ValueError):
        cardinality = 1
    return IntentRelation(
        id=relation_id,
        subject_id=subject,
        relation=relation,
        target_id=target,
        cardinality=max(1, cardinality),
        status=str(raw.get("status") or "required").strip() or "required",
        provenance=_build_provenance(raw.get("provenance")),
    )


def _build_constraint(
    raw: dict[str, Any],
    index: int,
    entity_ids: set[str],
    used_ids: set[str],
    warnings: list[str],
) -> IntentConstraint | None:
    entity_id = str(raw.get("entity_id") or "").strip()
    statement = str(raw.get("statement") or "").strip()
    if not statement:
        warnings.append(f"constraint {index} without statement dropped")
        return None
    if entity_id not in entity_ids:
        warnings.append(
            f"constraint {index} dropped: dangling entity {entity_id!r}"
        )
        return None
    constraint_id = str(raw.get("id") or "").strip() or _next_id("constraint", used_ids)
    if constraint_id in used_ids:
        constraint_id = _next_id(constraint_id, used_ids)
    used_ids.add(constraint_id)
    return IntentConstraint(
        id=constraint_id,
        entity_id=entity_id,
        kind=str(raw.get("kind") or "identity").strip() or "identity",
        statement=statement,
        status=str(raw.get("status") or "required").strip() or "required",
        provenance=_build_provenance(raw.get("provenance")),
    )


def _build_provenance(raw: Any) -> ConstraintProvenance | None:
    if not isinstance(raw, dict):
        return None
    origin = str(raw.get("origin") or "").strip()
    if origin not in ("explicit", "inferred"):
        return None
    source_text = str(raw.get("source_text") or "").strip()
    if origin == "explicit" and not source_text:
        return None
    return ConstraintProvenance(
        origin=origin,
        source_text=source_text,
        language=str(raw.get("language") or "").strip(),
        source_ref=str(raw.get("source_ref") or "").strip(),
    )


class SemanticIntentExtractor:
    """DSPy extraction adapter with an optional predictor seam.

    With no predictor configured (the default), the extractor is a no-op that
    returns an empty ledger and ``EXTRACTION_EMPTY`` status -- it never invents
    entities. With a predictor (production DSPy or a test double), it calls the
    model, decodes the structured output, and builds a valid ledger. A failed or
    malformed structured output is bounded: it returns an empty ledger with
    ``EXTRACTION_FAILED`` and a logged warning rather than raising, so the
    downstream subject/location stage always has a constructible artifact.
    """

    def __init__(self, predictor: Any | None = None):
        self.predictor = predictor

    def _invoke(self, story_idea: str, notes: str) -> Any:
        """Call the predictor. Subclasses override to add a DSPy LM context."""
        assert self.predictor is not None
        return self.predictor(story_idea=story_idea, notes=notes)

    def extract(
        self,
        story_idea: str,
        notes: str = "",
        *,
        language: str = "",
    ) -> SemanticIntentExtraction:
        if self.predictor is None:
            return SemanticIntentExtraction(
                ledger=IntentLedger(),
                status=EXTRACTION_EMPTY,
                warnings=["no extractor configured; empty ledger (no invention)"],
            )
        try:
            raw = self._invoke(story_idea, notes)
            extraction = self._decode(raw)
        except Exception as exc:  # noqa: BLE001 - bounded structured-output failure
            logger.warning("semantic intent extraction failed: %s", exc)
            return SemanticIntentExtraction(
                ledger=IntentLedger(),
                status=EXTRACTION_FAILED,
                warnings=[f"extraction failed: {type(exc).__name__}"],
            )
        ledger, warnings = ledger_from_extraction(extraction, language=language)
        status = EXTRACTION_EMPTY if not ledger.entities else EXTRACTION_OK
        return SemanticIntentExtraction(ledger=ledger, status=status, warnings=warnings)

    def _decode(self, raw: Any) -> IntentExtractionResult:
        payload = raw
        if isinstance(raw, dict):
            payload = raw.get("extraction", raw)
        elif hasattr(raw, "extraction"):
            payload = getattr(raw, "extraction")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        if isinstance(payload, str):
            from feverslop.domain.llm_parsing import extract_json_object

            payload = extract_json_object(payload)
        if not isinstance(payload, dict):
            raise ValueError("semantic intent extraction returned no structured result")
        return IntentExtractionResult.model_validate(payload)


class DspySemanticIntentExtractor(SemanticIntentExtractor):
    """Production DSPy adapter for multilingual intent extraction.

    Mirrors ``DspySubjectDirectivePlanner``: it builds the signature, creates a
    bounded LM (a compact structured task -- never the project-wide generation
    budget), and delegates to the tolerant base ``extract`` so a failed or
    malformed model response is bounded, not fatal.
    """

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        from feverslop.prompting.dspy_runtime import DspyRuntime

        self.runtime = dspy_runtime or DspyRuntime.create()
        import dspy

        self.predictor = self.runtime.predict(build_semantic_intent_signature(dspy))
        # A compact structured extraction. Do not inherit a project-wide
        # generation budget (which may be 65k+) and let one malformed story
        # idea consume the whole response or truncate before the JSON completes.
        self.lm = self.runtime.make_lm(llm, max_tokens=4096, task="analyzer")

    def _invoke(self, story_idea: str, notes: str) -> Any:
        """Run the predictor inside the DSPy LM context, with the guide."""
        from feverslop.prompting.guide_loader import load_markdown_guide

        guide = load_markdown_guide(SEMANTIC_INTENT_GUIDE)
        with self.runtime.context(lm=self.lm):
            return self.predictor(
                story_idea=story_idea,
                notes=notes,
                guide=guide,
            )
