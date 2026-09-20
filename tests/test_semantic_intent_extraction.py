"""Focused tests for the DSPy multilingual intent extraction stage (#544).

These use deterministic fake model responses (no live LLM) to cover the
acceptance criteria: German/English equivalence, no-invention for
no-performer stories, non-human leads, role-bound attributes, and bounded
structured-output failures.
"""

import unittest

from feverslop.prompting.semantic_intent_extraction import (
    EXTRACTION_EMPTY,
    EXTRACTION_FAILED,
    EXTRACTION_OK,
    SemanticIntentExtractor,
    build_semantic_intent_signature,
    ledger_from_extraction,
    IntentExtractionResult,
)


def _extractor_returning(payload) -> SemanticIntentExtractor:
    def predictor(**kwargs):
        return payload

    return SemanticIntentExtractor(predictor=predictor)


class SemanticIntentExtractionContractTests(unittest.TestCase):
    def test_signature_has_typed_input_and_output(self):
        bundle = build_semantic_intent_signature()
        fields = set(bundle.model_fields.keys())
        self.assertIn("story_idea", fields)
        self.assertIn("extraction", fields)

    def test_no_predictor_yields_empty_ledger_without_invention(self):
        # Acceptance: a story without a singer does not invent one.
        result = SemanticIntentExtractor().extract("A stone statue dances alone.")
        self.assertEqual(result.status, EXTRACTION_EMPTY)
        self.assertEqual(len(result.ledger.entities), 0)
        self.assertTrue(result.warnings)

    def test_empty_extraction_yields_empty_ledger(self):
        ledger, warnings = ledger_from_extraction(IntentExtractionResult(language="de"))
        self.assertEqual(len(ledger.entities), 0)
        self.assertEqual(warnings, [])


class SemanticIntentExtractionMultilingualTests(unittest.TestCase):
    def test_german_and_english_ideas_yield_equivalent_ledgers(self):
        # Acceptance: equivalent German and English ideas yield semantically
        # equivalent ledgers (same entity count, kinds, and role-bound attribute).
        def de_payload(**kwargs):
            return {"extraction": {
                "language": "de",
                "entities": [{"id": "stein", "kind": "object", "role": "lead"}],
                "constraints": [{
                    "entity_id": "stein", "kind": "identity", "statement": "immer aus Stein",
                    "provenance": {"origin": "explicit", "source_text": "immer aus Stein", "language": "de"},
                }],
            }}

        def en_payload(**kwargs):
            return {"extraction": {
                "language": "en",
                "entities": [{"id": "statue", "kind": "object", "role": "lead"}],
                "constraints": [{
                    "entity_id": "statue", "kind": "identity", "statement": "always made of stone",
                    "provenance": {"origin": "explicit", "source_text": "always made of stone", "language": "en"},
                }],
            }}

        de = _extractor_returning(de_payload()).extract("Ein animierter Stein.")
        en = _extractor_returning(en_payload()).extract("An animated stone statue.")

        self.assertEqual(len(de.ledger.entities), len(en.ledger.entities))
        self.assertEqual(
            {e.kind for e in de.ledger.entities}, {e.kind for e in en.ledger.entities}
        )
        self.assertEqual(len(de.ledger.constraints), len(en.ledger.constraints))
        # Provenance preserves the source-language wording.
        self.assertEqual(de.ledger.constraints[0].provenance.language, "de")
        self.assertEqual(en.ledger.constraints[0].provenance.language, "en")

    def test_animated_stone_statue_can_be_extracted_as_lead(self):
        # Acceptance: an animated stone statue can be extracted as the lead.
        def payload(**kwargs):
            return {"extraction": {
                "language": "en",
                "entities": [{"id": "statue", "kind": "object", "role": "lead"}],
            }}

        result = _extractor_returning(payload()).extract("An animated stone statue leads.")
        self.assertEqual(result.status, EXTRACTION_OK)
        self.assertEqual(result.ledger.entities[0].kind, "object")
        self.assertEqual(result.ledger.entities[0].role, "lead")


class SemanticIntentExtractionNoInventionTests(unittest.TestCase):
    def test_no_singer_story_does_not_gain_a_performer(self):
        # Acceptance: a story without a singer does not invent one.
        def payload(**kwargs):
            return {"extraction": {
                "language": "en",
                "entities": [{"id": "dream", "kind": "abstract"}],
            }}

        result = _extractor_returning(payload()).extract("A dream drifts over a city.")
        kinds = {e.kind for e in result.ledger.entities}
        self.assertNotIn("person", kinds)

    def test_role_bound_attribute_stays_attached_to_correct_entity(self):
        # Acceptance: explicit role-bound attributes remain attached to the
        # correct entity.
        def payload(**kwargs):
            return {"extraction": {
                "language": "en",
                "entities": [
                    {"id": "a", "kind": "person", "role": "narrator"},
                    {"id": "b", "kind": "person", "role": "listener"},
                ],
                "constraints": [{
                    "entity_id": "a", "kind": "identity", "statement": "wears a red coat",
                }],
            }}

        result = _extractor_returning(payload()).extract("A narrator and a listener.")
        constraint = result.ledger.constraints[0]
        self.assertEqual(constraint.entity_id, "a")


class SemanticIntentExtractionFailureTests(unittest.TestCase):
    def test_structured_output_failure_is_visible_and_bounded(self):
        # Acceptance: structured output failures are visible and bounded.
        def boom(**kwargs):
            raise RuntimeError("model exploded")

        result = SemanticIntentExtractor(predictor=boom).extract("any story")
        self.assertEqual(result.status, EXTRACTION_FAILED)
        self.assertEqual(len(result.ledger.entities), 0)
        self.assertTrue(any("extraction failed" in warning for warning in result.warnings))

    def test_malformed_payload_is_bounded(self):
        def payload(**kwargs):
            return "not a dict at all"

        result = _extractor_returning(payload()).extract("any story")
        self.assertEqual(result.status, EXTRACTION_FAILED)
        self.assertEqual(len(result.ledger.entities), 0)


class SemanticIntentExtractionRepairTests(unittest.TestCase):
    def test_place_and_location_are_accepted_as_location_entities(self):
        for input_kind in ("place", "location"):
            with self.subTest(input_kind=input_kind):
                extraction = IntentExtractionResult(
                    entities=[{"id": "forest", "kind": input_kind}]
                )

                ledger, _warnings = ledger_from_extraction(extraction)

                self.assertEqual(ledger.entities[0].kind, "location")

    def test_common_entity_kind_aliases_are_normalized(self):
        aliases = {
            "human": "person",
            "animal": "creature",
            "item": "object",
            "collective": "group",
            "concept": "abstract",
        }
        extraction = IntentExtractionResult(
            entities=[
                {"id": f"entity-{index}", "kind": input_kind}
                for index, input_kind in enumerate(aliases)
            ]
        )

        ledger, _warnings = ledger_from_extraction(extraction)

        self.assertEqual(
            [entity.kind for entity in ledger.entities],
            list(aliases.values()),
        )

    def test_duplicate_and_missing_ids_are_repaired(self):
        def payload(**kwargs):
            return {"extraction": {
                "language": "en",
                "entities": [
                    {"id": "a", "kind": "person"},
                    {"id": "a", "kind": "creature"},
                    {"kind": "object"},
                ],
            }}

        result = _extractor_returning(payload()).extract("story")
        ids = [e.id for e in result.ledger.entities]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)  # all unique after repair
        self.assertTrue(result.warnings)

    def test_dangling_relation_and_constraint_are_dropped(self):
        def payload(**kwargs):
            return {"extraction": {
                "language": "en",
                "entities": [{"id": "a", "kind": "person"}],
                "relations": [{"subject_id": "a", "relation": "sees", "target_id": "missing"}],
                "constraints": [{"entity_id": "missing", "kind": "identity", "statement": "x"}],
            }}

        result = _extractor_returning(payload()).extract("story")
        self.assertEqual(len(result.ledger.relations), 0)
        self.assertEqual(len(result.ledger.constraints), 0)
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()
