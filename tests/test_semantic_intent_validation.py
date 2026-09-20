import unittest

from feverslop.domain.semantic_intent_validation import (
    MIGRATED_FROM_KEY,
    IntentFinding,
    blocking_findings,
    legacy_cast_to_ledger,
    migrate_ledger_payload,
    restore_ledger_original,
    validate_intent_ledger,
)


def _entity(eid: str, kind: str = "person") -> dict:
    return {"id": eid, "kind": kind}


class IntentFindingTests(unittest.TestCase):
    def test_finding_is_stable_and_serializable(self):
        finding = IntentFinding("dangling_relation", "error", "r1", "msg")
        self.assertEqual(
            finding.as_dict(),
            {"code": "dangling_relation", "severity": "error", "subject_id": "r1", "message": "msg"},
        )


class ValidateLedgerTests(unittest.TestCase):
    def test_valid_ledger_has_no_findings(self):
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [_entity("e1"), _entity("e2")],
            "relations": [
                {"id": "r1", "subject_id": "e1", "relation": "next to", "target_id": "e2"}
            ],
            "constraints": [
                {"id": "c1", "entity_id": "e1", "kind": "identity", "statement": "wears red"}
            ],
        }
        self.assertEqual(validate_intent_ledger(payload), [])

    def test_validation_is_repeatable(self):
        # Acceptance: deterministic validation produces repeatable results.
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [_entity("e1"), _entity("e1")],
        }
        first = validate_intent_ledger(payload)
        second = validate_intent_ledger(payload)
        self.assertEqual([f.as_dict() for f in first], [f.as_dict() for f in second])

    def test_unsupported_schema_version_is_reported_by_id(self):
        payload = {"schema_version": "semantic-intent/v999", "entities": [_entity("e1")]}
        findings = validate_intent_ledger(payload)
        self.assertIn("unsupported_schema", [f.code for f in findings])

    def test_dangling_relation_target_is_reported_by_id(self):
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [_entity("e1")],
            "relations": [
                {"id": "r1", "subject_id": "e1", "relation": "x", "target_id": "missing"}
            ],
        }
        findings = validate_intent_ledger(payload)
        dangling = [f for f in findings if f.code == "dangling_relation_target"]
        self.assertEqual(len(dangling), 1)
        self.assertEqual(dangling[0].subject_id, "r1")

    def test_dangling_constraint_entity_is_reported_by_id(self):
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [_entity("e1")],
            "constraints": [
                {"id": "c1", "entity_id": "missing", "kind": "identity", "statement": "x"}
            ],
        }
        findings = validate_intent_ledger(payload)
        self.assertIn("dangling_constraint", [f.code for f in findings])

    def test_duplicate_entity_id_is_reported_by_id(self):
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [_entity("e1"), _entity("e1")],
        }
        findings = validate_intent_ledger(payload)
        self.assertIn("duplicate_entity_id", [f.code for f in findings])

    def test_malformed_entity_id_is_reported(self):
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [{"id": "   ", "kind": "person"}],
        }
        findings = validate_intent_ledger(payload)
        self.assertIn("malformed_entity_id", [f.code for f in findings])

    def test_unknown_entity_kind_is_a_warning(self):
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [{"id": "e1", "kind": "singer"}],
        }
        findings = validate_intent_ledger(payload)
        kind = [f for f in findings if f.code == "unknown_entity_kind"]
        self.assertEqual(len(kind), 1)
        self.assertEqual(kind[0].severity, "warning")

    def test_contradictory_required_identity_is_reported_by_id(self):
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [_entity("e1")],
            "constraints": [
                {"id": "c1", "entity_id": "e1", "kind": "identity", "statement": "wears red"},
                {"id": "c2", "entity_id": "e1", "kind": "identity", "statement": "wears blue"},
            ],
        }
        findings = validate_intent_ledger(payload)
        contradiction = [f for f in findings if f.code == "contradictory_constraint"]
        self.assertEqual(len(contradiction), 1)
        self.assertEqual(contradiction[0].subject_id, "c2")

    def test_rejects_invalid_mode(self):
        with self.assertRaises(ValueError):
            validate_intent_ledger({"entities": []}, mode="bogus")


class BlockingFindingsTests(unittest.TestCase):
    def _findings(self) -> list[IntentFinding]:
        return [
            IntentFinding("dangling_relation", "error", "r1", "e"),
            IntentFinding("unknown_entity_kind", "warning", "e1", "w"),
        ]

    def test_strict_blocks_errors_and_warnings(self):
        self.assertEqual(len(blocking_findings(self._findings(), "strict")), 2)

    def test_warn_blocks_errors_only(self):
        blocked = blocking_findings(self._findings(), "warn")
        self.assertEqual([f.severity for f in blocked], ["error"])

    def test_compat_blocks_nothing(self):
        self.assertEqual(blocking_findings(self._findings(), "compat"), [])

    def test_rejects_invalid_mode(self):
        with self.assertRaises(ValueError):
            blocking_findings(self._findings(), "bogus")


class LegacyCastTests(unittest.TestCase):
    def test_maps_actors_and_locations_without_fabricated_constraints(self):
        payload = legacy_cast_to_ledger(
            [{"id": "a1", "name": "Anna", "role": "singer", "visual_description": "red hair"}],
            [{"id": "l1", "name": "Club", "visual_description": "dim lights"}],
        )
        self.assertEqual(len(payload["entities"]), 2)
        # No unspecified demographic/identity attributes are introduced.
        self.assertEqual(payload["relations"], [])
        self.assertEqual(payload["constraints"], [])

    def test_accepts_none_inputs(self):
        payload = legacy_cast_to_ledger(None, None)
        self.assertEqual(payload["entities"], [])


class MigrationTests(unittest.TestCase):
    def test_v1_payload_passes_through_untouched(self):
        # Acceptance: migration is idempotent for supported versions.
        payload = {
            "schema_version": "semantic-intent/v1",
            "entities": [_entity("e1")],
        }
        migrated = migrate_ledger_payload(payload)
        self.assertEqual(migrated, payload)
        self.assertNotIn(MIGRATED_FROM_KEY, migrated)

    def test_legacy_payload_is_converted_and_original_preserved(self):
        legacy = {
            "schema_version": "legacy/v0",
            "actors": [{"id": "a1", "name": "Anna", "role": "singer"}],
            "locations": [{"id": "l1", "name": "Club"}],
        }
        migrated = migrate_ledger_payload(legacy)
        self.assertEqual(migrated["schema_version"], "semantic-intent/v1")
        self.assertEqual(len(migrated["entities"]), 2)
        # The original is preserved so the migration is reversible.
        self.assertEqual(migrated[MIGRATED_FROM_KEY], legacy)

    def test_migration_is_reversible(self):
        legacy = {
            "schema_version": "legacy/v0",
            "actors": [{"id": "a1", "name": "Anna"}],
        }
        migrated = migrate_ledger_payload(legacy)
        self.assertEqual(restore_ledger_original(migrated), legacy)

    def test_restore_is_identity_for_unmigrated_payload(self):
        payload = {"schema_version": "semantic-intent/v1", "entities": [_entity("e1")]}
        self.assertEqual(restore_ledger_original(payload), payload)


if __name__ == "__main__":
    unittest.main()
