import unittest

from pydantic import ValidationError

from feverslop.domain.semantic_intent import (
    SCHEMA_VERSION,
    IntentConstraint,
    IntentEntity,
    IntentLedger,
    IntentRelation,
    ConstraintProvenance,
    ledger_from_legacy_cast,
)


def _entity(eid: str, kind: str = "person", **kwargs) -> dict:
    return {"id": eid, "kind": kind, **kwargs}


class SemanticIntentEntityTests(unittest.TestCase):
    def test_accepts_non_human_entity_kinds(self):
        # Acceptance: arbitrary role labels and non-human leads are representable.
        for kind in ("creature", "object", "group", "location", "abstract"):
            entity = IntentEntity(id=f"e-{kind}", kind=kind, role="free label")
            self.assertEqual(entity.kind, kind)

    def test_rejects_unknown_entity_kind(self):
        with self.assertRaises(ValidationError):
            IntentEntity(id="e1", kind="singer")

    def test_rejects_malformed_entity_id(self):
        for bad in ("", "   "):
            with self.assertRaises(ValidationError):
                IntentEntity(id=bad, kind="person")

    def test_rejects_long_entity_id(self):
        with self.assertRaises(ValidationError):
            IntentEntity(id="x" * 129, kind="person")

    def test_rejects_unknown_recurrence_and_status(self):
        with self.assertRaises(ValidationError):
            IntentEntity(id="e1", kind="person", recurrence="sometimes")
        with self.assertRaises(ValidationError):
            IntentEntity(id="e1", kind="person", status="maybe")


class SemanticIntentConstraintTests(unittest.TestCase):
    def test_explicit_and_inferred_are_distinguishable(self):
        # Acceptance: explicit and inferred constraints are distinguishable.
        explicit = ConstraintProvenance(origin="explicit", source_text="der Stein singt", language="de")
        inferred = ConstraintProvenance(origin="inferred")
        self.assertEqual(explicit.origin, "explicit")
        self.assertEqual(inferred.origin, "inferred")
        self.assertNotEqual(explicit.origin, inferred.origin)

    def test_explicit_provenance_requires_source_text(self):
        with self.assertRaises(ValidationError):
            ConstraintProvenance(origin="explicit")

    def test_rejects_unknown_provenance_origin(self):
        with self.assertRaises(ValidationError):
            ConstraintProvenance(origin="guessed")

    def test_constraint_rejects_unknown_kind(self):
        with self.assertRaises(ValidationError):
            IntentConstraint(id="c1", entity_id="e1", kind="vibe", statement="x")


class SemanticIntentLedgerTests(unittest.TestCase):
    def test_rejects_dangling_relation_target(self):
        # Acceptance: typed models reject dangling relations.
        with self.assertRaises(ValidationError):
            IntentLedger(
                entities=[IntentEntity(id="e1", kind="person")],
                relations=[
                    IntentRelation(id="r1", subject_id="e1", relation="next to", target_id="missing")
                ],
            )

    def test_rejects_dangling_constraint_entity(self):
        with self.assertRaises(ValidationError):
            IntentLedger(
                entities=[IntentEntity(id="e1", kind="person")],
                constraints=[IntentConstraint(id="c1", entity_id="missing", kind="identity", statement="x")],
            )

    def test_rejects_duplicate_entity_ids(self):
        with self.assertRaises(ValidationError):
            IntentLedger(
                entities=[
                    IntentEntity(id="e1", kind="person"),
                    IntentEntity(id="e1", kind="creature"),
                ]
            )

    def test_rejects_duplicate_relation_ids(self):
        with self.assertRaises(ValidationError):
            IntentLedger(
                entities=[IntentEntity(id="e1", kind="person")],
                relations=[
                    IntentRelation(id="r1", subject_id="e1", relation="a", target_id="e1"),
                    IntentRelation(id="r1", subject_id="e1", relation="b", target_id="e1"),
                ],
            )

    def test_rejects_duplicate_constraint_ids(self):
        with self.assertRaises(ValidationError):
            IntentLedger(
                entities=[IntentEntity(id="e1", kind="person")],
                constraints=[
                    IntentConstraint(id="c1", entity_id="e1", kind="identity", statement="x"),
                    IntentConstraint(id="c1", entity_id="e1", kind="identity", statement="y"),
                ],
            )

    def test_rejects_dangling_relation_subject(self):
        with self.assertRaises(ValidationError):
            IntentLedger(
                entities=[IntentEntity(id="e1", kind="person")],
                relations=[IntentRelation(id="r1", subject_id="missing", relation="x", target_id="e1")],
            )

    def test_accepts_valid_ledger_with_all_entity_kinds(self):
        ledger = IntentLedger(
            entities=[
                IntentEntity(id="band", kind="group", role="die Band"),
                IntentEntity(id="statue", kind="object", description="animated stone statue"),
                IntentEntity(id="city", kind="location"),
                IntentEntity(id="fate", kind="abstract"),
            ],
            relations=[
                IntentRelation(id="r1", subject_id="statue", relation="guards", target_id="city", cardinality=2)
            ],
            constraints=[
                IntentConstraint(id="c1", entity_id="statue", kind="scene_obligation", statement="must appear in scene 3")
            ],
        )
        self.assertEqual(len(ledger.entity_ids()), 4)


class SemanticIntentSerializationTests(unittest.TestCase):
    def test_serialization_is_stable_and_versioned(self):
        # Acceptance: schema serialization is stable and versioned.
        ledger = IntentLedger(
            entities=[IntentEntity(id="e1", kind="person", role="narrator")],
            relations=[IntentRelation(id="r1", subject_id="e1", relation="speaks", target_id="e1")],
        )
        first = ledger.to_dict()
        second = IntentLedger.from_dict(first).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], SCHEMA_VERSION)

    def test_rejects_unsupported_schema_version(self):
        with self.assertRaises(ValidationError):
            IntentLedger(schema_version="semantic-intent/v999")

    def test_round_trip_preserves_provenance(self):
        ledger = IntentLedger(
            entities=[IntentEntity(id="e1", kind="person")],
            constraints=[
                IntentConstraint(
                    id="c1",
                    entity_id="e1",
                    kind="identity",
                    statement="always wears red",
                    provenance=ConstraintProvenance(origin="explicit", source_text="immer rot", language="de"),
                )
            ],
        )
        restored = IntentLedger.from_dict(ledger.to_dict())
        self.assertIsNotNone(restored.constraints[0].provenance)
        self.assertEqual(restored.constraints[0].provenance.origin, "explicit")
        self.assertEqual(restored.constraints[0].provenance.source_text, "immer rot")


class SemanticIntentCompatibilityTests(unittest.TestCase):
    def test_legacy_actors_and_locations_map_to_ledger(self):
        # Acceptance: legacy projects have a documented compatibility path.
        actors = [
            {"id": "a1", "name": "Anna", "role": "singer", "visual_description": "red hair"},
            {"id": "a2", "name": "Ben", "role": "drummer"},
        ]
        locations = [{"id": "l1", "name": "Club", "visual_description": "dim lights"}]
        ledger = ledger_from_legacy_cast(actors, locations)
        kinds = {entity.id: entity.kind for entity in ledger.entities}
        self.assertEqual(kinds, {"a1": "person", "a2": "person", "l1": "location"})
        # No fixed singer/band ontology: role stays a free label.
        self.assertEqual(ledger.entities[0].role, "singer")

    def test_legacy_cast_skips_entries_without_id(self):
        ledger = ledger_from_legacy_cast([{"name": "no id"}, {"id": "a1", "name": "A"}])
        self.assertEqual(ledger.entity_ids(), frozenset({"a1"}))

    def test_legacy_cast_accepts_none_inputs(self):
        ledger = ledger_from_legacy_cast(None, None)
        self.assertEqual(len(ledger.entities), 0)


class SemanticIntentFixtureTests(unittest.TestCase):
    def test_four_member_band_use_case(self):
        # Acceptance: the four-member band use case.
        members = [IntentEntity(id=f"member-{i}", kind="person", role="band member") for i in range(1, 5)]
        ledger = IntentLedger(
            entities=[IntentEntity(id="band", kind="group", role="vierkoepfige Band"), *members],
            relations=[
                IntentRelation(id=f"m{i}", subject_id=f"member-{i}", relation="member of", target_id="band")
                for i in range(1, 5)
            ],
        )
        self.assertEqual(len(ledger.entities), 5)
        self.assertEqual(len(ledger.relations), 4)

    def test_no_performer_story_is_representable(self):
        # Acceptance: no-performer stories (abstract/object leads).
        ledger = IntentLedger(
            entities=[
                IntentEntity(id="stone", kind="object", description="animated stone statue"),
                IntentEntity(id="dream", kind="abstract"),
            ]
        )
        self.assertEqual(ledger.entity_ids(), frozenset({"stone", "dream"}))

    def test_german_and_english_input_concepts(self):
        # Acceptance: unit fixtures cover German/English input concepts.
        ledger = IntentLedger(
            entities=[IntentEntity(id="e1", kind="person")],
            constraints=[
                IntentConstraint(
                    id="de",
                    entity_id="e1",
                    kind="identity",
                    statement="trägt immer einen Hut",
                    provenance=ConstraintProvenance(origin="explicit", source_text="trägt immer einen Hut", language="de"),
                ),
                IntentConstraint(
                    id="en",
                    entity_id="e1",
                    kind="identity",
                    statement="always wears a hat",
                    provenance=ConstraintProvenance(origin="explicit", source_text="always wears a hat", language="en"),
                ),
            ],
        )
        self.assertEqual(len(ledger.constraints), 2)


if __name__ == "__main__":
    unittest.main()
