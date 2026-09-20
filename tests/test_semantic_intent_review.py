"""Focused tests for the independent semantic intent review/repair stage (#545).

These use deterministic fake model responses (no live LLM) to cover the
acceptance criteria: the judge detects omitted/invented/contradicted/misbound
constraints; repair retains unaffected IDs and records; retry exhaustion stops
before expensive reference rendering or follows an explicit warning policy;
logs identify the constraint, source evidence, attempt, and outcome without
dumping full prompts; and correlated bad extraction/repair, ambiguous prose,
and false-positive controls are covered.
"""

import unittest

from pathlib import Path
from typing import Any

from feverslop.domain.semantic_intent import (
    IntentConstraint,
    IntentEntity,
    IntentLedger,
    IntentRelation,
)
from feverslop.prompting.semantic_intent_review import (
    AMBIGUOUS,
    DEFAULT_REVIEW_MAX_ATTEMPTS,
    FAILED,
    NEEDS_REPAIR,
    OK,
    REPAIRED,
    DspySemanticIntentReviewer,
    IntentFinding,
    IntentReviewResult,
    SemanticIntentReviewer,
    build_semantic_intent_review_signature,
    repair_ledger,
    review_and_repair,
)


def _reviewer_returning(payload, *, attempts=1) -> SemanticIntentReviewer:
    calls = {"n": 0}

    def predictor(**kwargs):
        calls["n"] += 1
        return payload

    return SemanticIntentReviewer(predictor=predictor)


def _reviewer_raising(exc, *, times) -> SemanticIntentReviewer:
    calls = {"n": 0}

    def predictor(**kwargs):
        calls["n"] += 1
        if calls["n"] <= times:
            raise exc
        return {"review": {"findings": [], "status": "ok"}}

    return SemanticIntentReviewer(predictor=predictor)


def _sample_ledger() -> IntentLedger:
    return IntentLedger(
        entities=[
            IntentEntity(id="statue", kind="object", role="lead"),
            IntentEntity(id="band", kind="group"),
        ],
        relations=[
            IntentRelation(id="r1", subject_id="statue", relation="part_of", target_id="band"),
        ],
        constraints=[
            IntentConstraint(id="c1", entity_id="band", kind="identity", statement="four members"),
        ],
    )


class SemanticIntentReviewContractTests(unittest.TestCase):
    def test_signature_has_typed_input_and_output(self):
        bundle = build_semantic_intent_review_signature()
        fields = set(bundle.model_fields.keys())
        self.assertIn("story_idea", fields)
        self.assertIn("ledger", fields)
        self.assertIn("review", fields)
        self.assertEqual(bundle.input_fields["ledger"].annotation, dict[str, Any])

    def test_no_predictor_yields_ok_without_findings(self):
        # Acceptance: with no reviewer configured, no findings are invented.
        result = SemanticIntentReviewer().review("A stone statue leads a band.", _sample_ledger())
        self.assertEqual(result.status, OK)
        self.assertEqual(result.findings, [])
        self.assertTrue(result.warnings)


class SemanticIntentReviewDetectionTests(unittest.TestCase):
    def test_successfully_applied_findings_report_repaired(self):
        payload = {
            "review": {
                "findings": [
                    {
                        "id": "drop-r1",
                        "kind": "invented",
                        "record_id": "r1",
                        "description": "not stated",
                    }
                ],
                "status": "needs_repair",
            }
        }

        result = review_and_repair("idea", _sample_ledger(), _reviewer_returning(payload))

        self.assertEqual(REPAIRED, result.status)

    def test_detects_all_four_finding_classes(self):
        # Acceptance: judge detects omitted, invented, contradicted, misbound.
        payload = {
            "review": {
                "findings": [
                    {"id": "f-omitted", "kind": "omitted", "description": "missing 'always stone'"},
                    {"id": "f-invented", "kind": "invented", "record_id": "r1", "description": "r1 not stated"},
                    {"id": "f-contradicted", "kind": "contradicted", "record_id": "c1", "description": "c1 conflicts"},
                    {"id": "f-misbound", "kind": "misbound", "record_id": "c1", "correct_entity_id": "statue", "description": "c1 misbound"},
                ],
                "status": "needs_repair",
            }
        }
        result = _reviewer_returning(payload).review("idea", _sample_ledger())
        self.assertEqual(result.status, NEEDS_REPAIR)
        kinds = {f.kind for f in result.findings}
        self.assertEqual(kinds, {"omitted", "invented", "contradicted", "misbound"})

    def test_false_positive_control_omits_unsupported_finding(self):
        # A finding without source evidence is a false positive; the bounded
        # reviewer drops unknown finding kinds, and a fully-supported ledger
        # yields ok with no findings.
        payload = {"review": {"findings": [], "status": "ok"}}
        result = _reviewer_returning(payload).review("idea", _sample_ledger())
        self.assertEqual(result.status, OK)
        self.assertEqual(result.findings, [])

    def test_unknown_finding_kind_is_dropped(self):
        payload = {
            "review": {
                "findings": [
                    {"id": "f-bad", "kind": "not-a-kind", "description": "x"},
                    {"id": "f-good", "kind": "invented", "record_id": "r1", "description": "y"},
                ],
                "status": "needs_repair",
            }
        }
        result = _reviewer_returning(payload).review("idea", _sample_ledger())
        self.assertEqual([f.id for f in result.findings], ["f-good"])


class SemanticIntentReviewAmbiguousTests(unittest.TestCase):
    def test_ambiguous_prose_does_not_silently_guess(self):
        # Acceptance: ambiguous input state that does not silently guess.
        payload = {
            "review": {
                "findings": [{"id": "f-amb", "kind": "omitted", "description": "maybe required"}],
                "status": "ambiguous",
            }
        }
        result = review_and_repair("maybe a band?", _sample_ledger(), _reviewer_returning(payload))
        self.assertEqual(result.status, AMBIGUOUS)
        # The ledger is returned unchanged (no targeted repair applied).
        self.assertEqual(len(result.ledger.entities), 2)
        self.assertEqual(len(result.ledger.relations), 1)
        self.assertTrue(result.warnings)


class SemanticIntentRepairTests(unittest.TestCase):
    def test_repair_drops_invented_and_rebinds_misbound_retaining_others(self):
        # Acceptance: repair retains unaffected IDs and records.
        findings = [
            IntentFinding(id="f1", kind="invented", record_id="r1", entity_id="statue", description="r1 not stated"),
            IntentFinding(id="f2", kind="misbound", record_id="c1", entity_id="band", correct_entity_id="statue", description="c1 misbound"),
            IntentFinding(id="f3", kind="omitted", description="missing constraint"),
        ]
        result = repair_ledger(_sample_ledger(), findings)
        self.assertEqual(result.status, NEEDS_REPAIR)
        self.assertEqual(result.applied, ["f1", "f2"])
        self.assertEqual(result.skipped, ["f3"])
        # Invented relation dropped; misbound constraint rebound; entity retained.
        self.assertEqual([r.id for r in result.ledger.relations], [])
        self.assertEqual([c.entity_id for c in result.ledger.constraints if c.id == "c1"], ["statue"])
        self.assertEqual(sorted(e.id for e in result.ledger.entities), ["band", "statue"])

    def test_omitted_finding_is_never_silently_invented(self):
        # Acceptance: omitted records are flagged, not invented.
        findings = [IntentFinding(id="f1", kind="omitted", description="missing")]
        result = repair_ledger(_sample_ledger(), findings)
        self.assertEqual(result.applied, [])
        self.assertEqual(result.skipped, ["f1"])
        # No new entity/constraint was invented.
        self.assertEqual(len(result.ledger.entities), 2)
        self.assertEqual(len(result.ledger.constraints), 1)

    def test_repair_of_unknown_record_is_skipped_not_fatal(self):
        findings = [IntentFinding(id="f1", kind="invented", record_id="nope", description="x")]
        result = repair_ledger(_sample_ledger(), findings)
        self.assertEqual(result.applied, [])
        self.assertEqual(result.skipped, ["f1"])
        # Original ledger retained.
        self.assertEqual(len(result.ledger.relations), 1)

    def test_correlated_bad_repair_is_bounded_not_fatal(self):
        # A repair that would break ledger invariants (misbind to an unknown
        # entity) is skipped with a warning rather than raising.
        findings = [
            IntentFinding(id="f1", kind="misbound", record_id="c1", correct_entity_id="ghost", description="x"),
        ]
        result = repair_ledger(_sample_ledger(), findings)
        self.assertEqual(result.applied, [])
        self.assertEqual(result.skipped, ["f1"])
        self.assertTrue(result.warnings)

    def test_misbound_unknown_record_is_not_reported_as_repaired(self):
        findings = [
            IntentFinding(
                id="f1",
                kind="misbound",
                record_id="missing",
                correct_entity_id="statue",
                description="unknown record",
            ),
        ]

        result = repair_ledger(_sample_ledger(), findings)

        self.assertEqual(result.status, NEEDS_REPAIR)
        self.assertEqual(result.applied, [])
        self.assertEqual(result.skipped, ["f1"])

    def test_review_warnings_and_unresolved_status_survive_repair(self):
        review = IntentReviewResult(
            status=NEEDS_REPAIR,
            findings=[],
            warnings=["judge could not identify an exact record"],
        )

        result = review_and_repair(
            "idea",
            _sample_ledger(),
            _reviewer_returning({"review": {"status": "ok"}}),
            review=review,
        )

        self.assertEqual(result.status, NEEDS_REPAIR)
        self.assertIn("judge could not identify an exact record", result.warnings)

    def test_dropping_entity_cascades_dependents_and_stays_constructible(self):
        # A "invented" finding that drops an entity must also drop the
        # relations/constraints that reference it, so the repaired ledger
        # stays constructible (no dangling references).
        findings = [
            IntentFinding(
                id="f1", kind="invented", record_id="band",
                description="the band is invented",
            ),
        ]
        result = repair_ledger(_sample_ledger(), findings)
        self.assertEqual(result.applied, ["f1"])
        ids = {e["id"] for e in result.ledger.to_dict()["entities"]}
        self.assertNotIn("band", ids)
        # The relation (statue part_of band) and constraint (on band) are
        # cascaded out, so the ledger is constructible.
        self.assertEqual(result.ledger.relations, [])
        self.assertEqual(result.ledger.constraints, [])


class SemanticIntentReviewRetryPolicyTests(unittest.TestCase):
    def test_retry_exhausted_warn_policy_keeps_unreviewed_ledger(self):
        # Acceptance: retry exhaustion stops before expensive reference
        # rendering (default warn policy) with a visible warning.
        reviewer = _reviewer_raising(ValueError("boom"), times=99)
        result = review_and_repair("idea", _sample_ledger(), reviewer, on_retry_exhausted="warn")
        self.assertEqual(result.status, OK)
        self.assertEqual(result.applied, [])
        self.assertTrue(any("warning policy" in w for w in result.warnings))
        # Unreviewed ledger retained.
        self.assertEqual(len(result.ledger.entities), 2)

    def test_retry_exhausted_block_policy_blocks(self):
        reviewer = _reviewer_raising(ValueError("boom"), times=99)
        result = review_and_repair("idea", _sample_ledger(), reviewer, on_retry_exhausted="block")
        self.assertEqual(result.status, AMBIGUOUS)
        self.assertTrue(any("block policy" in w for w in result.warnings))

    def test_bounded_retries_stop_after_max_attempts(self):
        # The judge must not loop forever; it stops after max_attempts.
        reviewer = _reviewer_raising(ValueError("boom"), times=99)
        result = reviewer.review("idea", _sample_ledger(), max_attempts=3)
        self.assertEqual(result.status, FAILED)
        self.assertEqual(result.attempts, DEFAULT_REVIEW_MAX_ATTEMPTS)

    def test_recoverable_failure_retries_then_succeeds(self):
        reviewer = _reviewer_raising(ValueError("boom"), times=2)
        result = reviewer.review("idea", _sample_ledger(), max_attempts=3)
        self.assertEqual(result.status, OK)
        self.assertEqual(result.attempts, 3)


class SemanticIntentReviewDiagnosticsTests(unittest.TestCase):
    def test_diagnostic_identifies_constraint_evidence_and_kind(self):
        # Acceptance: logs identify the constraint, source evidence, and
        # outcome without dumping full prompts.
        finding = IntentFinding(
            id="f1",
            kind="misbound",
            record_id="c1",
            entity_id="band",
            correct_entity_id="statue",
            description="bound to wrong entity",
            source_evidence="immer aus Stein",
        )
        diagnostic = finding.diagnostic
        self.assertIn("misbound", diagnostic)
        self.assertIn("f1", diagnostic)
        self.assertIn("c1", diagnostic)
        self.assertIn("immer aus Stein", diagnostic)


class SemanticIntentReviewDspyAdapterTests(unittest.TestCase):
    def test_dspy_adapter_uses_independent_judge_lm(self):
        # Model/prompt independence: the judge runs under its own LM context,
        # distinct from the extractor. Verify the adapter wires a bounded LM.

        class FakeLM:
            pass

        class FakeRuntime:
            def __init__(self):
                self.predictor = lambda **kw: {"review": {"findings": [], "status": "ok"}}
                self._lm = None

            def predict(self, signature):
                return self.predictor

            def make_lm(self, llm, *, max_tokens=None, task=None):
                self._lm = (max_tokens, task)
                return FakeLM()

            def context(self, *, lm):
                import contextlib

                @contextlib.contextmanager
                def _cm():
                    yield

                return _cm()

        runtime = FakeRuntime()
        reviewer = DspySemanticIntentReviewer(FakeLM(), dspy_runtime=runtime)
        result = reviewer.review("idea", _sample_ledger())
        self.assertEqual(result.status, OK)
        # The judge LM is a compact structured task (judge, bounded tokens).
        self.assertEqual(runtime._lm, (4096, "judge"))


class _FakeArtifactStore:
    def __init__(self, data: dict):
        self._data = data
        self.written: dict[str, dict] = {}

    def read_json(self, path):
        return self._data

    def write_json(self, path, payload):
        self.written[str(path)] = payload


class _FakeReporter:
    def __init__(self):
        self.messages: list[str] = []
        self.warnings: list[tuple[str | None, str]] = []

    def message(self, text):
        self.messages.append(text)

    def warning(self, text, *, title=None):
        self.warnings.append((title, text))


class TestPipelineWiring(unittest.TestCase):
    """The production pipeline stage is wired and tolerant (#545)."""

    def _pipeline(self, **kwargs):
        from feverslop.application.prompt_generation_pipeline import (
            PromptGenerationPipeline,
        )

        return PromptGenerationPipeline(
            llm_factory=lambda app_config: object(),
            prompt_pipeline_factory=lambda llm: object(),
            concept_batcher_factory=lambda llm: object(),
            scene_prompt_builder_factory=lambda llm: object(),
            **kwargs,
        )

    def test_noop_without_extraction_artifact(self):
        pipeline = self._pipeline()
        reporter = _FakeReporter()
        pipeline._review_semantic_intent(
            llm=object(),
            story_idea="A band plays.",
            semantic_intent_json=None,
            semantic_intent_review_json=None,
            artifact_store=_FakeArtifactStore({}),
            log_file=lambda name, path: None,
            reporter=reporter,
        )
        self.assertEqual(reporter.messages, [])

    def test_review_persists_reviewed_artifact(self):
        import tempfile

        from feverslop.domain.semantic_intent import IntentLedger

        ledger = IntentLedger(
            entities=[{"id": "statue", "name": "The Statue", "kind": "object"}],
            relations=[],
            constraints=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "semantic_intent.json"
            src.write_text(
                __import__("json").dumps({"ledger": ledger.to_dict()})
            )
            store = _FakeArtifactStore({"ledger": ledger.to_dict()})
            pipeline = self._pipeline()
            pipeline.intent_review_factory = lambda llm: _reviewer_returning(
                {"review": {"findings": [], "status": "ok"}}
            )
            reporter = _FakeReporter()
            pipeline._review_semantic_intent(
                llm=object(),
                story_idea="A band plays near a statue.",
                semantic_intent_json=src,
                semantic_intent_review_json=Path(tmp) / "review.json",
                artifact_store=store,
                log_file=lambda name, path: None,
                reporter=reporter,
            )
            self.assertEqual(len(store.written), 1)
            payload = next(iter(store.written.values()))
            self.assertEqual(payload["status"], OK)
            self.assertEqual(payload["ledger"], ledger.to_dict())

    def test_review_reports_applied_skipped_and_warning_details(self):
        import tempfile

        from feverslop.domain.semantic_intent import IntentLedger

        ledger = IntentLedger(entities=[{"id": "statue", "kind": "object"}])
        review_payload = {
            "review": {
                "status": "needs_repair",
                "findings": [
                    {
                        "id": "invented-statue",
                        "kind": "invented",
                        "record_id": "statue",
                        "description": "not in source",
                    },
                    {
                        "id": "omitted-band",
                        "kind": "omitted",
                        "record_id": "band",
                        "description": "missing band",
                    },
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "semantic_intent.json"
            src.write_text(__import__("json").dumps({"ledger": ledger.to_dict()}))
            store = _FakeArtifactStore({"ledger": ledger.to_dict()})
            pipeline = self._pipeline(intent_review_factory=lambda _llm: _reviewer_returning(review_payload))
            reporter = _FakeReporter()

            pipeline._review_semantic_intent(
                llm=object(),
                story_idea="A band plays.",
                semantic_intent_json=src,
                semantic_intent_review_json=Path(tmp) / "review.json",
                artifact_store=store,
                log_file=lambda name, path: None,
                reporter=reporter,
            )

        self.assertTrue(any("1 applied, 1 unresolved" in message for message in reporter.messages))
        self.assertEqual(
            [("Semantic intent review", "finding omitted-band skipped: omitted records are not invented")],
            reporter.warnings,
        )

    def test_review_failure_is_bounded_not_fatal(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "semantic_intent.json"
            src.write_text(
                __import__("json").dumps(
                    {"ledger": {"entities": [], "relations": [], "constraints": []}}
                )
            )
            store = _FakeArtifactStore({"ledger": {}})
            pipeline = self._pipeline()
            pipeline.intent_review_factory = lambda llm: _reviewer_raising(
                ValueError("boom"), times=99
            )
            reporter = _FakeReporter()
            pipeline._review_semantic_intent(
                llm=object(),
                story_idea="A band plays.",
                semantic_intent_json=src,
                semantic_intent_review_json=Path(tmp) / "review.json",
                artifact_store=store,
                log_file=lambda name, path: None,
                reporter=reporter,
            )
            # Bounded: a review failure is not raised; the unreviewed ledger
            # is persisted with a warning (warn policy) so rendering proceeds.
            self.assertEqual(len(store.written), 1)
            payload = next(iter(store.written.values()))
            self.assertEqual(payload["status"], OK)
            self.assertTrue(payload["warnings"])
            self.assertTrue(reporter.messages)


if __name__ == "__main__":
    unittest.main()
