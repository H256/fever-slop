import io
import hashlib
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from feverslop.tools.workflow_benchmark import (
    SCORE_NAMES,
    BenchmarkEnvironment,
    UnscoredBenchmarkRun,
    VisualConsistencyBenchmarkResult,
    blind_candidate_labels,
    create_blinded_review_artifact,
    evaluate_promotion,
    ingest_blinded_review_scores,
    run,
    validate_complete_matrix,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "visual_consistency"
    / "benchmark_plan.json"
)

ENVIRONMENT_PAYLOAD = {
    "schema": "feverslop.visual-consistency-environment/v1",
    "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    "audio_sha256": "1" * 64,
    "reference_sha256": {"hero": "2" * 64, "studio": "3" * 64},
    "model_sha256": {"ltx": "4" * 64},
    "workflow_sha256": {
        "msr-default": "5" * 64,
        "msr-startframe": "6" * 64,
    },
    "profiles": ["msr-default", "msr-startframe"],
    "config_sha256": "7" * 64,
    "hardware_id": "test-gpu-24gb",
}
ENVIRONMENT = BenchmarkEnvironment.from_dict(ENVIRONMENT_PAYLOAD)


def _write_environment(root: Path) -> Path:
    path = root / "environment.json"
    path.write_text(json.dumps(ENVIRONMENT_PAYLOAD), encoding="utf-8")
    return path


def _result(
    *,
    label: str = "candidate-a",
    scene: int = 1,
    wall_time_seconds: float = 100.0,
    scores: dict[str, int] | None = None,
    peak_vram_mb: int | None = 20_000,
    oom: bool = False,
    evidence_digest: str | None = "c" * 64,
) -> dict:
    payload = {
        "candidate_label": label,
        "scene": scene,
        "workflow_profile": (
            "msr-default" if label == "baseline" else "msr-startframe"
        ),
        "prepared_workflow_sha256": "a" * 64,
        "contract_fingerprint": "b" * 64,
        "wall_time_seconds": wall_time_seconds,
        "peak_vram_mb": peak_vram_mb,
        "scores": scores
        or {
            "identity": 4,
            "wardrobe": 4,
            "location": 4,
            "palette": 3,
            "transition": 3,
        },
        "preflight_errors": [],
        "manifest_errors": [],
        "oom": oom,
        "environment_fingerprint": ENVIRONMENT.fingerprint,
    }
    if evidence_digest is not None:
        payload["blinded_evidence_schema"] = (
            "feverslop.visual-consistency-evidence/v1"
        )
        payload["blinded_evidence_sha256"] = evidence_digest
    return payload


def _unscored_result(*, label: str, scene: int) -> dict:
    payload = _result(label=label, scene=scene, evidence_digest=None)
    payload.pop("scores")
    return payload


def _evidenced(
    results: list[VisualConsistencyBenchmarkResult],
) -> tuple[
    tuple[VisualConsistencyBenchmarkResult, ...],
    dict,
    dict,
]:
    review, mapping = create_blinded_review_artifact(
        results, environment=ENVIRONMENT, seed=4707
    )
    profiles = {
        item["candidate_label"]: item["workflow_profile"]
        for item in mapping["mappings"]
    }
    scores = {
        (item.workflow_profile, item.scene): dict(item.scores)
        for item in results
    }
    for entry in review["entries"]:
        entry["scores"] = scores[
            (profiles[entry["candidate_label"]], entry["scene"])
        ]
    return (
        ingest_blinded_review_scores(
            results, review, mapping, environment=ENVIRONMENT
        ),
        review,
        mapping,
    )


class VisualConsistencyBenchmarkResultTests(unittest.TestCase):
    def test_environment_rejects_non_fixture_digest(self):
        payload = dict(ENVIRONMENT_PAYLOAD, fixture_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "fixture SHA-256"):
            BenchmarkEnvironment.from_dict(payload)

    def test_environment_requires_exact_v1_schema(self):
        for schema in ("feverslop.visual-consistency-environment/v2", "", None):
            with self.subTest(schema=schema):
                payload = dict(ENVIRONMENT_PAYLOAD, schema=schema)
                with self.assertRaisesRegex(ValueError, "schema"):
                    BenchmarkEnvironment.from_dict(payload)

    def test_unscored_run_accepts_absent_or_null_scores_only(self):
        absent = _result(evidence_digest=None)
        absent.pop("scores")
        null_scores = dict(absent, scores=None)

        self.assertEqual(1, UnscoredBenchmarkRun.from_dict(absent).scene)
        self.assertEqual(1, UnscoredBenchmarkRun.from_dict(null_scores).scene)
        with self.assertRaisesRegex(ValueError, "unscored"):
            UnscoredBenchmarkRun.from_dict(_result(evidence_digest=None))

    def test_consistency_scores_are_complete_and_bounded(self):
        result = VisualConsistencyBenchmarkResult.from_dict(_result())

        self.assertEqual(set(SCORE_NAMES), set(result.scores))
        self.assertEqual(4, result.scores["identity"])

        invalid = _result()
        invalid["scores"]["identity"] = 6
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            VisualConsistencyBenchmarkResult.from_dict(invalid)

    def test_scores_require_exact_integer_dimensions(self):
        for scores, message in (
            (
                {
                    "identity": 4,
                    "wardrobe": 4,
                    "location": 4,
                    "palette": 3,
                },
                "exactly",
            ),
            (
                {
                    "identity": 4,
                    "wardrobe": 4,
                    "location": 4,
                    "palette": 3,
                    "transition": 3,
                    "motion": 5,
                },
                "exactly",
            ),
            (
                {
                    "identity": True,
                    "wardrobe": 4,
                    "location": 4,
                    "palette": 3,
                    "transition": 3,
                },
                "integer",
            ),
        ):
            with self.subTest(scores=scores):
                payload = _result(scores=scores)
                with self.assertRaisesRegex(ValueError, message):
                    VisualConsistencyBenchmarkResult.from_dict(payload)

    def test_records_required_provenance_runtime_and_optional_vram(self):
        result = VisualConsistencyBenchmarkResult.from_dict(
            _result(peak_vram_mb=None)
        )
        without_vram = _result()
        without_vram.pop("peak_vram_mb")

        self.assertEqual("a" * 64, result.prepared_workflow_sha256)
        self.assertEqual("b" * 64, result.contract_fingerprint)
        self.assertEqual(100.0, result.wall_time_seconds)
        self.assertIsNone(result.peak_vram_mb)
        self.assertIsNone(
            VisualConsistencyBenchmarkResult.from_dict(without_vram).peak_vram_mb
        )

        for field in ("prepared_workflow_sha256", "contract_fingerprint"):
            payload = _result()
            payload[field] = "not-a-hash"
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "SHA-256"
            ):
                VisualConsistencyBenchmarkResult.from_dict(payload)

    def test_rejects_unknown_result_fields(self):
        payload = _result()
        payload["operator_guess"] = "baseline"

        with self.assertRaisesRegex(ValueError, "Unknown benchmark result fields"):
            VisualConsistencyBenchmarkResult.from_dict(payload)


class VisualConsistencyPromotionTests(unittest.TestCase):
    @staticmethod
    def _promotion(
        baseline: list[VisualConsistencyBenchmarkResult],
        candidate: list[VisualConsistencyBenchmarkResult],
    ):
        evidenced, review, mapping = _evidenced([*baseline, *candidate])
        return evaluate_promotion(
            evidenced[:6],
            evidenced[6:],
            review=review,
            sealed_mapping=mapping,
            environment=ENVIRONMENT,
        )

    def test_promotes_median_identity_improvement_without_regression(self):
        baseline = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label="baseline", scene=scene)
            )
            for scene in range(1, 7)
        ]
        candidate = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(
                    scene=scene,
                    scores={
                        "identity": 5,
                        "wardrobe": 4,
                        "location": 4,
                        "palette": 3,
                        "transition": 3,
                    },
                )
            )
            for scene in range(1, 7)
        ]

        decision = self._promotion(baseline, candidate)

        self.assertTrue(decision.promote)
        self.assertEqual((), decision.failures)

    def test_equal_quality_requires_fifteen_percent_wall_time_reduction(self):
        baseline = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label="baseline", scene=scene, wall_time_seconds=100)
            )
            for scene in range(1, 7)
        ]
        fast = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(scene=scene, wall_time_seconds=85)
            )
            for scene in range(1, 7)
        ]
        slow = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(scene=scene, wall_time_seconds=85.01)
            )
            for scene in range(1, 7)
        ]

        self.assertTrue(self._promotion(baseline, fast).promote)
        self.assertFalse(self._promotion(baseline, slow).promote)

    def test_errors_regressions_and_new_supported_profile_oom_block_promotion(self):
        baseline = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label="baseline", scene=scene)
            )
            for scene in range(1, 7)
        ]
        payloads = [_result(scene=scene, wall_time_seconds=80) for scene in range(1, 7)]
        payloads[0]["manifest_errors"] = ["workflow hash mismatch"]
        payloads[1]["preflight_errors"] = ["missing actor"]
        payloads[2]["oom"] = True
        for payload in payloads[2:]:
            payload["scores"]["wardrobe"] = 1
        candidate = [
            VisualConsistencyBenchmarkResult.from_dict(payload)
            for payload in payloads
        ]

        decision = self._promotion(baseline, candidate)

        self.assertFalse(decision.promote)
        self.assertTrue(any("preflight" in item for item in decision.failures))
        self.assertTrue(any("manifest" in item for item in decision.failures))
        self.assertTrue(any("OOM" in item for item in decision.failures))
        self.assertTrue(any("wardrobe" in item for item in decision.failures))

    def test_incomplete_mixed_and_unmatched_matrices_are_inconclusive(self):
        complete_baseline = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label="baseline", scene=scene)
            )
            for scene in range(1, 7)
        ]
        complete_candidate = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(scene=scene, wall_time_seconds=80)
            )
            for scene in range(1, 7)
        ]
        mixed_candidate = list(complete_candidate)
        mixed_payload = mixed_candidate[-1].to_dict()
        mixed_payload["workflow_profile"] = "msr-default"
        mixed_candidate[-1] = VisualConsistencyBenchmarkResult.from_dict(
            mixed_payload
        )
        unmatched_candidate = list(complete_candidate)
        unmatched_payload = unmatched_candidate[-1].to_dict()
        unmatched_payload["scene"] = 7
        unmatched_candidate[-1] = VisualConsistencyBenchmarkResult.from_dict(
            unmatched_payload
        )

        evidenced, review, mapping = _evidenced(
            [*complete_baseline, *complete_candidate]
        )
        evidenced_baseline = list(evidenced[:6])
        evidenced_candidate = list(evidenced[6:])
        mixed_evidenced = list(evidenced_candidate)
        mixed_payload = mixed_evidenced[-1].to_dict()
        mixed_payload["workflow_profile"] = "msr-default"
        mixed_evidenced[-1] = VisualConsistencyBenchmarkResult.from_dict(
            mixed_payload
        )
        unmatched_evidenced = list(evidenced_candidate)
        unmatched_payload = unmatched_evidenced[-1].to_dict()
        unmatched_payload["scene"] = 7
        unmatched_evidenced[-1] = VisualConsistencyBenchmarkResult.from_dict(
            unmatched_payload
        )
        decisions = (
            evaluate_promotion(
                evidenced_baseline[:-1],
                evidenced_candidate,
                review=review,
                sealed_mapping=mapping,
                environment=ENVIRONMENT,
            ),
            evaluate_promotion(
                evidenced_baseline,
                mixed_evidenced,
                review=review,
                sealed_mapping=mapping,
                environment=ENVIRONMENT,
            ),
            evaluate_promotion(
                evidenced_baseline,
                unmatched_evidenced,
                review=review,
                sealed_mapping=mapping,
                environment=ENVIRONMENT,
            ),
        )

        for decision in decisions:
            with self.subTest(failures=decision.failures):
                self.assertFalse(decision.promote)
                self.assertTrue(decision.failures)

    def test_promotion_without_review_artifacts_is_inconclusive(self):
        baseline = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label="baseline", scene=scene)
            )
            for scene in range(1, 7)
        ]
        candidate = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(scene=scene, wall_time_seconds=80)
            )
            for scene in range(1, 7)
        ]

        decision = evaluate_promotion(baseline, candidate)

        self.assertFalse(decision.promote)
        self.assertTrue(
            any("blinded evidence artifacts" in item for item in decision.failures)
        )

    def test_promotion_rejects_reversed_baseline_and_candidate_arguments(self):
        baseline = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label="baseline", scene=scene)
            )
            for scene in range(1, 7)
        ]
        candidate = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(scene=scene, wall_time_seconds=80)
            )
            for scene in range(1, 7)
        ]
        evidenced, review, mapping = _evidenced([*baseline, *candidate])

        decision = evaluate_promotion(
            evidenced[6:],
            evidenced[:6],
            review=review,
            sealed_mapping=mapping,
            environment=ENVIRONMENT,
        )

        self.assertFalse(decision.promote)
        self.assertTrue(any("baseline profile" in item for item in decision.failures))

    def test_complete_matrix_requires_fixture_profiles_and_consistent_opaque_labels(self):
        results = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label="baseline", scene=scene)
            )
            for scene in range(1, 7)
        ] + [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(scene=scene, wall_time_seconds=80)
            )
            for scene in range(1, 7)
        ]

        self.assertFalse(validate_complete_matrix(results).valid)
        self.assertTrue(
            any(
                "blinded evidence artifacts" in failure
                for failure in validate_complete_matrix(results).failures
            )
        )

        mixed_label = [item.to_dict() for item in results]
        mixed_label[-1]["candidate_label"] = "candidate-b"
        decision = validate_complete_matrix(
            [
                VisualConsistencyBenchmarkResult.from_dict(item)
                for item in mixed_label
            ]
        )
        self.assertFalse(decision.valid)
        self.assertTrue(any("label" in failure for failure in decision.failures))


class VisualConsistencyBenchmarkFixtureTests(unittest.TestCase):
    def test_fixed_fixture_has_six_scenes_and_frozen_render_inputs(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        scenes = payload["scenes"]

        self.assertEqual(
            [
                "same-actor-location-cut",
                "same-actor-new-location",
                "valid-continuous-transition",
                "two-actor-scene",
                "instrumental-b-roll-no-lip-sync",
                "return-to-original-actor-location",
            ],
            [scene["benchmark_case"] for scene in scenes],
        )
        self.assertEqual(6, len({scene["seed"] for scene in scenes}))
        for scene in scenes:
            self.assertTrue(scene["prompt"])
            self.assertTrue(scene["references"]["actor_ids"])
            self.assertTrue(scene["references"]["location_id"])
            self.assertEqual(2, len(scene["audio_window_seconds"]))
            self.assertGreater(scene["audio_window_seconds"][1], scene["audio_window_seconds"][0])
            self.assertGreater(scene["width"], 0)
            self.assertGreater(scene["height"], 0)
            self.assertTrue(scene["workflow_profiles"])
        self.assertFalse(scenes[4]["lip_sync"])

    def test_blinded_labels_are_seeded_unique_and_do_not_reveal_names(self):
        first = blind_candidate_labels(
            ("baseline-profile", "candidate-profile"), seed=4707
        )
        second = blind_candidate_labels(
            ("baseline-profile", "candidate-profile"), seed=4707
        )

        self.assertEqual(first, second)
        self.assertEqual({"candidate-a", "candidate-b"}, set(first.values()))
        self.assertNotEqual(first["baseline-profile"], "baseline-profile")

    def test_cli_validates_recorded_results_without_creating_fake_results(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(
                json.dumps({"results": [_result()]}),
                encoding="utf-8",
            )
            before = path.read_bytes()
            output = io.StringIO()

            with redirect_stdout(output):
                status = run([str(path), "--record-only", "--json"])

            self.assertEqual(0, status)
            self.assertEqual(
                {"mode": "records", "valid": True, "result_count": 1},
                json.loads(output.getvalue()),
            )
            self.assertEqual(before, path.read_bytes())

    def test_cli_complete_evidence_rejects_singleton(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            path = root / "results.json"
            path.write_text(
                json.dumps({"results": [_result()]}),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                status = run([
                    str(path),
                    "--environment",
                    str(environment_path),
                    "--json",
                ])

        payload = json.loads(output.getvalue())
        self.assertEqual(2, status)
        self.assertEqual("matrix", payload["mode"])
        self.assertFalse(payload["valid"])
        self.assertTrue(payload["failures"])

    def test_cli_complete_evidence_reports_missing_or_malformed_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            results_path.write_text(
                json.dumps({
                    "results": [
                        _result(label=label, scene=scene)
                        for label in ("baseline", "candidate")
                        for scene in range(1, 7)
                    ]
                }),
                encoding="utf-8",
            )
            malformed = root / "malformed.json"
            malformed.write_text("{bad", encoding="utf-8")
            missing = root / "missing.json"

            for review_path, mapping_path in (
                (missing, malformed),
                (malformed, missing),
            ):
                with self.subTest(
                    review_path=review_path,
                    mapping_path=mapping_path,
                ):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        status = run([
                            str(results_path),
                            "--review",
                            str(review_path),
                            "--sealed-mapping",
                            str(mapping_path),
                            "--environment",
                            str(environment_path),
                            "--json",
                        ])
                    payload = json.loads(output.getvalue())
                    self.assertEqual(1, status)
                    self.assertFalse(payload["valid"])
                    self.assertTrue(payload["error"])

    def test_review_artifact_is_randomized_sanitized_and_round_trips_scores(self):
        source = tuple(
            VisualConsistencyBenchmarkResult.from_dict(
                _result(label=label, scene=scene)
            )
            for label in ("baseline", "candidate")
            for scene in range(1, 7)
        )

        first_review, first_mapping = create_blinded_review_artifact(
            source, environment=ENVIRONMENT, seed=4707
        )
        second_review, second_mapping = create_blinded_review_artifact(
            source, environment=ENVIRONMENT, seed=4707
        )

        self.assertEqual(first_review, second_review)
        self.assertEqual(first_mapping, second_mapping)
        serialized_review = json.dumps(first_review, sort_keys=True)
        for secret in (
            "baseline",
            "candidate",
            "msr-default",
            "msr-startframe",
            "a" * 64,
            "b" * 64,
        ):
            if secret in {"baseline", "candidate"}:
                self.assertNotIn(f'"{secret}"', serialized_review)
            else:
                self.assertNotIn(secret, serialized_review)
        self.assertEqual(
            {"candidate_label", "scene", "scores"},
            set(first_review["entries"][0]),
        )
        self.assertEqual(
            {name: None for name in SCORE_NAMES},
            first_review["entries"][0]["scores"],
        )
        self.assertNotEqual(
            [(item.candidate_label, item.scene) for item in source],
            [
                (item["candidate_label"], item["scene"])
                for item in first_review["entries"]
            ],
        )
        completed_review = json.loads(json.dumps(first_review))
        source_scores = {
            (item.workflow_profile, item.scene): dict(item.scores)
            for item in source
        }
        opaque_profiles = {
            item["candidate_label"]: item["workflow_profile"]
            for item in first_mapping["mappings"]
        }
        for entry in completed_review["entries"]:
            entry["scores"] = source_scores[
                (opaque_profiles[entry["candidate_label"]], entry["scene"])
            ]
        restored = ingest_blinded_review_scores(
            source,
            completed_review,
            first_mapping,
            environment=ENVIRONMENT,
        )

        self.assertEqual(
            [dict(item.scores) for item in source],
            [dict(item.scores) for item in restored],
        )
        self.assertEqual(
            {
                "feverslop.visual-consistency-evidence/v1"
            },
            {item.blinded_evidence_schema for item in restored},
        )
        evidence_digests = {
            item.blinded_evidence_sha256 for item in restored
        }
        self.assertEqual(1, len(evidence_digests))
        self.assertTrue(validate_complete_matrix(
            restored,
            review=completed_review,
            sealed_mapping=first_mapping,
            environment=ENVIRONMENT,
        ).valid)
        tampered_payload = restored[0].to_dict()
        tampered_payload["wall_time_seconds"] += 1
        tampered = (
            VisualConsistencyBenchmarkResult.from_dict(tampered_payload),
            *restored[1:],
        )
        self.assertFalse(validate_complete_matrix(
            tampered,
            review=completed_review,
            sealed_mapping=first_mapping,
            environment=ENVIRONMENT,
        ).valid)

    def test_handcrafted_scores_are_not_complete_promotion_evidence(self):
        legacy = [
            VisualConsistencyBenchmarkResult.from_dict(
                _result(
                    label=label,
                    scene=scene,
                    evidence_digest=None,
                )
            )
            for label in ("baseline", "candidate")
            for scene in range(1, 7)
        ]

        decision = validate_complete_matrix(legacy)

        self.assertFalse(decision.valid)
        self.assertTrue(any("blinded evidence" in item for item in decision.failures))

    def test_cli_writes_review_and_sealed_mapping_separately(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            review_path = root / "review.json"
            mapping_path = root / "sealed.json"
            results_path.write_text(
                json.dumps(
                    {
                        "results": [
                            _unscored_result(label=label, scene=scene)
                            for label in ("baseline", "candidate")
                            for scene in range(1, 7)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                status = run(
                    [
                        str(results_path),
                        "--create-review",
                        str(review_path),
                        "--sealed-mapping",
                        str(mapping_path),
                        "--seed",
                        "4707",
                        "--environment",
                        str(environment_path),
                        "--json",
                    ]
                )

            review = json.loads(review_path.read_text(encoding="utf-8"))
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

        self.assertEqual(0, status)
        self.assertEqual("review", json.loads(output.getvalue())["mode"])
        self.assertEqual(
            "feverslop.visual-consistency-review/v1", review["schema"]
        )
        self.assertEqual(
            "feverslop.visual-consistency-review-map/v1", mapping["schema"]
        )
        self.assertEqual(
            {name: None for name in SCORE_NAMES},
            review["entries"][0]["scores"],
        )

    def test_cli_rejects_source_overwrite_before_writing_any_artifact(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            mapping_path = root / "sealed.json"
            original = json.dumps(
                {
                    "results": [
                        _unscored_result(label=label, scene=scene)
                        for label in ("baseline", "candidate")
                        for scene in range(1, 7)
                    ]
                }
            )
            results_path.write_text(original, encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                status = run(
                    [
                        str(results_path),
                        "--create-review",
                        str(results_path),
                        "--sealed-mapping",
                        str(mapping_path),
                        "--seed",
                        "4707",
                        "--environment",
                        str(environment_path),
                        "--json",
                    ]
                )

            self.assertEqual(1, status)
            self.assertEqual(original, results_path.read_text(encoding="utf-8"))
            self.assertFalse(mapping_path.exists())

    def test_cli_rejects_same_or_aliased_artifact_paths_without_writes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "results": [
                            _unscored_result(label=label, scene=scene)
                            for label in ("baseline", "candidate")
                            for scene in range(1, 7)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            alias_path = root / "results-alias.json"
            os.link(results_path, alias_path)
            same_output = root / "same-output.json"

            cases = (
                (alias_path, root / "alias-map.json"),
                (same_output, same_output),
            )
            for review_path, mapping_path in cases:
                with self.subTest(
                    review_path=review_path,
                    mapping_path=mapping_path,
                ):
                    before = {
                        path.name: path.read_bytes()
                        for path in root.iterdir()
                        if path.is_file()
                    }
                    output = io.StringIO()
                    with redirect_stdout(output):
                        status = run(
                            [
                                str(results_path),
                                "--create-review",
                                str(review_path),
                                "--sealed-mapping",
                                str(mapping_path),
                                "--seed",
                                "4707",
                                "--environment",
                                str(environment_path),
                                "--json",
                            ]
                        )
                    self.assertEqual(1, status)
                    self.assertEqual(
                        before,
                        {
                            path.name: path.read_bytes()
                            for path in root.iterdir()
                            if path.is_file()
                        },
                    )

    def test_cli_rejects_environment_alias_to_source_or_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "results.json"
            review_path = root / "review.json"
            mapping_path = root / "sealed.json"
            results_path.write_text(json.dumps({"results": []}), encoding="utf-8")
            environment_alias = root / "environment-alias.json"
            os.link(results_path, environment_alias)
            output = io.StringIO()

            with redirect_stdout(output):
                status = run([
                    str(results_path), "--create-review", str(review_path),
                    "--sealed-mapping", str(mapping_path), "--seed", "4707",
                    "--environment", str(environment_alias), "--json",
                ])

            self.assertEqual(1, status)
            self.assertFalse(review_path.exists())
            self.assertFalse(mapping_path.exists())

    def test_cli_rolls_back_when_second_artifact_replace_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            review_path = root / "review.json"
            mapping_path = root / "sealed.json"
            results_path.write_text(
                json.dumps({
                    "results": [
                        _unscored_result(label=label, scene=scene)
                        for label in ("baseline", "candidate")
                        for scene in range(1, 7)
                    ]
                }),
                encoding="utf-8",
            )
            real_replace = os.replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected second replace failure")
                return real_replace(source, destination)

            output = io.StringIO()
            with patch(
                "feverslop.tools.workflow_benchmark.os.replace",
                side_effect=fail_second_replace,
            ), redirect_stdout(output):
                status = run([
                    str(results_path),
                    "--create-review",
                    str(review_path),
                    "--sealed-mapping",
                    str(mapping_path),
                    "--seed",
                    "4707",
                    "--environment",
                    str(environment_path),
                    "--json",
                ])

            self.assertEqual(1, status)
            self.assertFalse(review_path.exists())
            self.assertFalse(mapping_path.exists())
            self.assertEqual(
                {"environment.json", "results.json"},
                {path.name for path in root.iterdir()},
            )

    def test_cli_retains_backup_when_install_and_restore_fail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            review_path = root / "review.json"
            mapping_path = root / "sealed.json"
            review_path.write_text("old-review", encoding="utf-8")
            mapping_path.write_text("old-mapping", encoding="utf-8")
            results_path.write_text(json.dumps({"results": [
                _unscored_result(label=label, scene=scene)
                for label in ("baseline", "candidate")
                for scene in range(1, 7)
            ]}), encoding="utf-8")
            real_replace = os.replace
            calls = 0

            def fail_install_and_restore(source, destination):
                nonlocal calls
                calls += 1
                if calls in {4, 5}:
                    raise OSError(f"injected replace failure {calls}")
                return real_replace(source, destination)

            output = io.StringIO()
            with patch(
                "feverslop.tools.workflow_benchmark.os.replace",
                side_effect=fail_install_and_restore,
            ), redirect_stdout(output):
                status = run([
                    str(results_path), "--create-review", str(review_path),
                    "--sealed-mapping", str(mapping_path), "--seed", "4707",
                    "--environment", str(environment_path), "--json",
                ])

            payload = json.loads(output.getvalue())
            self.assertEqual(1, status)
            self.assertIn("retained recovery paths", payload["error"])
            self.assertEqual("old-mapping", mapping_path.read_text(encoding="utf-8"))
            retained = [
                path for path in root.iterdir()
                if path.name not in {
                    "environment.json", "results.json", "sealed.json"
                }
            ]
            self.assertEqual(1, len(retained))
            self.assertEqual(b"old-review", retained[0].read_bytes())

    def test_cli_rollback_continues_after_partial_unlink_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            review_path = root / "review.json"
            mapping_path = root / "sealed.json"
            review_path.write_text("old-review", encoding="utf-8")
            mapping_path.write_text("old-mapping", encoding="utf-8")
            results_path.write_text(json.dumps({"results": [
                _unscored_result(label=label, scene=scene)
                for label in ("baseline", "candidate")
                for scene in range(1, 7)
            ]}), encoding="utf-8")
            real_replace = os.replace
            real_unlink = Path.unlink
            replace_calls = 0

            def fail_second_install(source, destination):
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 4:
                    raise OSError("injected install failure")
                return real_replace(source, destination)

            def fail_installed_unlink(path, *args, **kwargs):
                if path == review_path:
                    raise OSError("injected unlink failure")
                return real_unlink(path, *args, **kwargs)

            output = io.StringIO()
            with patch(
                "feverslop.tools.workflow_benchmark.os.replace",
                side_effect=fail_second_install,
            ), patch.object(Path, "unlink", fail_installed_unlink), redirect_stdout(output):
                status = run([
                    str(results_path), "--create-review", str(review_path),
                    "--sealed-mapping", str(mapping_path), "--seed", "4707",
                    "--environment", str(environment_path), "--json",
                ])

            self.assertEqual(1, status)
            self.assertIn("injected unlink failure", output.getvalue())
            self.assertEqual("old-review", review_path.read_text(encoding="utf-8"))
            self.assertEqual("old-mapping", mapping_path.read_text(encoding="utf-8"))

    def test_cli_rejects_directory_artifact_target_without_partial_write(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment_path = _write_environment(root)
            results_path = root / "results.json"
            review_path = root / "review.json"
            mapping_path = root / "sealed.json"
            review_path.mkdir()
            results_path.write_text(
                json.dumps({
                    "results": [
                        _unscored_result(label=label, scene=scene)
                        for label in ("baseline", "candidate")
                        for scene in range(1, 7)
                    ]
                }),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                status = run([
                    str(results_path),
                    "--create-review",
                    str(review_path),
                    "--sealed-mapping",
                    str(mapping_path),
                    "--seed",
                    "4707",
                    "--environment",
                    str(environment_path),
                    "--json",
                ])

            self.assertEqual(1, status)
            self.assertTrue(review_path.is_dir())
            self.assertFalse(mapping_path.exists())


if __name__ == "__main__":
    unittest.main()
