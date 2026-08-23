from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any

SCORE_NAMES = (
    "identity",
    "wardrobe",
    "location",
    "palette",
    "transition",
)
FIXED_SCENE_IDS = frozenset(range(1, 7))
FIXED_WORKFLOW_PROFILES = ("msr-default", "msr-startframe")
REVIEW_SCHEMA = "feverslop.visual-consistency-review/v1"
REVIEW_MAP_SCHEMA = "feverslop.visual-consistency-review-map/v1"
EVIDENCE_SCHEMA = "feverslop.visual-consistency-evidence/v1"
ENVIRONMENT_SCHEMA = "feverslop.visual-consistency-environment/v1"
_RESULT_FIELDS = {
    "candidate_label",
    "scene",
    "workflow_profile",
    "prepared_workflow_sha256",
    "contract_fingerprint",
    "wall_time_seconds",
    "peak_vram_mb",
    "scores",
    "preflight_errors",
    "manifest_errors",
    "oom",
    "blinded_evidence_schema",
    "blinded_evidence_sha256",
    "environment_fingerprint",
}
_OPTIONAL_RESULT_FIELDS = {
    "peak_vram_mb",
    "blinded_evidence_schema",
    "blinded_evidence_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class BenchmarkEnvironment:
    fixture_sha256: str
    audio_sha256: str
    reference_sha256: Mapping[str, str]
    model_sha256: Mapping[str, str]
    workflow_sha256: Mapping[str, str]
    profiles: tuple[str, ...]
    config_sha256: str
    hardware_id: str
    fingerprint: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkEnvironment:
        expected = {
            "schema", "fixture_sha256", "audio_sha256", "reference_sha256",
            "model_sha256", "workflow_sha256", "profiles", "config_sha256",
            "hardware_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("Benchmark environment fields do not match schema")
        if payload["schema"] != ENVIRONMENT_SCHEMA:
            raise ValueError("Unsupported benchmark environment schema")
        fixture_sha = _sha256(payload["fixture_sha256"], "fixture_sha256")
        if fixture_sha != _fixed_fixture_sha256():
            raise ValueError("Benchmark environment fixture SHA-256 does not match")
        profiles = payload["profiles"]
        if not isinstance(profiles, list) or tuple(profiles) != FIXED_WORKFLOW_PROFILES:
            raise ValueError("Benchmark environment profiles do not match fixture")
        canonical = {
            "schema": ENVIRONMENT_SCHEMA,
            "fixture_sha256": fixture_sha,
            "audio_sha256": _sha256(payload["audio_sha256"], "audio_sha256"),
            "reference_sha256": _hash_mapping(payload["reference_sha256"], "reference_sha256"),
            "model_sha256": _hash_mapping(payload["model_sha256"], "model_sha256"),
            "workflow_sha256": _hash_mapping(payload["workflow_sha256"], "workflow_sha256"),
            "profiles": list(profiles),
            "config_sha256": _sha256(payload["config_sha256"], "config_sha256"),
            "hardware_id": _nonblank(payload["hardware_id"], "hardware_id"),
        }
        if set(canonical["workflow_sha256"]) != set(FIXED_WORKFLOW_PROFILES):
            raise ValueError("Environment workflow hashes must cover fixed profiles")
        fingerprint = hashlib.sha256(_canonical_json(canonical)).hexdigest()
        return cls(
            fixture_sha256=fixture_sha,
            audio_sha256=canonical["audio_sha256"],
            reference_sha256=MappingProxyType(canonical["reference_sha256"]),
            model_sha256=MappingProxyType(canonical["model_sha256"]),
            workflow_sha256=MappingProxyType(canonical["workflow_sha256"]),
            profiles=tuple(profiles),
            config_sha256=canonical["config_sha256"],
            hardware_id=canonical["hardware_id"],
            fingerprint=fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ENVIRONMENT_SCHEMA,
            "fixture_sha256": self.fixture_sha256,
            "audio_sha256": self.audio_sha256,
            "reference_sha256": dict(self.reference_sha256),
            "model_sha256": dict(self.model_sha256),
            "workflow_sha256": dict(self.workflow_sha256),
            "profiles": list(self.profiles),
            "config_sha256": self.config_sha256,
            "hardware_id": self.hardware_id,
        }


@dataclass(frozen=True)
class VisualConsistencyBenchmarkResult:
    candidate_label: str
    scene: int
    workflow_profile: str
    prepared_workflow_sha256: str
    contract_fingerprint: str
    wall_time_seconds: float
    peak_vram_mb: int | None
    scores: Mapping[str, int]
    preflight_errors: tuple[str, ...]
    manifest_errors: tuple[str, ...]
    oom: bool
    environment_fingerprint: str
    blinded_evidence_schema: str | None = None
    blinded_evidence_sha256: str | None = None

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any],
    ) -> VisualConsistencyBenchmarkResult:
        if not isinstance(payload, Mapping):
            raise ValueError("Benchmark result must be an object")
        unknown = sorted(set(payload) - _RESULT_FIELDS)
        if unknown:
            raise ValueError(
                "Unknown benchmark result fields: " + ", ".join(unknown),
            )
        missing = sorted(_RESULT_FIELDS - _OPTIONAL_RESULT_FIELDS - set(payload))
        if missing:
            raise ValueError(
                "Missing benchmark result fields: " + ", ".join(missing),
            )

        candidate_label = _nonblank(payload["candidate_label"], "candidate_label")
        workflow_profile = _nonblank(payload["workflow_profile"], "workflow_profile")
        scene = _positive_integer(payload["scene"], "scene")
        prepared_hash = _sha256(
            payload["prepared_workflow_sha256"], "prepared_workflow_sha256",
        )
        fingerprint = _sha256(
            payload["contract_fingerprint"], "contract_fingerprint",
        )
        wall_time = _positive_number(
            payload["wall_time_seconds"], "wall_time_seconds",
        )
        peak_vram = payload.get("peak_vram_mb")
        if peak_vram is not None:
            peak_vram = _nonnegative_integer(peak_vram, "peak_vram_mb")
        scores = _scores(payload["scores"])
        preflight_errors = _errors(payload["preflight_errors"], "preflight_errors")
        manifest_errors = _errors(payload["manifest_errors"], "manifest_errors")
        oom = payload["oom"]
        if not isinstance(oom, bool):
            raise ValueError("oom must be a boolean")
        environment_fingerprint = _sha256(
            payload["environment_fingerprint"], "environment_fingerprint",
        )
        evidence_schema = payload.get("blinded_evidence_schema")
        evidence_sha256 = payload.get("blinded_evidence_sha256")
        if (evidence_schema is None) != (evidence_sha256 is None):
            raise ValueError(
                "blinded evidence schema and SHA-256 must appear together",
            )
        if evidence_schema is not None:
            if evidence_schema != EVIDENCE_SCHEMA:
                raise ValueError("Unsupported blinded evidence schema")
            evidence_sha256 = _sha256(
                evidence_sha256, "blinded_evidence_sha256",
            )

        return cls(
            candidate_label=candidate_label,
            scene=scene,
            workflow_profile=workflow_profile,
            prepared_workflow_sha256=prepared_hash,
            contract_fingerprint=fingerprint,
            wall_time_seconds=wall_time,
            peak_vram_mb=peak_vram,
            scores=MappingProxyType(scores),
            preflight_errors=preflight_errors,
            manifest_errors=manifest_errors,
            oom=oom,
            environment_fingerprint=environment_fingerprint,
            blinded_evidence_schema=evidence_schema,
            blinded_evidence_sha256=evidence_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "candidate_label": self.candidate_label,
            "scene": self.scene,
            "workflow_profile": self.workflow_profile,
            "prepared_workflow_sha256": self.prepared_workflow_sha256,
            "contract_fingerprint": self.contract_fingerprint,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "scores": dict(self.scores),
            "preflight_errors": list(self.preflight_errors),
            "manifest_errors": list(self.manifest_errors),
            "oom": self.oom,
            "environment_fingerprint": self.environment_fingerprint,
        }
        if self.blinded_evidence_schema is not None:
            payload["blinded_evidence_schema"] = self.blinded_evidence_schema
            payload["blinded_evidence_sha256"] = self.blinded_evidence_sha256
        return payload


@dataclass(frozen=True)
class UnscoredBenchmarkRun:
    candidate_label: str
    scene: int
    workflow_profile: str
    prepared_workflow_sha256: str
    contract_fingerprint: str
    wall_time_seconds: float
    peak_vram_mb: int | None
    preflight_errors: tuple[str, ...]
    manifest_errors: tuple[str, ...]
    oom: bool
    environment_fingerprint: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UnscoredBenchmarkRun:
        allowed = _RESULT_FIELDS - {
            "blinded_evidence_schema",
            "blinded_evidence_sha256",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                "Unknown unscored benchmark run fields: " + ", ".join(unknown),
            )
        if "scores" in payload and payload["scores"] is not None:
            raise ValueError("unscored benchmark run scores must be absent or null")
        required = allowed - {"scores", "peak_vram_mb"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                "Missing unscored benchmark run fields: " + ", ".join(missing),
            )
        peak_vram = payload.get("peak_vram_mb")
        if peak_vram is not None:
            peak_vram = _nonnegative_integer(peak_vram, "peak_vram_mb")
        oom = payload["oom"]
        if not isinstance(oom, bool):
            raise ValueError("oom must be a boolean")
        return cls(
            candidate_label=_nonblank(
                payload["candidate_label"], "candidate_label",
            ),
            scene=_positive_integer(payload["scene"], "scene"),
            workflow_profile=_nonblank(
                payload["workflow_profile"], "workflow_profile",
            ),
            prepared_workflow_sha256=_sha256(
                payload["prepared_workflow_sha256"],
                "prepared_workflow_sha256",
            ),
            contract_fingerprint=_sha256(
                payload["contract_fingerprint"], "contract_fingerprint",
            ),
            wall_time_seconds=_positive_number(
                payload["wall_time_seconds"], "wall_time_seconds",
            ),
            peak_vram_mb=peak_vram,
            preflight_errors=_errors(
                payload["preflight_errors"], "preflight_errors",
            ),
            manifest_errors=_errors(
                payload["manifest_errors"], "manifest_errors",
            ),
            oom=oom,
            environment_fingerprint=_sha256(
                payload["environment_fingerprint"], "environment_fingerprint",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_label": self.candidate_label,
            "scene": self.scene,
            "workflow_profile": self.workflow_profile,
            "prepared_workflow_sha256": self.prepared_workflow_sha256,
            "contract_fingerprint": self.contract_fingerprint,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "preflight_errors": list(self.preflight_errors),
            "manifest_errors": list(self.manifest_errors),
            "oom": self.oom,
            "environment_fingerprint": self.environment_fingerprint,
        }


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    failures: tuple[str, ...]
    baseline_medians: Mapping[str, float]
    candidate_medians: Mapping[str, float]
    wall_time_reduction: float


@dataclass(frozen=True)
class MatrixValidation:
    valid: bool
    failures: tuple[str, ...]


def blind_candidate_labels(
    candidate_names: Sequence[str], *, seed: int,
) -> dict[str, str]:
    names = tuple(_nonblank(name, "candidate name") for name in candidate_names)
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Blind-label seed must be an integer")
    labels = [f"candidate-{_alphabetic_label(index)}" for index in range(len(names))]
    random.Random(seed).shuffle(labels)
    return dict(zip(names, labels, strict=True))


def evaluate_promotion(
    baseline: Sequence[VisualConsistencyBenchmarkResult],
    candidate: Sequence[VisualConsistencyBenchmarkResult],
    *,
    review: Mapping[str, Any] | None = None,
    sealed_mapping: Mapping[str, Any] | None = None,
    environment: BenchmarkEnvironment | None = None,
) -> PromotionDecision:
    baseline_results = tuple(baseline)
    candidate_results = tuple(candidate)
    validation = validate_complete_matrix(
        (*baseline_results, *candidate_results),
        review=review,
        sealed_mapping=sealed_mapping,
        environment=environment,
    )
    role_failures = _comparison_matrix_failures(
        baseline_results, candidate_results,
    )
    if role_failures:
        validation = MatrixValidation(
            False, (*validation.failures, *role_failures),
        )
    if not validation.valid:
        return PromotionDecision(
            promote=False,
            failures=validation.failures,
            baseline_medians=MappingProxyType({}),
            candidate_medians=MappingProxyType({}),
            wall_time_reduction=0.0,
        )

    baseline_medians = _score_medians(baseline_results)
    candidate_medians = _score_medians(candidate_results)
    failures: list[str] = []
    for matrix_name, results in (
        ("baseline", baseline_results),
        ("candidate", candidate_results),
    ):
        if any(item.preflight_errors for item in results):
            failures.append(f"{matrix_name} has preflight errors")
        if any(item.manifest_errors for item in results):
            failures.append(f"{matrix_name} has manifest errors")

    baseline_by_scene = {item.scene: item for item in baseline_results}
    if any(
        item.oom and not baseline_by_scene[item.scene].oom
        for item in candidate_results
    ):
        failures.append("candidate has a new supported-profile OOM")

    for score_name in ("identity", "wardrobe", "location"):
        if candidate_medians[score_name] < baseline_medians[score_name]:
            failures.append(f"median {score_name} regressed")

    quality_improved = (
        candidate_medians["identity"] - baseline_medians["identity"] >= 1
        or candidate_medians["transition"] - baseline_medians["transition"] >= 1
    )
    equal_quality = candidate_medians == baseline_medians
    baseline_wall_time = statistics.median(
        item.wall_time_seconds for item in baseline_results
    )
    candidate_wall_time = statistics.median(
        item.wall_time_seconds for item in candidate_results
    )
    wall_time_reduction = (baseline_wall_time - candidate_wall_time) / baseline_wall_time
    if not quality_improved and not (
        equal_quality and wall_time_reduction >= 0.15
    ):
        failures.append(
            "candidate lacks required quality improvement or 15% equal-quality "
            "wall-time reduction",
        )

    return PromotionDecision(
        promote=not failures,
        failures=tuple(failures),
        baseline_medians=MappingProxyType(baseline_medians),
        candidate_medians=MappingProxyType(candidate_medians),
        wall_time_reduction=wall_time_reduction,
    )


def validate_complete_matrix(
    results: Sequence[VisualConsistencyBenchmarkResult],
    *,
    review: Mapping[str, Any] | None = None,
    sealed_mapping: Mapping[str, Any] | None = None,
    environment: BenchmarkEnvironment | None = None,
) -> MatrixValidation:
    recorded = tuple(results)
    failures = list(_complete_matrix_structure_failures(recorded))
    if review is None or sealed_mapping is None or environment is None:
        failures.append(
            "blinded evidence artifacts are required for complete matrix validation",
        )
    else:
        try:
            ingested = ingest_blinded_review_scores(
                recorded, review, sealed_mapping, environment=environment,
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"invalid blinded evidence artifacts: {exc}")
        else:
            evidence_sha256 = _blinded_evidence_sha256(
                recorded, review, sealed_mapping, environment,
            )
            failures.extend(
                _embedded_evidence_failures(
                    recorded, expected_sha256=evidence_sha256,
                ),
            )
            if any(
                dict(actual.scores) != dict(expected.scores)
                for actual, expected in zip(recorded, ingested, strict=True)
            ):
                failures.append(
                    "recorded scores do not match the blinded review artifact",
                )
    return MatrixValidation(
        valid=not failures,
        failures=tuple(dict.fromkeys(failures)),
    )


def _complete_matrix_structure_failures(
    recorded: tuple[VisualConsistencyBenchmarkResult, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    by_profile = {
        profile: tuple(
            item for item in recorded if item.workflow_profile == profile
        )
        for profile in FIXED_WORKFLOW_PROFILES
    }
    unexpected_profiles = sorted(
        {item.workflow_profile for item in recorded}
        - set(FIXED_WORKFLOW_PROFILES),
    )
    if unexpected_profiles:
        failures.append(
            "matrix has profiles outside the fixed fixture: "
            + ", ".join(unexpected_profiles),
        )
    failures.extend(
        _comparison_matrix_failures(
            by_profile[FIXED_WORKFLOW_PROFILES[0]],
            by_profile[FIXED_WORKFLOW_PROFILES[1]],
        ),
    )
    if len(recorded) != len(FIXED_SCENE_IDS) * 2:
        failures.append("complete matrix must contain exactly 12 records")
    return tuple(dict.fromkeys(failures))


def create_blinded_review_artifact(
    results: Sequence[VisualConsistencyBenchmarkResult],
    *,
    environment: BenchmarkEnvironment,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded = tuple(results)
    failures = _complete_matrix_structure_failures(recorded)
    if failures:
        raise ValueError(
            "Cannot blind incomplete benchmark matrix: "
            + "; ".join(failures),
        )
    _validate_environment_fingerprints(recorded, environment)
    labels = blind_candidate_labels(FIXED_WORKFLOW_PROFILES, seed=seed)
    entries = [
        {
            "candidate_label": labels[item.workflow_profile],
            "scene": item.scene,
            "scores": dict.fromkeys(SCORE_NAMES),
        }
        for item in recorded
    ]
    random.Random(seed ^ 0x47A7).shuffle(entries)
    review = {
        "schema": REVIEW_SCHEMA,
        "scoring_dimensions": list(SCORE_NAMES),
        "entries": entries,
    }
    sealed_mapping = {
        "schema": REVIEW_MAP_SCHEMA,
        "seed": seed,
        "mappings": [
            {
                "candidate_label": labels[profile],
                "workflow_profile": profile,
                "source_candidate_label": _single_label(
                    tuple(
                        item
                        for item in recorded
                        if item.workflow_profile == profile
                    ),
                    profile,
                ),
            }
            for profile in FIXED_WORKFLOW_PROFILES
        ],
    }
    return review, sealed_mapping


def ingest_blinded_review_scores(
    source_results: Sequence[VisualConsistencyBenchmarkResult],
    review: Mapping[str, Any],
    sealed_mapping: Mapping[str, Any],
    *,
    environment: BenchmarkEnvironment,
) -> tuple[VisualConsistencyBenchmarkResult, ...]:
    source = tuple(source_results)
    failures = _complete_matrix_structure_failures(source)
    if failures:
        raise ValueError(
            "Cannot ingest scores for incomplete benchmark matrix: "
            + "; ".join(failures),
        )
    _validate_environment_fingerprints(source, environment)
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("Unsupported blinded review schema")
    if sealed_mapping.get("schema") != REVIEW_MAP_SCHEMA:
        raise ValueError("Unsupported sealed review mapping schema")
    mappings = sealed_mapping.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError("Sealed review mapping requires a mappings list")
    identities: dict[str, tuple[str, str]] = {}
    for item in mappings:
        if not isinstance(item, Mapping):
            raise ValueError("Sealed review mappings must be objects")
        opaque = _nonblank(item.get("candidate_label"), "candidate_label")
        identity = (
            _nonblank(item.get("workflow_profile"), "workflow_profile"),
            _nonblank(
                item.get("source_candidate_label"), "source_candidate_label",
            ),
        )
        if opaque in identities:
            raise ValueError("Sealed review mapping has duplicate labels")
        identities[opaque] = identity
    entries = review.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Blinded review requires an entries list")
    reviewed_scores: dict[tuple[str, int], dict[str, int]] = {}
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != {
            "candidate_label",
            "scene",
            "scores",
        }:
            raise ValueError("Blinded review entries have invalid fields")
        opaque = _nonblank(item["candidate_label"], "candidate_label")
        if opaque not in identities:
            raise ValueError("Blinded review label is absent from sealed mapping")
        profile, source_label = identities[opaque]
        scene = _positive_integer(item["scene"], "scene")
        key = (profile, scene)
        if key in reviewed_scores:
            raise ValueError("Blinded review has duplicate profile/scene entries")
        reviewed_scores[key] = _scores(item["scores"])
        if not any(
            result.workflow_profile == profile
            and result.candidate_label == source_label
            for result in source
        ):
            raise ValueError("Sealed identity does not match source benchmark")
    if len(reviewed_scores) != len(source):
        raise ValueError("Blinded review does not contain the complete matrix")
    evidence_sha256 = _blinded_evidence_sha256(
        source, review, sealed_mapping, environment,
    )
    restored = []
    for result in source:
        payload = result.to_dict()
        try:
            payload["scores"] = reviewed_scores[
                (result.workflow_profile, result.scene)
            ]
        except KeyError:
            raise ValueError(
                "Blinded review does not match source profile/scene matrix",
            ) from None
        payload["blinded_evidence_schema"] = EVIDENCE_SCHEMA
        payload["blinded_evidence_sha256"] = evidence_sha256
        restored.append(VisualConsistencyBenchmarkResult.from_dict(payload))
    return tuple(restored)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate recorded cross-scene visual consistency benchmark results.",
    )
    parser.add_argument("results", help="Path to a recorded benchmark result JSON file.")
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Validate individual records without claiming complete matrix evidence.",
    )
    parser.add_argument(
        "--create-review",
        metavar="PATH",
        help="Write a sanitized, randomized review JSON file.",
    )
    parser.add_argument(
        "--sealed-mapping",
        metavar="PATH",
        help="Write the review identity mapping separately.",
    )
    parser.add_argument(
        "--review",
        metavar="PATH",
        help="Completed blinded review artifact for complete matrix validation.",
    )
    parser.add_argument("--seed", type=int, help="Randomization seed for review creation.")
    parser.add_argument(
        "--environment",
        metavar="PATH",
        help="Validated fixed-fixture environment manifest.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        source_path = Path(args.results).resolve(strict=True)
        if args.create_review:
            if args.review is not None:
                raise ValueError("--review cannot be used with --create-review")
            if args.record_only:
                raise ValueError("--record-only cannot be used with --create-review")
            if (
                args.sealed_mapping is None
                or args.seed is None
                or args.environment is None
            ):
                raise ValueError(
                    "--create-review requires --sealed-mapping, --seed, and --environment",
                )
            review_path, mapping_path = _distinct_paths(
                source_path,
                args.create_review,
                args.sealed_mapping,
                args.environment,
            )
        elif args.record_only:
            if (
                args.review is not None
                or args.sealed_mapping is not None
                or args.seed is not None
                or args.environment is not None
            ):
                raise ValueError(
                    "review, sealed mapping, and seed are not used in record-only mode",
                )
        else:
            if args.environment is None:
                raise ValueError(
                    "complete matrix validation requires --environment",
                )
            if (args.review is None) != (args.sealed_mapping is None):
                raise ValueError(
                    "--review and --sealed-mapping must be supplied together",
                )
            if args.seed is not None:
                raise ValueError("--seed is only used with --create-review")
            if args.review is None:
                review_path = mapping_path = None
            else:
                review_path, mapping_path = _distinct_paths(
                    source_path,
                    args.review,
                    args.sealed_mapping,
                    args.environment,
                )

        environment = (
            None
            if args.record_only
            else BenchmarkEnvironment.from_dict(
                json.loads(
                    Path(args.environment).read_text(encoding="utf-8-sig"),
                ),
            )
        )

        payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raise ValueError("Benchmark file must contain a results list")
        parser = (
            UnscoredBenchmarkRun.from_dict
            if args.create_review
            else VisualConsistencyBenchmarkResult.from_dict
        )
        results = tuple(parser(item) for item in raw_results)
        if not results:
            raise ValueError("Benchmark results list must not be empty")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        output = {"valid": False, "error": str(exc)}
        print(json.dumps(output, sort_keys=True) if args.json else output["error"])
        return 1

    if args.create_review:
        failures = _complete_matrix_structure_failures(results)
        if failures:
            output = {
                "mode": "review",
                "valid": False,
                "failures": list(failures),
            }
            print(
                json.dumps(output, sort_keys=True)
                if args.json
                else "; ".join(failures),
            )
            return 2
        review, sealed_mapping = create_blinded_review_artifact(
            results, environment=environment, seed=args.seed,
        )
        try:
            _write_json_pair_atomic(
                review_path,
                review,
                mapping_path,
                sealed_mapping,
            )
        except OSError as exc:
            output = {"valid": False, "error": str(exc)}
            print(
                json.dumps(output, sort_keys=True)
                if args.json
                else output["error"],
            )
            return 1
        output = {
            "mode": "review",
            "valid": True,
            "entry_count": len(review["entries"]),
            "review_path": str(review_path),
            "sealed_mapping_path": str(mapping_path),
        }
    elif args.record_only:
        output = {
            "mode": "records",
            "valid": True,
            "result_count": len(results),
        }
    else:
        try:
            review = (
                None
                if review_path is None
                else json.loads(review_path.read_text(encoding="utf-8-sig"))
            )
            sealed_mapping = (
                None
                if mapping_path is None
                else json.loads(mapping_path.read_text(encoding="utf-8-sig"))
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            output = {"valid": False, "error": str(exc)}
            print(
                json.dumps(output, sort_keys=True)
                if args.json
                else output["error"],
            )
            return 1
        validation = validate_complete_matrix(
            results,
            review=review,
            sealed_mapping=sealed_mapping,
            environment=environment,
        )
        output = {
            "mode": "matrix",
            "valid": validation.valid,
            "result_count": len(results),
            "failures": list(validation.failures),
        }
        if not validation.valid:
            print(
                json.dumps(output, sort_keys=True)
                if args.json
                else "; ".join(validation.failures),
            )
            return 2
    print(
        json.dumps(output, sort_keys=True)
        if args.json
        else _human_success(output),
    )
    return 0


def _scores(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(SCORE_NAMES):
        raise ValueError(
            "scores must contain exactly: " + ", ".join(SCORE_NAMES),
        )
    scores: dict[str, int] = {}
    for name in SCORE_NAMES:
        score = value[name]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"score {name} must be an integer")
        if not 1 <= score <= 5:
            raise ValueError(f"score {name} must be between 1 and 5")
        scores[name] = score
    return scores


def _errors(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a list of nonblank strings")
    return tuple(item.strip() for item in value)


def _matrix_failures(
    results: tuple[VisualConsistencyBenchmarkResult, ...], name: str,
) -> list[str]:
    failures: list[str] = []
    if not results:
        return [f"{name} benchmark matrix must not be empty"]
    scenes = [item.scene for item in results]
    if len(set(scenes)) != len(scenes):
        failures.append(f"{name} benchmark matrix has duplicate scenes")
    scene_ids = set(scenes)
    if scene_ids != FIXED_SCENE_IDS:
        failures.append(
            f"{name} benchmark matrix scenes must be exactly 1 through 6",
        )
    profiles = {item.workflow_profile for item in results}
    if len(profiles) != 1:
        failures.append(f"{name} benchmark matrix must use one workflow profile")
    labels = {item.candidate_label for item in results}
    if len(labels) != 1:
        failures.append(f"{name} benchmark matrix must use one candidate label")
    return failures


def _comparison_matrix_failures(
    baseline: tuple[VisualConsistencyBenchmarkResult, ...],
    candidate: tuple[VisualConsistencyBenchmarkResult, ...],
) -> tuple[str, ...]:
    failures = [
        *_matrix_failures(baseline, "baseline"),
        *_matrix_failures(candidate, "candidate"),
    ]
    if not failures:
        if {item.scene for item in baseline} != {item.scene for item in candidate}:
            failures.append("baseline and candidate scene matrices must match")
        baseline_profiles = {item.workflow_profile for item in baseline}
        candidate_profiles = {item.workflow_profile for item in candidate}
        if baseline_profiles != {FIXED_WORKFLOW_PROFILES[0]}:
            failures.append("baseline profile does not match the fixed fixture")
        if candidate_profiles != {FIXED_WORKFLOW_PROFILES[1]}:
            failures.append("candidate profile does not match the fixed fixture")
        if {item.candidate_label for item in baseline} == {
            item.candidate_label for item in candidate
        }:
            failures.append("baseline and candidate labels must be distinct")
    return tuple(failures)


def _embedded_evidence_failures(
    results: tuple[VisualConsistencyBenchmarkResult, ...],
    *,
    expected_sha256: str | None = None,
) -> tuple[str, ...]:
    if not results:
        return ()
    schemas = {item.blinded_evidence_schema for item in results}
    digests = {item.blinded_evidence_sha256 for item in results}
    failures: list[str] = []
    if schemas != {EVIDENCE_SCHEMA}:
        failures.append(
            "complete benchmark results require a valid blinded evidence marker",
        )
    if len(digests) != 1 or None in digests:
        failures.append(
            "complete benchmark results require one blinded evidence SHA-256",
        )
    elif expected_sha256 is not None and digests != {expected_sha256}:
        failures.append(
            "blinded evidence SHA-256 does not match review and sealed mapping",
        )
    return tuple(failures)


def _score_medians(
    results: tuple[VisualConsistencyBenchmarkResult, ...],
) -> dict[str, float]:
    return {
        name: float(statistics.median(item.scores[name] for item in results))
        for name in SCORE_NAMES
    }


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _hash_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty hash mapping")
    return {
        _nonblank(key, f"{name} key"): _sha256(item, f"{name}[{key}]")
        for key, item in value.items()
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _fixed_fixture_sha256() -> str:
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "visual_consistency"
        / "benchmark_plan.json"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _alphabetic_label(index: int) -> str:
    value = index
    label = ""
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("a") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _single_label(
    results: tuple[VisualConsistencyBenchmarkResult, ...], profile: str,
) -> str:
    labels = {item.candidate_label for item in results}
    if len(labels) != 1:
        raise ValueError(f"Profile {profile} does not have one candidate label")
    return next(iter(labels))


def _blinded_evidence_sha256(
    source_results: Sequence[Any],
    review: Mapping[str, Any],
    sealed_mapping: Mapping[str, Any],
    environment: BenchmarkEnvironment,
) -> str:
    source = []
    for item in source_results:
        payload = item.to_dict()
        payload.pop("scores", None)
        payload.pop("blinded_evidence_schema", None)
        payload.pop("blinded_evidence_sha256", None)
        source.append(payload)
    source.sort(key=lambda item: (item["workflow_profile"], item["scene"]))
    canonical = _canonical_json({
        "source": source,
        "review": review,
        "sealed_mapping": sealed_mapping,
        "environment": environment.to_dict(),
    })
    return hashlib.sha256(canonical).hexdigest()


def _validate_environment_fingerprints(
    results: Sequence[Any], environment: BenchmarkEnvironment,
) -> None:
    fingerprints = {item.environment_fingerprint for item in results}
    if fingerprints != {environment.fingerprint}:
        raise ValueError(
            "Benchmark records do not match supplied environment fingerprint",
        )


def _distinct_paths(
    source: str | Path,
    review: str | Path,
    sealed_mapping: str | Path,
    *additional: str | Path,
) -> tuple[Path, Path]:
    paths = (
        Path(source).resolve(),
        Path(review).resolve(),
        Path(sealed_mapping).resolve(),
        *(Path(path).resolve() for path in additional),
    )
    normalized = {os.path.normcase(str(path)) for path in paths}
    if len(normalized) != len(paths):
        raise ValueError(
            "source, review, sealed mapping, and environment paths must be distinct",
        )
    for index, first in enumerate(paths):
        if not first.exists():
            continue
        for second in paths[index + 1 :]:
            if second.exists() and os.path.samefile(first, second):
                raise ValueError(
                    "source, review, sealed mapping, and environment paths must not alias",
                )
    return paths[1], paths[2]


def _write_json_pair_atomic(
    first_path: Path,
    first_payload: Mapping[str, Any],
    second_path: Path,
    second_payload: Mapping[str, Any],
) -> None:
    destinations = (first_path, second_path)
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.parent.is_dir():
            raise OSError(f"Artifact parent is not a directory: {destination.parent}")
        if destination.exists() and destination.is_dir():
            raise OSError(f"Artifact target is a directory: {destination}")
        if not os.access(destination.parent, os.W_OK):
            raise OSError(f"Artifact parent is not writable: {destination.parent}")

    temporary_paths: list[Path] = []
    backup_paths: dict[Path, Path] = {}
    installed: set[Path] = set()
    retained_backups: set[Path] = set()
    try:
        for destination, payload in (
            (first_path, first_payload),
            (second_path, second_payload),
        ):
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temporary_paths.append(Path(handle.name))
        for destination in destinations:
            if destination.exists():
                with NamedTemporaryFile(
                    dir=destination.parent,
                    delete=False,
                ) as handle:
                    backup = Path(handle.name)
                backup.unlink()
                os.replace(destination, backup)
                backup_paths[destination] = backup
        for temporary, destination in zip(
            tuple(temporary_paths), destinations, strict=True,
        ):
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)
            installed.add(destination)
    except OSError as install_error:
        recovery_errors: list[str] = []
        for destination in installed:
            try:
                destination.unlink(missing_ok=True)
            except OSError as unlink_error:
                recovery_errors.append(f"{destination}: {unlink_error}")
        for destination, backup in backup_paths.items():
            if backup.exists():
                try:
                    os.replace(backup, destination)
                except OSError as restore_error:
                    retained_backups.add(backup)
                    recovery_errors.append(
                        f"{destination}: {restore_error}",
                    )
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                recovery_errors.append(f"{temporary}: {cleanup_error}")
        for backup in backup_paths.values():
            if backup.exists():
                retained_backups.add(backup)
        details = (
            f"; rollback errors ({'; '.join(recovery_errors)})"
            if recovery_errors
            else ""
        )
        retained = (
            "; retained recovery paths: "
            + ", ".join(str(path) for path in retained_backups)
            if retained_backups
            else ""
        )
        raise OSError(f"{install_error}{details}{retained}") from install_error
    else:
        cleanup_errors: list[str] = []
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(f"{temporary}: {cleanup_error}")
        for backup in backup_paths.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(f"{backup}: {cleanup_error}")
        if cleanup_errors:
            raise OSError("Artifact cleanup failed: " + "; ".join(cleanup_errors))


def _human_success(output: Mapping[str, Any]) -> str:
    if output["mode"] == "review":
        return f"Wrote blinded review with {output['entry_count']} entries."
    return (
        f"Validated {output['result_count']} benchmark result(s) "
        f"in {output['mode']} mode."
    )


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
