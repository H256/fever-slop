"""Evaluate pinned reference-sheet benchmark results without storing media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "identity_consistency",
    "view_coverage",
    "sharpness",
    "layout_continuity",
    "reproducibility",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a pinned reference-sheet benchmark configuration."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark configuration must be a JSON object")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    for key in ("schema_version", "fixture_set", "candidates", "quality_gate"):
        if key not in config:
            raise ValueError(f"benchmark configuration is missing {key!r}")
    if not isinstance(config["candidates"], list) or not config["candidates"]:
        raise ValueError("benchmark configuration requires at least one candidate")
    gate = config["quality_gate"]
    if not isinstance(gate, dict):
        raise ValueError("quality_gate must be an object")
    missing = [metric for metric in REQUIRED_METRICS if metric not in gate]
    if missing:
        raise ValueError(f"quality_gate is missing metrics: {', '.join(missing)}")
    for candidate in config["candidates"]:
        if not isinstance(candidate, dict) or not candidate.get("name"):
            raise ValueError("each candidate requires a name")
        if not isinstance(candidate.get("runs"), list) or not candidate["runs"]:
            raise ValueError(f"candidate {candidate.get('name', '<unknown>')} has no runs")
        for run in candidate["runs"]:
            if not isinstance(run, dict) or not run.get("fixture_id"):
                raise ValueError("each run requires fixture_id")
            metrics = run.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("each run requires machine-readable metrics")
            missing = [metric for metric in REQUIRED_METRICS if metric not in metrics]
            if missing:
                raise ValueError(
                    f"run {run.get('fixture_id')} is missing metrics: {', '.join(missing)}"
                )


def _candidate_report(candidate: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    runs = candidate["runs"]
    averages = {
        metric: round(sum(float(run["metrics"][metric]) for run in runs) / len(runs), 6)
        for metric in REQUIRED_METRICS
    }
    checks = {metric: averages[metric] >= float(gate[metric]) for metric in REQUIRED_METRICS}
    failures = sum(int(run.get("failures", 0)) for run in runs)
    retries = sum(int(run.get("retries", 0)) for run in runs)
    passed = all(checks.values()) and failures == 0
    return {
        "name": candidate["name"],
        "backend": candidate.get("backend", "unknown"),
        "runs": len(runs),
        "metrics": averages,
        "quality_gate": checks,
        "failures": failures,
        "retries": retries,
        "passed": passed,
        "provenance": candidate.get("provenance", []),
    }


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    candidates = [_candidate_report(candidate, config["quality_gate"]) for candidate in config["candidates"]]
    passing = [candidate for candidate in candidates if candidate["passed"]]
    recommendation = passing[0]["name"] if passing else None
    return {
        "schema_version": config["schema_version"],
        "fixture_set": config["fixture_set"],
        "quality_gate": config["quality_gate"],
        "candidates": candidates,
        "recommendation": recommendation,
        "decision": "replace" if recommendation else "fallback",
        "limitations": config.get("limitations", []),
    }


def run(config_path: Path, report_path: Path) -> dict[str, Any]:
    report = evaluate(_load(config_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run(args.config, args.report)
    print(f"Reference-sheet benchmark: {report['decision']}")
    if report["recommendation"]:
        print(f"Recommended candidate: {report['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
