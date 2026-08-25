import json
import tempfile
import unittest
from pathlib import Path

from feverslop.tools.reference_sheet_benchmark import evaluate, run


def config(*, passing=True):
    value = 0.9 if passing else 0.4
    return {
        "schema_version": 1,
        "fixture_set": "reference-sheet-fixtures-2026-08-25",
        "quality_gate": {metric: 0.8 for metric in (
            "identity_consistency", "view_coverage", "sharpness",
            "layout_continuity", "reproducibility",
        )},
        "candidates": [{
            "name": "current",
            "backend": "image_views",
            "provenance": [{"config_sha256": "fixture"}],
            "runs": [{
                "fixture_id": "character-interior-01",
                "metrics": {metric: value for metric in (
                    "identity_consistency", "view_coverage", "sharpness",
                    "layout_continuity", "reproducibility",
                )},
                "runtime_seconds": 12.0,
                "failures": 0,
                "retries": 0,
            }],
        }],
        "limitations": ["No generated media is included in the report."],
    }


class ReferenceSheetBenchmarkTests(unittest.TestCase):
    def test_recommends_candidate_only_when_all_gates_pass(self):
        report = evaluate(config())
        self.assertEqual("replace", report["decision"])
        self.assertEqual("current", report["recommendation"])

        report = evaluate(config(passing=False))
        self.assertEqual("fallback", report["decision"])
        self.assertIsNone(report["recommendation"])

    def test_failures_force_fallback_even_when_scores_pass(self):
        value = config()
        value["candidates"][0]["runs"][0]["failures"] = 1
        report = evaluate(value)
        self.assertEqual("fallback", report["decision"])
        self.assertFalse(report["candidates"][0]["passed"])

    def test_run_writes_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "benchmark.json"
            report_path = root / "reports" / "report.json"
            config_path.write_text(json.dumps(config()), encoding="utf-8")
            run(config_path, report_path)
            saved = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual("reference-sheet-fixtures-2026-08-25", saved["fixture_set"])
        self.assertIn("provenance", saved["candidates"][0])

    def test_missing_metric_is_rejected(self):
        value = config()
        del value["candidates"][0]["runs"][0]["metrics"]["sharpness"]
        with self.assertRaisesRegex(ValueError, "sharpness"):
            evaluate(value)
