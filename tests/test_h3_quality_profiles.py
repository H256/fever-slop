from __future__ import annotations

import json
import unittest
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "h3_quality_profiles.json"


class H3QualityProfileSnapshotTests(unittest.TestCase):
    def test_snapshot_covers_three_two_pass_quality_profiles(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual("two_pass", payload["pass_strategy"])
        self.assertEqual({"draft", "standard", "final"}, set(payload["profiles"]))
        for profile in payload["profiles"].values():
            self.assertEqual(1.0, profile["pass1"]["denoise"])
            self.assertEqual(2.0, profile["pass2"]["resolution_scale"])
            self.assertGreater(profile["pass2"]["max_megapixels"], profile["pass1"]["resolution"]["megapixels"])
            self.assertTrue(profile["resource_class"])

    def test_quality_budget_increases_without_a_third_pass(self):
        profiles = json.loads(FIXTURE.read_text(encoding="utf-8"))["profiles"]
        budgets = [(item["pass1"]["steps"], item["pass2"]["steps"]) for item in profiles.values()]
        self.assertEqual(sorted(budgets), budgets)
        self.assertNotIn("pass3", json.dumps(profiles))

    def test_sampling_snapshot_matches_the_domain_defaults(self):
        from feverslop.domain.h3_two_pass import default_h3_two_pass_spec

        profiles = json.loads(FIXTURE.read_text(encoding="utf-8"))["profiles"]
        for quality, profile in profiles.items():
            spec = default_h3_two_pass_spec(quality)
            self.assertEqual(spec.pass1_steps, profile["pass1"]["steps"])
            self.assertEqual(spec.pass2_steps, profile["pass2"]["steps"])
            self.assertEqual(spec.pass2_denoise, profile["pass2"]["denoise"])


if __name__ == "__main__":
    unittest.main()
