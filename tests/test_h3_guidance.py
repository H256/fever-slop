"""Tests for the H3 block guidance decoder."""

from __future__ import annotations

import unittest

from feverslop.domain.h3_guidance import h3_block_guide


class H3BlockGuideTests(unittest.TestCase):
    def test_plan_missing_suggests_replan(self):
        result = h3_block_guide(["h3.fallback.plan_missing"])
        self.assertIn("planner did not produce", result.lower())
        self.assertIn("replan-failed", result)

    def test_truncation_suggested_budget_raise(self):
        result = h3_block_guide(
            ["h3.prompt.truncated"],
            truncation_suspected=True,
        )
        self.assertIn("truncated", result.lower())
        self.assertIn("prompt_planner_max_tokens", result)

    def test_exhausted_suggests_input_correction(self):
        result = h3_block_guide(["h3.recovery.exhausted"])
        self.assertIn("exhausted", result.lower())
        self.assertIn("inputs", result.lower())

    def test_lyrics_mismatch(self):
        result = h3_block_guide(["h3.performance.lyrics_mismatch"])
        self.assertIn("lyrics", result.lower())

    def test_unknown_code_gets_generic_fallback(self):
        result = h3_block_guide(["some.unknown.code"])
        self.assertIn("validation failed", result.lower())

    def test_empty_reason_codes(self):
        result = h3_block_guide([])
        self.assertIn("failed preparation", result.lower())


if __name__ == "__main__":
    unittest.main()
