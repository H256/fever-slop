"""Focused tests for the per-family boundary spec registry (P2)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from feverslop.domain.workflow_import_families import (
    FAMILY_SPECS,
    NON_WORKFLOW_FAMILIES,
    spec_for,
)
from feverslop.domain.workflow_import_inspector import inspect_workflow

_WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


class FamilySpecRegistryTests(unittest.TestCase):
    def test_all_video_families_have_specs(self):
        self.assertEqual(
            {"minimax_h3", "ltx_ingredients", "ltx_msr", "ltx_i2v"},
            set(FAMILY_SPECS),
        )

    def test_krea_is_a_non_workflow_family(self):
        self.assertIn("krea", NON_WORKFLOW_FAMILIES)
        self.assertIsNone(spec_for("krea"))

    def test_spec_for_unknown_is_none(self):
        self.assertIsNone(spec_for("does-not-exist"))

    def test_spec_for_returns_the_family_spec(self):
        spec = spec_for("minimax_h3")
        self.assertIsNotNone(spec)
        self.assertEqual("minimax_h3", spec.pipeline)


class GenericInspectorAcrossFamiliesTests(unittest.TestCase):
    """The single inspector must handle every family via its spec (no re-impl)."""

    def test_ltx_i2v_workflow_inspects_cleanly(self):
        graph = json.loads(
            (_WORKFLOWS / "video_ltxv_i2v_v2.json").read_text(encoding="utf-8")
        )
        analysis = inspect_workflow(graph, spec=spec_for("ltx_i2v"))
        self.assertEqual("auto_compatible", analysis.compatibility)
        self.assertIsNotNone(analysis.mapping)

    def test_ltx_msr_workflow_inspects_cleanly(self):
        graph = json.loads(
            (_WORKFLOWS / "video_ltxv_msr_1actor_1background_v4.json").read_text(
                encoding="utf-8"
            )
        )
        analysis = inspect_workflow(graph, spec=spec_for("ltx_msr"))
        self.assertEqual("auto_compatible", analysis.compatibility)
        self.assertIsNotNone(analysis.mapping)

    def test_ltx_ingredients_workflow_inspects_deterministically(self):
        # The 2-stage ingredients template has two samplers; the first stage's
        # sampler is not a direct ancestor of the final output, so the
        # inspector conservatively flags it for confirmation. Either
        # auto_compatible or needs_confirmation is a valid, deterministic
        # outcome (never unsupported/crash) — the point is the generic
        # inspector handles the family without re-implementation.
        graph = json.loads(
            (_WORKFLOWS / "video_ltxv_ingredients_2stage_v6.json").read_text(
                encoding="utf-8"
            )
        )
        analysis = inspect_workflow(graph, spec=spec_for("ltx_ingredients"))
        self.assertIn(
            analysis.compatibility, ("auto_compatible", "needs_confirmation")
        )
        # Determinism: inspecting twice yields the same classification.
        again = inspect_workflow(graph, spec=spec_for("ltx_ingredients"))
        self.assertEqual(analysis.compatibility, again.compatibility)


if __name__ == "__main__":
    unittest.main()
