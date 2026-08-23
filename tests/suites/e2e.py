"""Explicit end-to-end tests, intended for manually dispatched CI runs."""

from __future__ import annotations

import unittest

from tests.suites.common import load_modules

E2E_MODULES = (
    "full_auto_cli",
    "generate_pipeline_smoke",
    "generate_render_plan_e2e_fake_ports",
    "render_ltx_cli",
    "run_pipeline",
    "runner_scripts",
    "scaffold_movie_cli",
    "visual_consistency_cli",
)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(load_modules(E2E_MODULES))
    raise SystemExit(0 if result.wasSuccessful() else 1)
