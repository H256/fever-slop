"""Explicit integration-test suite.

These tests may use local binaries such as FFmpeg and exercise multiple
application layers. They must not be included in the unit suite.
"""

from __future__ import annotations

import unittest

from tests.suites.common import load_modules

INTEGRATION_MODULES = (
    "full_auto_pipeline_runner",
    "dspy_runtime",
    "global_library_adapter",
    "minimax_h3_integration",
    "movie_planning_refine_actors",
    "movie_project",
    "timeline_integration",
)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(load_modules(INTEGRATION_MODULES))
    raise SystemExit(0 if result.wasSuccessful() else 1)
