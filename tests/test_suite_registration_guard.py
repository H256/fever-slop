"""Guard: every tracked ``test_*.py`` must be assigned to a test suite.

The suites (``tests/suites/{unit,integration,e2e}.py``) use explicit module
lists via ``load_modules`` rather than implicit ``unittest discover``. A new
test file that is not added to one of those tuples is silently never run in
CI. This test fails when that happens so the omission is caught immediately
instead of hiding for weeks.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_FILES = ("tests/suites/unit.py", "tests/suites/integration.py", "tests/suites/e2e.py")


def _registered_modules() -> set[str]:
    registered: set[str] = set()
    for relative in SUITE_FILES:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r'"([a-z_0-9]+)"', text):
            registered.add(match.group(1))
    return registered


def _tracked_test_modules() -> set[str]:
    return {
        path.name[5:-3]
        for path in (REPO_ROOT / "tests").glob("test_*.py")
        if path.is_file()
    }


class TestModuleRegistrationGuard(unittest.TestCase):
    def test_every_tracked_test_module_is_assigned_to_a_suite(self):
        registered = _registered_modules()
        tracked = _tracked_test_modules()
        unregistered = sorted(tracked - registered)
        self.assertEqual(
            unregistered,
            [],
            "These test modules are not assigned to any suite "
            f"(unit/integration/e2e) and would never run in CI: {unregistered}",
        )


if __name__ == "__main__":
    unittest.main()
