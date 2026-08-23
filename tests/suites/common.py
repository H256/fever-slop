"""Shared helpers for the explicitly separated test suites."""

from __future__ import annotations

import unittest


def load_modules(modules: tuple[str, ...]) -> unittest.TestSuite:
    """Load exactly the test modules assigned to one suite."""

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module in modules:
        suite.addTests(loader.loadTestsFromName(f"tests.test_{module}"))
    return suite
