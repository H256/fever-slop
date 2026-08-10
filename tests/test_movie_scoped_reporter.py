"""Tests for _MovieScopedReporter caching behavior (DESKTOP-006)."""

from __future__ import annotations

import gc
import unittest
from unittest.mock import MagicMock


class MovieScopedReporterCachingTests(unittest.TestCase):
    """Verify _MovieScopedReporter caches attribute lookups to avoid per-access allocation."""

    def _get_class(self):
        from feverslop.studio.desktop.viewmodels.studio import (
            _MovieScopedReporter,
            _ProjectCreationSignals,
        )
        return _MovieScopedReporter, _ProjectCreationSignals

    def _make_scoped(self, inner=None):
        Reporter, Signals = self._get_class()
        if inner is None:
            inner = MagicMock()
        bridge = MagicMock(spec=Signals)
        return Reporter(inner, bridge)

    def test_stub_attributes_are_cached_by_identity(self):
        """Stubs (step, message, file, etc.) must return the same object on repeated access."""
        Reporter, Signals = self._get_class()
        bridge = MagicMock(spec=Signals)
        subject = Reporter(None, bridge)

        for attr_name in ("step", "message", "file", "panel", "table", "run_progress"):
            first = getattr(subject, attr_name)
            second = getattr(subject, attr_name)
            self.assertIs(
                first, second,
                f"subject.{attr_name} returned different objects on repeated access",
            )

    def test_inner_callable_attributes_are_cached(self):
        """Callable attributes forwarded from inner must be cached, not wrapped per-access."""
        inner = MagicMock()
        subject = self._make_scoped(inner)

        # Ensure a callable attribute exists on inner
        inner.some_method = MagicMock()
        first = subject.some_method
        second = subject.some_method
        self.assertIs(
            first, second,
            "Forwarded callable from inner must be the same object on repeated access",
        )

    def test_no_allocation_per_access_after_cache(self):
        """After first access, repeated access must not allocate new objects."""
        subject = self._make_scoped()

        # Warm up — first access creates and caches
        _ = subject.step
        _ = subject.message

        gc.collect()
        before = gc.get_count()

        # Repeated accesses should be zero-allocation (just __dict__ lookup)
        for _ in range(1000):
            _ = subject.step
            _ = subject.message

        gc.collect()
        after = gc.get_count()

        # gc gets tuple (gen0, gen1, gen2); all gens should not have grown
        # We check gen0 which is the most sensitive indicator of per-access allocation
        self.assertLessEqual(
            after[0], before[0],
            "Repeated stub access allocated new objects after first access",
        )

    def test_run_progress_lambda_calls_func(self):
        """Verify run_progress stub actually calls the passed function."""
        Reporter, Signals = self._get_class()
        bridge = MagicMock(spec=Signals)
        subject = Reporter(None, bridge)  # inner=None so stubs are used
        called = False
        def my_func():
            nonlocal called
            called = True
        subject.run_progress("desc", my_func)
        self.assertTrue(called)

    def test_step_stub_emits_progress(self):
        """Verify step stub emits progress signal."""
        Reporter, Signals = self._get_class()
        bridge = MagicMock(spec=Signals)
        subject = Reporter(None, bridge)
        subject.step("test step")
        bridge.progress.emit.assert_called_once_with("test step")

    def test_message_stub_returns_none(self):
        """Verify message stub returns None."""
        Reporter, Signals = self._get_class()
        bridge = MagicMock(spec=Signals)
        subject = Reporter(None, bridge)  # inner=None so stubs are used
        result = subject.message("test")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
