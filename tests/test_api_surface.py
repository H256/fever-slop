"""Tests for the feverslop public API surface (API-001, API-002, API-003)."""

import subprocess
import sys
import unittest
import warnings


class TestVersion(unittest.TestCase):
    """API-001: __version__ attribute exists and is correct."""

    def test_version_attribute_exists(self):
        import feverslop
        self.assertTrue(hasattr(feverslop, "__version__"))
        self.assertEqual(feverslop.__version__, "0.4.1")

    def test_version_is_string(self):
        import feverslop
        self.assertIsInstance(feverslop.__version__, str)


class TestLazyLoading(unittest.TestCase):
    """API-002: submodules are not eagerly loaded."""

    def test_submodules_not_loaded_on_top_level_import(self):
        """Must run in subprocess to avoid test-runner module caching."""
        code = """
import sys
import feverslop
loaded = [m for m in sys.modules if m.startswith("feverslop.") and not m.startswith("feverslop.utils.deprecation")]
assert loaded == [], f"Unexpected submodules loaded: {loaded}"
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, f"stderr: {result.stderr}")

    def test_all_symbols_accessible(self):
        import feverslop
        for name in feverslop.__all__:
            with self.subTest(symbol=name):
                val = getattr(feverslop, name)
                self.assertIsNotNone(val)

    def test_dir_includes_all_symbols(self):
        import feverslop
        dir_result = dir(feverslop)
        for name in feverslop.__all__:
            self.assertIn(name, dir_result)
        self.assertIn("__version__", dir_result)


class TestDeprecationUtility(unittest.TestCase):
    """API-003: deprecation decorator works correctly."""

    def test_deprecated_function_warns(self):
        from feverslop.utils.deprecation import deprecated

        @deprecated("old way", since="0.1.0", alternative="new_func")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(42, old_func())
            self.assertEqual(1, len(w))
            self.assertIs(DeprecationWarning, w[0].category)
            self.assertIn("old_func", str(w[0].message))

    def test_deprecated_class_warns_on_init(self):
        from feverslop.utils.deprecation import deprecated

        @deprecated("old class", since="0.1.0")
        class OldClass:
            def __init__(self, x=0):
                self.x = x

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass(x=7)
            self.assertEqual(1, len(w))
            self.assertIs(DeprecationWarning, w[0].category)
            self.assertEqual(7, obj.x)


class TestBackwardCompatibility(unittest.TestCase):
    """Verify that from feverslop import X still works."""

    def test_from_import_syntax(self):
        code = """
from feverslop import AutoProduceMovieUseCase, MovieProject, Reporter, slugify_project_name
from feverslop import __version__
assert __version__ == "0.4.1"
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, f"stderr: {result.stderr}")

    def test_attr_access_syntax(self):
        code = """
import feverslop
assert feverslop.__version__ == "0.4.1"
assert feverslop.AutoProduceMovieUseCase is not None
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, f"stderr: {result.stderr}")

    def test_nonexistent_attribute(self):
        import feverslop
        with self.assertRaises(AttributeError):
            _ = feverslop.this_symbol_does_not_exist


if __name__ == "__main__":
    unittest.main()
