import unittest

from feverslop.errors import (
    FeverSlopAdaptationError,
    FeverSlopConfigError,
    FeverSlopDataError,
    FeverSlopError,
    FeverSlopLMLError,
    FeverSlopRenderError,
    FeverSlopValidationError,
    FeverSlopWorkflowError,
)


class FeverSlopErrorHierarchyTests(unittest.TestCase):
    def test_all_typed_exceptions_inherit_from_base(self):
        subclasses = [
            FeverSlopLMLError,
            FeverSlopRenderError,
            FeverSlopConfigError,
            FeverSlopWorkflowError,
            FeverSlopValidationError,
            FeverSlopAdaptationError,
            FeverSlopDataError,
        ]
        for cls in subclasses:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, FeverSlopError))

    def test_workflow_error_inherits_render_error(self):
        self.assertTrue(issubclass(FeverSlopWorkflowError, FeverSlopRenderError))

    def test_base_catches_all_subclasses(self):
        for cls in [
            FeverSlopLMLError,
            FeverSlopRenderError,
            FeverSlopConfigError,
            FeverSlopWorkflowError,
            FeverSlopValidationError,
            FeverSlopAdaptationError,
            FeverSlopDataError,
        ]:
            with self.subTest(cls=cls.__name__):
                try:
                    raise cls("test")
                except FeverSlopError:
                    pass
                else:
                    self.fail(f"{cls.__name__} was not caught by FeverSlopError")

    def test_base_does_not_catch_bare_value_error(self):
        try:
            raise ValueError("bare")
        except FeverSlopError:
            self.fail("FeverSlopError caught bare ValueError")
        except ValueError:
            pass

    def test_base_does_not_catch_bare_runtime_error(self):
        try:
            raise RuntimeError("bare")
        except FeverSlopError:
            self.fail("FeverSlopError caught bare RuntimeError")
        except RuntimeError:
            pass

    def test_validation_error_is_not_render_error(self):
        self.assertFalse(issubclass(FeverSlopValidationError, FeverSlopRenderError))

    def test_config_error_is_not_render_error(self):
        self.assertFalse(issubclass(FeverSlopConfigError, FeverSlopRenderError))

    def test_root_imports_are_same_objects_as_submodule(self):
        """Root-level imports resolve to same class objects as submodule."""
        import feverslop
        from feverslop.errors import (
            FeverSlopError as SubModuleError,
            FeverSlopAdaptationError as SubModuleAdaptationError,
            FeverSlopConfigError as SubModuleConfigError,
            FeverSlopDataError as SubModuleDataError,
            FeverSlopLMLError as SubModuleLMLError,
            FeverSlopRenderError as SubModuleRenderError,
            FeverSlopValidationError as SubModuleValidationError,
            FeverSlopWorkflowError as SubModuleWorkflowError,
        )
        self.assertIs(feverslop.FeverSlopError, SubModuleError)
        self.assertIs(feverslop.FeverSlopAdaptationError, SubModuleAdaptationError)
        self.assertIs(feverslop.FeverSlopConfigError, SubModuleConfigError)
        self.assertIs(feverslop.FeverSlopDataError, SubModuleDataError)
        self.assertIs(feverslop.FeverSlopLMLError, SubModuleLMLError)
        self.assertIs(feverslop.FeverSlopRenderError, SubModuleRenderError)
        self.assertIs(feverslop.FeverSlopValidationError, SubModuleValidationError)
        self.assertIs(feverslop.FeverSlopWorkflowError, SubModuleWorkflowError)

    def test_root_imports_preserve_hierarchy(self):
        """Root-level imports maintain correct inheritance chain."""
        import feverslop
        self.assertTrue(issubclass(feverslop.FeverSlopLMLError, feverslop.FeverSlopError))
        self.assertTrue(issubclass(feverslop.FeverSlopRenderError, feverslop.FeverSlopError))
        self.assertTrue(issubclass(feverslop.FeverSlopWorkflowError, feverslop.FeverSlopRenderError))
        self.assertTrue(issubclass(feverslop.FeverSlopWorkflowError, feverslop.FeverSlopError))


if __name__ == "__main__":
    unittest.main()

class InternalSubmoduleBlockingTests(unittest.TestCase):
    """Verify internal submodules cannot be imported from package root."""

    def test_internal_submodules_raise_import_error(self):
        """Internal submodules must raise ImportError, not leak via Python fallback."""
        import feverslop

        for name in ["config", "application", "path_utils", "domain", "ports", "errors"]:
            with self.subTest(name=name):
                with self.assertRaises(ImportError):
                    feverslop.__getattr__(name)

    def test_direct_submodule_imports_still_work(self):
        """Direct submodule imports must still work for internal code."""
        from feverslop.errors import FeverSlopError
        from feverslop.config.app_config import AppConfig

        self.assertTrue(issubclass(FeverSlopError, Exception))
        self.assertTrue(callable(AppConfig))
