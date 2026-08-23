import importlib.util
import unittest


class StudioPackageRemovalTests(unittest.TestCase):
    def test_deprecated_studio_package_is_not_importable(self):
        self.assertIsNone(importlib.util.find_spec("feverslop.studio"))

