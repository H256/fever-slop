import unittest

from feverslop.config.project_validation import validate_pipeline_mode
from feverslop.studio.project_validation import validate_pipeline_mode as legacy_validate_pipeline_mode


class ProjectValidationConfigTests(unittest.TestCase):
    def test_validation_has_canonical_config_owner_and_legacy_alias(self):
        self.assertIs(legacy_validate_pipeline_mode, validate_pipeline_mode)
        self.assertEqual("msr", validate_pipeline_mode("msr"))
        with self.assertRaisesRegex(ValueError, "pipeline_mode"):
            validate_pipeline_mode("unknown")


if __name__ == "__main__":
    unittest.main()
