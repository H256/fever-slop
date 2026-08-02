from pathlib import Path
import unittest


class DependencyAuditHookTests(unittest.TestCase):
    def test_pre_commit_config_runs_dependency_audit_without_filenames(self):
        config_path = Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml"
        self.assertTrue(config_path.is_file())
        config = config_path.read_text(encoding="utf-8")

        self.assertIn("id: dependency-audit", config)
        self.assertIn("entry: uvx pip-audit .", config)
        self.assertIn("pass_filenames: false", config)


if __name__ == "__main__":
    unittest.main()
