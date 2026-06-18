import unittest
from pathlib import Path


class RunnerScriptTests(unittest.TestCase):
    def test_test_ps1_forwards_project_config_to_ltx(self):
        script = Path("test.ps1").read_text(encoding="utf-8")

        self.assertIn('"--project-config", $projectConfigPath', script)
        self.assertNotIn('"--min-duration", "$sceneMinDuration"', script)
        self.assertNotIn('"--max-duration", "$sceneMaxDuration"', script)


if __name__ == "__main__":
    unittest.main()
