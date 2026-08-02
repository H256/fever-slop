from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QmlLintRunnerTests(unittest.TestCase):
    def test_runner_validates_the_repository_qml_tree(self):
        result = subprocess.run(
            [sys.executable, "scripts/qml_lint.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"QML lint failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
