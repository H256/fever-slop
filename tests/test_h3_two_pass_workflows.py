import json
import unittest
from pathlib import Path

from feverslop.domain.h3_two_pass import default_h3_two_pass_spec, validate_h3_two_pass_topology


class H3TwoPassWorkflowTests(unittest.TestCase):
    def test_generated_profiles_have_native_two_pass_topology(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "workflows" / "video" / "minimax_h3").glob("*_two_pass.json"))
        self.assertEqual(3, len(paths))
        for path in paths:
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text(encoding="utf-8"))
                validate_h3_two_pass_topology(workflow, default_h3_two_pass_spec("draft"))


if __name__ == "__main__":
    unittest.main()
