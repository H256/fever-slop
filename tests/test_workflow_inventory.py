import re
import unittest
from pathlib import Path


class WorkflowInventoryTests(unittest.TestCase):
    def test_path_map_covers_every_tracked_workflow_json(self):
        root = Path(__file__).resolve().parents[1]
        documented = {
            match.group(1)
            for match in re.finditer(r"\| `([^`]+\.json)` \|", (root / "documentation" / "workflow-path-map.md").read_text(encoding="utf-8"))
        }
        tracked = {
            path.relative_to(root).as_posix()
            for path in (root / "workflows").rglob("*.json")
        }

        self.assertEqual(tracked, documented)


if __name__ == "__main__":
    unittest.main()
