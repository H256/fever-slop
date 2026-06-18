import unittest
from pathlib import Path


class ImportBoundaryTests(unittest.TestCase):
    def test_package_code_does_not_import_root_architecture_packages(self):
        package_root = Path("src/autoprompter")
        forbidden = [
            "from application.",
            "from adapters.",
            "from domain.",
            "from ports.",
            "import application",
            "import adapters",
            "import domain",
            "import ports",
        ]

        offenders = []
        for path in package_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)

    def test_compatibility_docs_define_new_import_policy(self):
        text = Path("docs/architecture_compatibility.md").read_text(encoding="utf-8")

        self.assertIn("new implementation imports must use `autoprompter.*`", text)
        self.assertIn("no new code should import `application.*`, `adapters.*`, `domain.*`, or `ports.*`", text)


if __name__ == "__main__":
    unittest.main()
