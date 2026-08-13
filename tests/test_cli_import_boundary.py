"""Guard the CLI/application boundary from the Studio package."""

import ast
import unittest
from pathlib import Path


class CliImportBoundaryTests(unittest.TestCase):
    def test_cli_runtime_modules_do_not_import_studio(self) -> None:
        roots = [
            Path("src/feverslop/cli"),
            Path("src/feverslop/composition"),
            Path("src/feverslop/adapters"),
        ]
        offenders: list[str] = []
        for root in roots:
            for path in root.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    else:
                        continue
                    if any(module == "feverslop.studio" or module.startswith("feverslop.studio.") for module in modules):
                        offenders.append(f"{path}:{node.lineno}")
        self.assertEqual([], offenders)

    def test_movie_cli_configuration_is_available_without_studio_import(self) -> None:
        import sys

        sys.modules.pop("feverslop.studio", None)
        from feverslop.composition import movie_pipeline_jobs

        self.assertEqual("msr", movie_pipeline_jobs.movie_runtime_config()["movie_video_workflow"])
        self.assertNotIn("feverslop.studio", sys.modules)


if __name__ == "__main__":
    unittest.main()
