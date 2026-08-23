"""Guard the CLI/application boundary from the Studio package."""

import ast
import unittest
from pathlib import Path


class CliImportBoundaryTests(unittest.TestCase):
    def test_headless_studio_services_use_canonical_implementations(self) -> None:
        forbidden = {
            "feverslop.studio.artifact_catalog",
            "feverslop.studio.artifact_locking",
            "feverslop.studio.movie_pipeline_jobs",
        }
        offenders: list[str] = []
        for path in (
            Path("src/feverslop/studio/projects.py"),
            Path("src/feverslop/studio/job_service.py"),
        ):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules = [alias.name for alias in node.names]
                    modules = imported_modules
                elif isinstance(node, ast.ImportFrom):
                    imported_modules = [
                        f"{node.module}.{alias.name}" if node.module else alias.name
                        for alias in node.names
                    ]
                    modules = [node.module or "", *imported_modules]
                else:
                    continue
                for module in modules:
                    if module in forbidden or any(module.startswith(f"{prefix}.") for prefix in forbidden):
                        offenders.append(f"{path}:{node.lineno}: {module}")

        self.assertEqual([], offenders)

    def test_legacy_studio_modules_reexport_canonical_objects(self) -> None:
        from feverslop.adapters.artifact_catalog import ArtifactCatalog
        from feverslop.adapters.artifact_locking import artifact_write_lock
        from feverslop.composition.movie_pipeline_jobs import movie_runtime_config
        from feverslop.studio.artifact_catalog import ArtifactCatalog as LegacyArtifactCatalog
        from feverslop.studio.artifact_locking import artifact_write_lock as legacy_artifact_write_lock
        from feverslop.studio.movie_pipeline_jobs import movie_runtime_config as legacy_movie_runtime_config

        self.assertIs(ArtifactCatalog, LegacyArtifactCatalog)
        self.assertIs(artifact_write_lock, legacy_artifact_write_lock)
        self.assertIs(movie_runtime_config, legacy_movie_runtime_config)

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
