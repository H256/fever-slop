import tomllib
import unittest
from pathlib import Path


class UndeclaredDependencyTests(unittest.TestCase):
    def test_requests_and_httpx_are_declared_direct_dependencies(self):
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        dependencies = {dep.split(">=")[0] for dep in project["dependencies"]}

        self.assertIn("requests", dependencies)
        self.assertIn("httpx", dependencies)

    def test_modules_using_requests_and_httpx_import_cleanly(self):
        import feverslop.adapters.comfyui_client  # noqa: F401
        import feverslop.adapters.gemma4_startframe_validator  # noqa: F401
        import feverslop.adapters.llm_client  # noqa: F401
        import feverslop.application.orbitsheets_logic  # noqa: F401


if __name__ == "__main__":
    unittest.main()
