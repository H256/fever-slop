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

    def test_application_layer_does_not_import_concrete_adapters_or_root_modules(self):
        app_root = Path("src/autoprompter/application")
        forbidden = [
            "from autoprompter.adapters.",
            "import autoprompter.adapters.",
            "from app_config",
            "from project_config",
            "from beat_analysis",
            "from demucs_separator",
            "from vocal_timeline_analyzer",
            "from prompt_pipeline",
            "from concept_prompt_batcher",
            "from scene_prompt_builder",
            "from render_plan_builder",
            "from prompt_relay_builder",
            "from stage1_segment_builder",
            "from scene_duration_enforcer",
            "from comfyui_client",
        ]
        offenders = []
        for path in app_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
