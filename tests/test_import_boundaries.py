import unittest
from pathlib import Path


class ImportBoundaryTests(unittest.TestCase):
    def test_package_code_does_not_import_root_architecture_packages(self):
        package_root = Path("src/feverslop")
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

    def test_package_code_does_not_import_root_cli_modules(self):
        package_root = Path("src/feverslop")
        forbidden = [
            "import run_pipeline",
            "from run_pipeline",
            "import full_auto",
            "from full_auto",
            "import main",
            "from main",
            "import render_ltx",
            "from render_ltx",
            "import render_storyboard",
            "from render_storyboard",
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

        self.assertIn("new implementation imports must use `feverslop.*`", text)
        self.assertIn("no new code should import `application.*`, `adapters.*`, `domain.*`, or `ports.*`", text)

    def test_application_layer_does_not_import_concrete_adapters_or_root_modules(self):
        app_root = Path("src/feverslop/application")
        forbidden = [
            "from feverslop.adapters.",
            "import feverslop.adapters.",
            "from feverslop.audio.",
            "import feverslop.audio.",
            "from feverslop.pipeline.",
            "import feverslop.pipeline.",
            "from feverslop.prompting.",
            "import feverslop.prompting.",
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

    def test_inner_layers_do_not_import_concrete_adapters(self):
        inner_layers = [
            Path("src/feverslop/domain"),
            Path("src/feverslop/ports"),
            Path("src/feverslop/application"),
            Path("src/feverslop/pipeline"),
            Path("src/feverslop/prompting"),
        ]
        forbidden = [
            "from feverslop.adapters.",
            "import feverslop.adapters.",
        ]
        offenders = []
        for layer_root in inner_layers:
            for path in layer_root.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for token in forbidden:
                    if token in text:
                        offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)

    def test_adapters_do_not_import_application_or_composition_layers(self):
        adapters_root = Path("src/feverslop/adapters")
        forbidden = [
            "from feverslop.application.",
            "import feverslop.application.",
            "from feverslop.composition.",
            "import feverslop.composition.",
        ]

        offenders = []
        for path in adapters_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)

    def test_prompting_layer_does_not_import_application_layer(self):
        prompting_root = Path("src/feverslop/prompting")
        forbidden = [
            "from feverslop.application.",
            "import feverslop.application.",
        ]
        offenders = []
        for path in prompting_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)

    def test_composition_layer_does_not_import_root_compatibility_facades(self):
        composition_root = Path("src/feverslop/composition")
        forbidden = [
            "from storyboard_renderer import",
            "from workflow_patcher import",
            "from ltx_video_renderer import",
        ]
        offenders = []
        for path in composition_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)

    def test_config_layer_does_not_import_concrete_adapters(self):
        config_root = Path("src/feverslop/config")
        forbidden = [
            "from feverslop.adapters.",
            "import feverslop.adapters.",
        ]
        offenders = []
        for path in config_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
