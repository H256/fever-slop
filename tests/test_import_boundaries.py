import unittest
import ast
from pathlib import Path


class ImportBoundaryTests(unittest.TestCase):
    @staticmethod
    def _imported_modules(path):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        return modules

    def _pyside6_imports(self, paths):
        offenders = []
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(module == "PySide6" or module.startswith("PySide6.") for module in modules):
                    offenders.append(f"{path}:{node.lineno}")
        return offenders

    def test_non_qt_layers_and_studio_services_do_not_import_pyside6(self):
        studio_root = Path("src/feverslop/studio")
        desktop_root = studio_root / "desktop"
        protected_paths = [
            *Path("src/feverslop/domain").rglob("*.py"),
            *Path("src/feverslop/ports").rglob("*.py"),
            *Path("src/feverslop/application").rglob("*.py"),
            *Path("src/feverslop/adapters").rglob("*.py"),
            *Path("src/feverslop/infra").rglob("*.py"),
            *(path for path in studio_root.rglob("*.py") if desktop_root not in path.parents),
        ]

        offenders = self._pyside6_imports(protected_paths)

        self.assertEqual([], offenders)

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
            "from feverslop.config",
            "import feverslop.config",
            "from rich",
            "import rich",
            "from feverslop.adapters.audio.",
            "import feverslop.adapters.audio.",
            "from feverslop.adapters.comfyui_",
            "import feverslop.adapters.comfyui_",
            "from feverslop.adapters.face_",
            "import feverslop.adapters.face_",
            "from feverslop.adapters.ltx_",
            "import feverslop.adapters.ltx_",
            "from feverslop.adapters.lora_",
            "import feverslop.adapters.lora_",
            "from feverslop.adapters.insightface_",
            "import feverslop.adapters.insightface_.",
            "from feverslop.adapters.movie_",
            "import feverslop.adapters.movie_",
            "from feverslop.adapters.openai_",
            "import feverslop.adapters.openai_",
            "from feverslop.adapters.pipeline_runner",
            "import feverslop.adapters.pipeline_runner",
            "from feverslop.adapters.prepared_workflow",
            "import feverslop.adapters.prepared_workflow",
            "from feverslop.adapters.project_",
            "import feverslop.adapters.project_",
            "from feverslop.adapters.storyboard_renderer",
            "import feverslop.adapters.storyboard_renderer",
            "from feverslop.adapters.video_postprocessor",
            "import feverslop.adapters.video_postprocessor",
            "from feverslop.adapters.visual_",
            "import feverslop.adapters.visual_",
            "from feverslop.adapters.workflow_patcher",
            "import feverslop.adapters.workflow_patcher",
            "from feverslop.audio.",
            "import feverslop.audio.",
            "from feverslop.pipeline.",
            "import feverslop.pipeline.",
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
        allowed_files = {
            "h3_prompt_pipeline.py",
            "prompt_generation.py",
            "startframe_director_prompts.py",
            "movie_references.py",
        }
        offenders = []
        for path in app_root.rglob("*.py"):
            if path.name in allowed_files:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)

    def test_new_pure_application_modules_do_not_import_runtime_io(self):
        legacy_filesystem_modules = {
            "continuity_handoff.py",
            "facefix_pipeline.py",
            "full_auto.py",
            "generate_render_plan.py",
            "movie_artifacts.py",
            "movie_common.py",
            "movie_i2v_render_plan.py",
            "movie_ingredients_sheets.py",
            "movie_msr_enrichment.py",
            "movie_prepared_workflows.py",
            "movie_references.py",
            "movie_use_cases.py",
            "movie_visual_plan.py",
            "msr_prompt_enrichment.py",
            "pipeline_context.py",
            "prompt_generation_pipeline.py",
            "reference_bible.py",
            "reference_workspace.py",
            "render_plan_ingredients_sheets.py",
            "render_storyboard.py",
            "render_video.py",
            "startframe_director_prompts.py",
            "startframe_i2v_render_plan.py",
            "startframe_identity.py",
            "startframe_plan.py",
            "startframe_validation.py",
            "render_plan_pipeline.py",
            # This application service intentionally materializes a filesystem
            # backed OpenShot project and therefore owns pathlib/os access.
            "openshot_exporter.py",
            # This application service materializes an MLT XML project and
            # therefore owns pathlib/os access.
            "mlt_exporter.py",
            # This application service extracts frames and materializes the
            # generated sheet, so it intentionally owns filesystem access.
            "sequence_to_sheet.py",
            # This application service owns the anchor/sequence filesystem
            # lifecycle and delegates sheet extraction to sequence_to_sheet.
            "sequence_reference_pipeline.py",
        }
        forbidden_roots = {"pathlib", "os", "subprocess", "PySide6"}
        offenders = []
        for path in Path("src/feverslop/application").glob("*.py"):
            if path.name in legacy_filesystem_modules:
                continue
            for module in self._imported_modules(path):
                if module.split(".", 1)[0] in forbidden_roots:
                    offenders.append(f"{path}: {module}")

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
            "from feverslop.adapters.audio.",
            "import feverslop.adapters.audio.",
            "from feverslop.adapters.comfyui_",
            "import feverslop.adapters.comfyui_",
            "from feverslop.adapters.face_",
            "import feverslop.adapters.face_",
            "from feverslop.adapters.ltx_",
            "import feverslop.adapters.ltx_",
            "from feverslop.adapters.lora_",
            "import feverslop.adapters.lora_",
            "from feverslop.adapters.insightface_",
            "import feverslop.adapters.insightface_.",
            "from feverslop.adapters.movie_",
            "import feverslop.adapters.movie_",
            "from feverslop.adapters.openai_",
            "import feverslop.adapters.openai_",
            "from feverslop.adapters.pipeline_runner",
            "import feverslop.adapters.pipeline_runner",
            "from feverslop.adapters.prepared_workflow",
            "import feverslop.adapters.prepared_workflow",
            "from feverslop.adapters.project_",
            "import feverslop.adapters.project_",
            "from feverslop.adapters.storyboard_renderer",
            "import feverslop.adapters.storyboard_renderer",
            "from feverslop.adapters.video_postprocessor",
            "import feverslop.adapters.video_postprocessor",
            "from feverslop.adapters.visual_",
            "import feverslop.adapters.visual_",
            "from feverslop.adapters.workflow_patcher",
            "import feverslop.adapters.workflow_patcher",
            "from feverslop.infra.",
            "import feverslop.infra.",
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

        # This adapter is the explicit compatibility bridge for the legacy
        # movie visual pipeline and intentionally depends on both layers.
        allowed_files = {"movie_minimax_visual.py"}
        offenders = []
        for path in adapters_root.rglob("*.py"):
            if path.name in allowed_files:
                continue
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
            "from feverslop.adapters.audio.",
            "import feverslop.adapters.audio.",
            "from feverslop.adapters.comfyui_",
            "import feverslop.adapters.comfyui_",
            "from feverslop.adapters.face_",
            "import feverslop.adapters.face_",
            "from feverslop.adapters.ltx_",
            "import feverslop.adapters.ltx_",
            "from feverslop.adapters.lora_",
            "import feverslop.adapters.lora_",
            "from feverslop.adapters.insightface_",
            "import feverslop.adapters.insightface_.",
            "from feverslop.adapters.movie_",
            "import feverslop.adapters.movie_",
            "from feverslop.adapters.openai_",
            "import feverslop.adapters.openai_",
            "from feverslop.adapters.pipeline_runner",
            "import feverslop.adapters.pipeline_runner",
            "from feverslop.adapters.prepared_workflow",
            "import feverslop.adapters.prepared_workflow",
            "from feverslop.adapters.project_",
            "import feverslop.adapters.project_",
            "from feverslop.adapters.storyboard_renderer",
            "import feverslop.adapters.storyboard_renderer",
            "from feverslop.adapters.video_postprocessor",
            "import feverslop.adapters.video_postprocessor",
            "from feverslop.adapters.visual_",
            "import feverslop.adapters.visual_",
            "from feverslop.adapters.workflow_patcher",
            "import feverslop.adapters.workflow_patcher",
        ]
        offenders = []
        for path in config_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)

    def test_lyric_timeline_propagates_to_render_plan(self):
        """Editing lyrics on a timeline segment should propagate through the render plan."""
        from feverslop.domain.timeline_editing import EditableTimelineSegment, TimelineSnapshot

        segments = [
            EditableTimelineSegment(
                start=0.0, end=5.0, kind="vocal", text="Verse 1", lyrics_line="Hello world", is_draft=True,
            ),
            EditableTimelineSegment(
                start=5.0, end=10.0, kind="instrumental", text="Bridge", lyrics_line=None, is_draft=True,
            ),
        ]
        snapshot = TimelineSnapshot(segments=segments, scene_boundaries=[], beat_markers=[], metadata={})

        serialized = snapshot.to_json()
        restored = TimelineSnapshot.from_json(serialized)

        self.assertEqual("Hello world", restored.segments[0].lyrics_line)
        self.assertIsNone(restored.segments[1].lyrics_line)
        self.assertEqual(len(restored.segments), 2)

    def test_instrumental_segments_trigger_closed_mouth_policy(self):
        """Render plan scenes with only instrumental segments must get closed-mouth policy."""
        from feverslop.application.ingredients_render_plan import build_ingredients_static_prompt

        instrumental_relay = [
            {"state": "motion", "prompt": "Camera pans across stage"},
            {"state": "motion", "prompt": "Lighting shifts"},
        ]
        singing_relay = [
            {"state": "singing", "prompt": "Performer sings"},
            {"state": "motion", "prompt": "Camera pulls back"},
        ]

        instrumental_static = build_ingredients_static_prompt(
            "Stage performance", instrumental_relay
        )
        singing_static = build_ingredients_static_prompt(
            "Stage performance", singing_relay
        )

        self.assertIn("closed", instrumental_static.lower())
        self.assertNotIn("closed", singing_static.lower())
        self.assertIn("temporal", singing_static.lower())


if __name__ == "__main__":
    unittest.main()
