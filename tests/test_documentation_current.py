from pathlib import Path
import re
import unittest


class DocumentationCurrentTests(unittest.TestCase):
    OPERATIONAL_DOCUMENTS = (
        Path("README.md"),
        Path("docs/setup.md"),
        Path("docs/running.md"),
        Path("docs/pipelines.md"),
        Path("docs/project_workflow.md"),
        Path("docs/app_config.md"),
        Path("docs/comfyui_model_resolution.md"),
        Path("docs/minimax-h3-setup.md"),
        Path("docs/projects.md"),
    )

    def test_operational_docs_do_not_reference_removed_workflows(self):
        removed_workflows = (
            "video_ltxv_i2v_v1.json",
            "video_ltxv_msr_1actor_1background_v2.json",
            "video_ltxv_ingredients_audio_2stage_v5.json",
            "video_ltxv_relay_v1.json",
            "image_seg_sam3_v1.json",
            "image_inpaint_ipadapter_v1.json",
            "image_detail_easyuse_v1.json",
            "video_minimax_h3_i2v.json",
        )
        for path in self.OPERATIONAL_DOCUMENTS:
            text = path.read_text(encoding="utf-8")
            for workflow in removed_workflows:
                with self.subTest(path=path, workflow=workflow):
                    self.assertNotIn(workflow, text)

    def test_operational_workflow_references_exist_or_are_explicit_placeholders(self):
        placeholders = {
            "workflows/example.json",
            "workflows/video_example.json",
            "workflows/x.json",
            "workflows/my_custom_reference_workflow.json",
            "workflows/my_custom_edit_workflow.json",
            "workflows/your_prompt_relay_workflow.json",
        }
        pattern = re.compile(r"(?:\.\\|\./)?(workflows[\\/][A-Za-z0-9_.-]+\.json)")
        for path in self.OPERATIONAL_DOCUMENTS:
            text = path.read_text(encoding="utf-8")
            for raw_reference in pattern.findall(text):
                reference = raw_reference.replace("\\", "/")
                with self.subTest(path=path, reference=reference):
                    if reference in placeholders:
                        continue
                    self.assertTrue(Path(reference).is_file())

    def test_readme_does_not_reference_removed_dead_code_candidates(self):
        text = Path("README.md").read_text(encoding="utf-8")

        self.assertNotIn("prompt_pipeline_batch_patch.py", text)
        self.assertNotIn("extract_lyrics.py", text)
        self.assertNotIn("noise_reduction.py", text)
        self.assertNotIn("test.ps1", text)
        self.assertNotIn("test.bat", text)

    def test_current_workflow_docs_use_python_runner(self):
        for path in [Path("AGENTS.md"), Path("docs/project_workflow.md")]:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                self.assertIn("run_pipeline.py", text)
                self.assertNotIn("test.ps1", text)
                self.assertNotIn("test.bat", text)

    def test_readme_mentions_current_package_layout(self):
        text = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("FeverSlop", text)
        self.assertIn("src/feverslop", text)
        self.assertIn("feverslop.adapters.comfyui_video_backend", text)
        self.assertIn("feverslop.composition.generate_render_plan", text)

    def test_readme_uses_english_primary_documentation(self):
        text = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("## Requirements", text)
        self.assertIn("## Full-Auto", text)
        for german_heading in ("## Voraussetzungen", "## Dateien", "## Projekt config.json", "## Modi"):
            self.assertNotIn(german_heading, text)

    def test_architecture_compatibility_mentions_final_boundaries(self):
        text = Path("docs/architecture_compatibility.md").read_text(encoding="utf-8")

        self.assertIn("feverslop.composition", text)
        self.assertIn("application layer does not import concrete adapters", text)
        self.assertIn("ports do not import adapters", text)

    def test_examples_explain_workflow_scene_duration_limits(self):
        text = Path("docs/examples.md").read_text(encoding="utf-8")

        self.assertIn("default_max_render_duration_seconds", text)
        self.assertIn("video_workflow_limits", text)
        self.assertIn("Requested scene duration", text)


if __name__ == "__main__":
    unittest.main()
