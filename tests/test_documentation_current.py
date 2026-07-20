from pathlib import Path
import unittest


class DocumentationCurrentTests(unittest.TestCase):
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
