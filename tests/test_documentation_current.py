from pathlib import Path
import unittest


class DocumentationCurrentTests(unittest.TestCase):
    def test_readme_does_not_reference_removed_dead_code_candidates(self):
        text = Path("README.md").read_text(encoding="utf-8")

        self.assertNotIn("prompt_pipeline_batch_patch.py", text)
        self.assertNotIn("extract_lyrics.py", text)
        self.assertNotIn("noise_reduction.py", text)

    def test_readme_mentions_current_package_layout(self):
        text = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("FeverSlop", text)
        self.assertIn("src/feverslop", text)
        self.assertIn("feverslop.adapters.comfyui_video_backend", text)
        self.assertIn("feverslop.composition.generate_render_plan", text)

    def test_architecture_compatibility_mentions_final_boundaries(self):
        text = Path("docs/architecture_compatibility.md").read_text(encoding="utf-8")

        self.assertIn("feverslop.composition", text)
        self.assertIn("application layer does not import concrete adapters", text)
        self.assertIn("ports do not import adapters", text)


if __name__ == "__main__":
    unittest.main()
