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

        self.assertIn("src/autoprompter", text)
        self.assertIn("autoprompter.adapters.comfyui_video_backend", text)
        self.assertIn("autoprompter.composition.generate_render_plan", text)


if __name__ == "__main__":
    unittest.main()
