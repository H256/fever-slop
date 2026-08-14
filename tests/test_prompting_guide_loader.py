import unittest
from pathlib import Path
import tempfile

from feverslop.prompting.guide_loader import (
    PromptGuideNotFoundError,
    load_markdown_guide,
    resolve_guide_path,
)


class PromptingGuideLoaderTests(unittest.TestCase):
    def test_loads_bundled_markdown_guide_by_stem_or_filename(self):
        by_stem = load_markdown_guide("minimax-h3-base")
        by_filename = load_markdown_guide("minimax-h3-base.md")

        self.assertEqual(by_stem, by_filename)
        self.assertIn("integrated_multimodal_description", by_stem)

    def test_resolves_legacy_path_to_package_local_guide_filename(self):
        path = resolve_guide_path(Path("/tmp/not-the-package/minimax-h3-references.md"))

        self.assertEqual("minimax-h3-references.md", path.name)
        self.assertTrue(path.is_file())

    def test_explicit_path_is_narrowed_to_package_local_guide_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom = Path(temp_dir) / "minimax-h3-base.md"
            custom.write_text("custom guide", encoding="utf-8")

            loaded = load_markdown_guide(custom)

        self.assertIn("integrated_multimodal_description", loaded)
        self.assertNotEqual("custom guide", loaded)

    def test_missing_guide_error_names_the_requested_markdown_file(self):
        with self.assertRaises(PromptGuideNotFoundError) as caught:
            load_markdown_guide("../not-bundled")

        self.assertIn("not-bundled.md", str(caught.exception))
        self.assertIn("Prompt guide not found", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
