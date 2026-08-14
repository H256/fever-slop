from __future__ import annotations

import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from feverslop.prompting.guide_loader import load_markdown_guide


class PromptingPackagingTests(unittest.TestCase):
    def test_migrated_movie_prompt_path_has_no_jinja_templates(self):
        prompt_dir = Path(__file__).parents[1] / "src" / "feverslop" / "adapters" / "prompts"

        self.assertEqual([], list(prompt_dir.glob("*.j2")))

    def test_every_source_guide_is_available_through_resources(self):
        guide_dir = Path(__file__).parents[1] / "src" / "feverslop" / "prompting" / "guides"
        source_guides = sorted(path.stem for path in guide_dir.glob("*.md"))

        self.assertGreater(len(source_guides), 0)
        for name in source_guides:
            self.assertTrue(load_markdown_guide(name).strip(), name)

    def test_guides_are_in_both_build_artifacts(self):
        project_root = Path(__file__).parents[1]
        source_guides = {path.name for path in (project_root / "src/feverslop/prompting/guides").glob("*.md")}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ["uv", "build", "--sdist", "--wheel", "--out-dir", temp_dir],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                self.fail(f"uv build failed:\n{result.stdout}\n{result.stderr}")

            artifacts = list(Path(temp_dir).iterdir())
            sdist = next(path for path in artifacts if path.suffix == ".gz")
            wheel = next(path for path in artifacts if path.suffix == ".whl")

            with tarfile.open(sdist) as archive:
                packaged_sdist = {Path(member.name).name for member in archive.getmembers()}
            with zipfile.ZipFile(wheel) as archive:
                packaged_wheel = {Path(name).name for name in archive.namelist()}

        self.assertTrue(source_guides <= packaged_sdist)
        self.assertTrue(source_guides <= packaged_wheel)


if __name__ == "__main__":
    unittest.main()
