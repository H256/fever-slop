import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.application.global_character_creator import (
    AssetIdea,
    GuidedAssetGenerator,
)
from feverslop.domain.global_library import AssetKind, GlobalAsset


class GlobalAssetGeneratorTests(unittest.TestCase):
    def test_preview_normalizes_typed_intake_and_requires_explicit_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            generator = GuidedAssetGenerator(GlobalLibraryAdapter(Path(temp) / "library"), profiles={"character-sheet-v1": lambda **_: {}})
            idea = AssetIdea(kind="character", asset_id=" ava ", name=" Ava ", visual_concept="silver bob")
            preview = generator.preview(idea, profile_id="character-sheet-v1")
            self.assertEqual("ava", preview["asset_id"])
            self.assertEqual("character-sheet-v1", preview["workflow_profile"])
            with self.assertRaises(ValueError):
                generator.preview(idea, profile_id="unknown")

    def test_generate_registers_validated_media_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def workflow(**kwargs):
                output = Path(kwargs["run_dir"]) / "hero.png"
                output.write_bytes(b"generated")
                return {"hero_image": "hero.png"}
            generator = GuidedAssetGenerator(GlobalLibraryAdapter(root / "library"), profiles={"character-sheet-v1": workflow}, runs_root=root / "runs")
            result = generator.generate(AssetIdea("character", "ava", "Ava", "silver bob"), profile_id="character-sheet-v1")
            self.assertEqual("ava", result.asset.id)
            self.assertTrue((root / "library" / "character" / "ava" / "hero.png").is_file())
            resumed = generator.resume(result.run_id)
            self.assertEqual(result.run_id, resumed.run_id)
            self.assertEqual("completed", resumed.status)

    def test_generate_recovers_orphan_asset_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def workflow(**kwargs):
                output = Path(kwargs["run_dir"]) / "hero.png"
                output.write_bytes(b"generated")
                return {"hero_image": "hero.png"}
            library = GlobalLibraryAdapter(root / "library")
            orphan = library.root / "character" / "ava"
            orphan.mkdir(parents=True)
            generator = GuidedAssetGenerator(library, profiles={"character-sheet-v1": workflow}, runs_root=root / "runs")
            result = generator.generate(AssetIdea("character", "ava", "Ava", "silver bob"), profile_id="character-sheet-v1")
            self.assertEqual("ava", result.asset.id)
            self.assertTrue((orphan / "manifest.json").is_file())
            self.assertTrue((orphan / "hero.png").is_file())

    def test_generate_refuses_when_asset_manifest_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def workflow(**kwargs):
                output = Path(kwargs["run_dir"]) / "hero.png"
                output.write_bytes(b"generated")
                return {"hero_image": "hero.png"}
            library = GlobalLibraryAdapter(root / "library")
            library.create(GlobalAsset("ava", AssetKind.CHARACTER, "Ava"))
            generator = GuidedAssetGenerator(library, profiles={"character-sheet-v1": workflow}, runs_root=root / "runs")
            with self.assertRaises(FileExistsError) as ctx:
                generator.generate(AssetIdea("character", "ava", "Ava", "silver bob"), profile_id="character-sheet-v1")
            self.assertIn("already exists", str(ctx.exception))
            self.assertEqual(1, library.get("character", "ava").revision)

    def test_idea_requires_concept_and_named_profile(self):
        with self.assertRaises(ValueError):
            AssetIdea("prop", "lamp", "Lamp", "")


if __name__ == "__main__":
    unittest.main()
