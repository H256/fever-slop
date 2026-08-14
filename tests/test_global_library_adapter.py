import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset


class GlobalLibraryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "library"
        self.adapter = GlobalLibraryAdapter(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_crud_and_revision_conflict(self):
        asset = GlobalAsset("ava", AssetKind.CHARACTER, "Ava", looks=(AssetLook("default", "Default"),))
        self.adapter.create(asset)
        self.assertEqual(asset, self.adapter.get(AssetKind.CHARACTER, "ava"))

        changed = GlobalAsset("ava", AssetKind.CHARACTER, "Ava Renamed", looks=asset.looks, revision=2)
        self.adapter.update(changed, expected_revision=1)
        with self.assertRaises(ValueError):
            self.adapter.update(GlobalAsset("ava", AssetKind.CHARACTER, "Stale", revision=3), expected_revision=1)

    def test_materialize_is_project_local_and_reports_stale_revision(self):
        source = self.root / "character" / "ava" / "looks" / "default" / "hero.png"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"hero")
        asset = GlobalAsset(
            "ava", AssetKind.CHARACTER, "Ava",
            looks=(AssetLook("default", "Default", hero_image="looks/default/hero.png"),),
        )
        self.adapter.create(asset)
        destination = self.adapter.materialize(AssetKind.CHARACTER, "ava", "default", Path(self.temp_dir.name) / "project" / "references")

        self.assertEqual(b"hero", (destination / "hero.png").read_bytes())
        self.assertEqual(1, self.adapter.snapshot_revision(destination))
        self.assertFalse(self.adapter.is_stale(destination))

        self.adapter.update(GlobalAsset("ava", AssetKind.CHARACTER, "Ava", looks=asset.looks, revision=2), expected_revision=1)
        self.assertTrue(self.adapter.is_stale(destination))

    def test_delete_requires_existing_asset_and_removes_only_asset_directory(self):
        asset = GlobalAsset("lamp", AssetKind.PROP, "Lamp")
        self.adapter.create(asset)
        self.adapter.delete(AssetKind.PROP, "lamp")
        self.assertFalse((self.root / "prop" / "lamp").exists())
        with self.assertRaises(FileNotFoundError):
            self.adapter.delete(AssetKind.PROP, "lamp")


if __name__ == "__main__":
    unittest.main()
