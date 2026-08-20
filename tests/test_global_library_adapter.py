import json
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

    def test_materialize_writes_snapshot_manifest_atomically(self):
        source = self.root / "character" / "ava" / "looks" / "default" / "hero.png"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"hero")
        asset = GlobalAsset(
            "ava", AssetKind.CHARACTER, "Ava",
            looks=(AssetLook("default", "Default", hero_image="looks/default/hero.png"),),
        )
        self.adapter.create(asset)
        destination = self.adapter.materialize(AssetKind.CHARACTER, "ava", "default", Path(self.temp_dir.name) / "project" / "references")

        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual("ava", manifest["asset_id"])
        self.assertEqual("character", manifest["kind"])
        self.assertEqual("default", manifest["look_id"])
        self.assertEqual(1, manifest["revision"])
        self.assertEqual([], [p for p in destination.rglob("*") if p.name.endswith(".tmp")])

    def test_delete_requires_existing_asset_and_removes_only_asset_directory(self):
        asset = GlobalAsset("lamp", AssetKind.PROP, "Lamp")
        self.adapter.create(asset)
        self.adapter.delete(AssetKind.PROP, "lamp")
        self.assertFalse((self.root / "prop" / "lamp").exists())
        with self.assertRaises(FileNotFoundError):
            self.adapter.delete(AssetKind.PROP, "lamp")

    def test_materialize_copies_multiview_artifacts(self):
        asset_dir = self.root / "location" / "room"
        for relative, payload in {
            "looks/default/anchor.png": b"anchor",
            "looks/default/sequence.mp4": b"sequence",
            "looks/default/frame_0001.png": b"frame",
            "looks/default/sheet.png": b"sheet",
        }.items():
            target = asset_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        asset = GlobalAsset(
            "room",
            AssetKind.LOCATION,
            "Room",
            looks=(AssetLook(
                "default",
                "Default",
                anchor_image="looks/default/anchor.png",
                sequence_video="looks/default/sequence.mp4",
                selected_frames=("looks/default/frame_0001.png",),
                sheet_image="looks/default/sheet.png",
            ),),
        )
        self.adapter.create(asset)

        destination = self.adapter.materialize(
            AssetKind.LOCATION,
            "room",
            "default",
            Path(self.temp_dir.name) / "project" / "references",
        )

        self.assertEqual(b"anchor", (destination / "anchor.png").read_bytes())
        self.assertEqual(b"sequence", (destination / "sequence.mp4").read_bytes())
        self.assertEqual(b"frame", (destination / "frame_0001.png").read_bytes())
        self.assertEqual(b"sheet", (destination / "sheet.png").read_bytes())

    def test_update_look_artifacts_publishes_one_new_revision(self):
        asset = GlobalAsset(
            "room",
            AssetKind.LOCATION,
            "Room",
            looks=(AssetLook("default", "Default"),),
        )
        self.adapter.create(asset)
        source = Path(self.temp_dir.name) / "run"
        source.mkdir()
        artifacts = {}
        for name, payload in {
            "anchor.png": b"anchor",
            "sequence.mp4": b"sequence",
            "frame_0001.png": b"frame",
            "sheet.png": b"sheet",
        }.items():
            path = source / name
            path.write_bytes(payload)
            artifacts[name] = path

        updated = self.adapter.update_look_artifacts(
            AssetKind.LOCATION,
            "room",
            "default",
            anchor_image=artifacts["anchor.png"],
            sequence_video=artifacts["sequence.mp4"],
            selected_frames=(artifacts["frame_0001.png"],),
            sheet_image=artifacts["sheet.png"],
            provenance={"backend": "ltx", "profile": "sequence_to_sheet_ltx_v1"},
            expected_revision=1,
        )

        self.assertEqual(2, updated.revision)
        stored = self.adapter.get(AssetKind.LOCATION, "room").looks[0]
        self.assertEqual("looks/default/anchor.png", stored.anchor_image)
        self.assertEqual("looks/default/sequence.mp4", stored.sequence_video)
        self.assertEqual(("looks/default/frame_0001.png",), stored.selected_frames)
        self.assertEqual("ltx", dict(stored.metadata)["backend"])
        self.assertEqual(b"sheet", (self.root / "location" / "room" / "looks" / "default" / "sheet.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
