import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
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

    def test_get_never_observes_a_half_deleted_asset(self):
        media_rel = "looks/base/hero.png"
        asset = GlobalAsset(
            "ava", AssetKind.CHARACTER, "Ava",
            looks=(AssetLook("base", "Base", hero_image=media_rel),),
        )
        self.adapter.create(asset)
        media_path = self.root / "character" / "ava" / media_rel
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(b"hero")
        asset_dir = self.root / "character" / "ava"
        manifest = asset_dir / "manifest.json"
        stop = threading.Event()
        violations: list[str] = []

        def reader():
            while not stop.is_set():
                try:
                    current = self.adapter.get("character", "ava")
                except FileNotFoundError:
                    continue
                # get() releases its shared lock before returning, so only a missing
                # media file under a still-present manifest counts as a half-deleted view.
                if manifest.is_file():
                    for look in current.looks:
                        if look.hero_image and not (asset_dir / look.hero_image).is_file():
                            violations.append(f"manifest present but {look.hero_image} missing")
                time.sleep(0.001)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(0.05)
        self.adapter.delete("character", "ava")
        stop.set()
        thread.join()
        self.assertEqual([], violations)
        self.assertFalse(asset_dir.exists())

    @unittest.skipUnless(os.name == "nt", "Windows unlink semantics")
    def test_delete_tolerates_reader_that_passed_manifest_check_before_delete(self):
        class CoordinatedAdapter(GlobalLibraryAdapter):
            def __init__(self, root):
                super().__init__(root)
                self.coordinate = False
                self.exclusive_exits = 0
                self.reader_ready = threading.Event()
                self.allow_reader = threading.Event()
                self.reader_acquired = threading.Event()
                self.release_reader = threading.Event()

            @contextmanager
            def _lock(self, directory, *, shared=False):
                if self.coordinate and shared:
                    self.reader_ready.set()
                    if not self.allow_reader.wait(5):
                        raise TimeoutError("delete did not release its validation lock")
                with super()._lock(directory, shared=shared):
                    if self.coordinate and shared:
                        self.reader_acquired.set()
                        if not self.release_reader.wait(5):
                            raise TimeoutError("test did not release coordinated reader")
                    yield
                if self.coordinate and not shared:
                    self.exclusive_exits += 1
                    if self.exclusive_exits == 2:
                        self.allow_reader.set()
                        if not self.reader_acquired.wait(5):
                            raise TimeoutError("reader did not acquire the released lock")

        adapter = CoordinatedAdapter(self.root)
        adapter.create(GlobalAsset("ava", AssetKind.CHARACTER, "Ava"))
        adapter.coordinate = True
        reader_error: list[Exception] = []

        def reader():
            try:
                adapter.get("character", "ava")
            except FileNotFoundError:
                pass
            except Exception as exc:
                reader_error.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        self.assertTrue(adapter.reader_ready.wait(5))
        delete_error = None
        try:
            adapter.delete("character", "ava")
        except Exception as exc:
            delete_error = exc
        finally:
            adapter.release_reader.set()
            thread.join(5)

        self.assertIsNone(delete_error)
        self.assertEqual([], reader_error)
        self.assertFalse(thread.is_alive())

    def test_delete_removes_asset_directory_and_second_delete_has_no_side_effects(self):
        self.adapter.create(GlobalAsset("lamp", AssetKind.PROP, "Lamp"))
        asset_dir = self.root / "prop" / "lamp"
        self.adapter.delete(AssetKind.PROP, "lamp")
        self.assertFalse(asset_dir.exists())
        with self.assertRaises(FileNotFoundError):
            self.adapter.delete(AssetKind.PROP, "lamp")
        self.assertFalse(asset_dir.exists())

    @unittest.skipIf(os.name == "nt", "flock-based test")
    def test_materialize_blocks_while_exclusive_lock_held(self):
        import fcntl
        media_rel = "looks/base/hero.png"
        asset = GlobalAsset(
            "ava", AssetKind.CHARACTER, "Ava",
            looks=(AssetLook("base", "Base", hero_image=media_rel),),
        )
        self.adapter.create(asset)
        media_path = self.root / "character" / "ava" / media_rel
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(b"materialize-me")
        lock_handle = (self.root / "character" / "ava" / ".lock").open("a+b")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        finished = threading.Event()
        outcome: dict = {}

        def materialize():
            try:
                outcome["path"] = self.adapter.materialize("character", "ava", "base", Path(self.temp_dir.name) / "project")
            except Exception as exc:
                outcome["error"] = exc
            finally:
                finished.set()

        thread = threading.Thread(target=materialize, daemon=True)
        thread.start()
        try:
            self.assertFalse(finished.wait(0.5), "materialize did not block on the held exclusive lock")
        finally:
            lock_handle.close()
        self.assertTrue(finished.wait(5.0))
        self.assertNotIn("error", outcome)
        self.assertEqual(b"materialize-me", (outcome["path"] / "hero.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
