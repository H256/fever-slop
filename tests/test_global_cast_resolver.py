import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.application.global_cast_resolver import GlobalCastResolver
from feverslop.config.project_config import GlobalAssetConfig
from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset


class GlobalCastResolverTests(unittest.TestCase):
    def test_resolves_global_cast_and_materializes_a_portable_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = GlobalLibraryAdapter(root / "library")
            media = root / "library" / "character" / "ava" / "looks" / "default" / "hero.png"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"hero")
            adapter.create(GlobalAsset("ava", AssetKind.CHARACTER, "Ava", description="Singer", looks=(
                AssetLook("default", "Default", hero_image="looks/default/hero.png"),
            )))
            result = GlobalCastResolver(adapter).resolve(
                cast=(GlobalAssetConfig("ava", "default", "lead"),),
                locations=(), styles=(), props=(),
                project_reference_dir=root / "project" / "references",
            )

            self.assertEqual("ava", result.actors[0]["id"])
            self.assertEqual("lead", result.actors[0]["role"])
            self.assertTrue(Path(result.actors[0]["hero_path"]).is_file())
            self.assertEqual(1, result.snapshots[0]["revision"])

    def test_missing_asset_and_look_explain_the_fix(self):
        with tempfile.TemporaryDirectory() as temp:
            resolver = GlobalCastResolver(GlobalLibraryAdapter(Path(temp) / "library"))
            with self.assertRaises(ValueError) as missing:
                resolver.resolve(
                    cast=(GlobalAssetConfig("missing"),), locations=(), styles=(), props=(),
                    project_reference_dir=Path(temp) / "refs",
                )
            self.assertIn("create or import", str(missing.exception))

    def test_resolves_multiview_paths_for_reference_consumers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = GlobalLibraryAdapter(root / "library")
            look_dir = root / "library" / "location" / "room" / "looks" / "default"
            look_dir.mkdir(parents=True)
            for name in ("anchor.png", "sequence.mp4", "sheet.png", "frame_0001.png"):
                (look_dir / name).write_bytes(name.encode())
            adapter.create(GlobalAsset("room", AssetKind.LOCATION, "Room", looks=(AssetLook(
                "default", "Default", anchor_image="looks/default/anchor.png",
                sequence_video="looks/default/sequence.mp4", sheet_image="looks/default/sheet.png",
                selected_frames=("looks/default/frame_0001.png",),
            ),)))

            result = GlobalCastResolver(adapter).resolve(
                cast=(), locations=(GlobalAssetConfig("room", "default", "set"),), styles=(), props=(),
                project_reference_dir=root / "project" / "references",
            )

            location = result.locations[0]
            self.assertTrue(Path(location["anchor_path"]).is_file())
            self.assertTrue(Path(location["sequence_path"]).is_file())
            self.assertEqual(1, len(location["selected_frame_paths"]))


if __name__ == "__main__":
    unittest.main()
