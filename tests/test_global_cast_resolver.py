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


if __name__ == "__main__":
    unittest.main()
