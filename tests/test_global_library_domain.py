import json
import unittest

from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset


class GlobalLibraryDomainTests(unittest.TestCase):
    def test_manifest_round_trips_all_asset_kinds_and_looks(self):
        asset = GlobalAsset(
            id="ava",
            kind=AssetKind.CHARACTER,
            name="Ava",
            description="A singer with a silver bob.",
            looks=(AssetLook(id="default", name="Default", hero_image="looks/default/hero.png"),),
            revision=3,
        )

        restored = GlobalAsset.from_dict(json.loads(json.dumps(asset.to_dict())))

        self.assertEqual(asset, restored)
        self.assertEqual("character", restored.to_dict()["kind"])

    def test_manifest_rejects_invalid_kind_duplicate_look_and_unsafe_media_path(self):
        with self.assertRaises(ValueError):
            GlobalAsset.from_dict({"id": "x", "kind": "vehicle", "name": "X", "looks": []})
        with self.assertRaises(ValueError):
            GlobalAsset(
                id="x", kind=AssetKind.PROP, name="X",
                looks=(AssetLook(id="same", name="A"), AssetLook(id="same", name="B")),
            )
        with self.assertRaises(ValueError):
            AssetLook(id="default", name="Default", hero_image="../outside.png")

    def test_ids_are_stable_and_display_names_do_not_change_them(self):
        first = GlobalAsset(id="nightclub", kind=AssetKind.LOCATION, name="Night Club", looks=())
        renamed = GlobalAsset(id="nightclub", kind=AssetKind.LOCATION, name="The Neon Nightclub", looks=())

        self.assertEqual(first.id, renamed.id)
        self.assertNotEqual(first.name, renamed.name)

    def test_multiview_artifacts_round_trip_without_using_sheet_as_anchor(self):
        look = AssetLook(
            id="default",
            name="Default",
            anchor_image="looks/default/anchor.png",
            sheet_image="looks/default/sheet.png",
            sequence_video="looks/default/sequence.mp4",
            selected_frames=("looks/default/frame_0001.png", "looks/default/frame_0002.png"),
        )

        restored = AssetLook.from_dict(look.to_dict())

        self.assertEqual(look, restored)
        self.assertEqual("looks/default/anchor.png", restored.anchor_image)
        self.assertNotEqual(restored.anchor_image, restored.sheet_image)


if __name__ == "__main__":
    unittest.main()
