import unittest

from feverslop.domain.global_library import AssetKind
from feverslop.domain.reference_sheet import (
    AnchorKind,
    ReferenceArtifact,
    ReferenceSheetProvenance,
    ReferenceSheetRequest,
    ReferenceSheetResult,
)


class ReferenceSheetDomainTests(unittest.TestCase):
    def test_request_accepts_only_a_single_view_anchor(self):
        request = ReferenceSheetRequest(
            asset_kind=AssetKind.CHARACTER,
            asset_id="ava",
            look_id="default",
            anchor_image="looks/default/anchor.png",
            backend="ltx",
            profile="character_turnaround",
            anchor_kind=AnchorKind.SINGLE_VIEW,
        )

        self.assertEqual("looks/default/anchor.png", request.anchor_image)
        self.assertEqual(AnchorKind.SINGLE_VIEW, request.anchor_kind)

    def test_request_rejects_an_implicit_four_panel_sheet_anchor(self):
        with self.assertRaises(ValueError):
            ReferenceSheetRequest(
                asset_kind=AssetKind.CHARACTER,
                asset_id="ava",
                look_id="default",
                anchor_image="looks/default/hero_sheet.png",
                backend="ltx",
                profile="character_turnaround",
                anchor_kind=AnchorKind.SHEET,
            )

    def test_request_allows_explicit_sheet_anchor_override(self):
        request = ReferenceSheetRequest(
            asset_kind=AssetKind.CHARACTER,
            asset_id="ava",
            look_id="default",
            anchor_image="looks/default/hero_sheet.png",
            backend="ltx",
            profile="character_turnaround",
            anchor_kind=AnchorKind.SHEET,
            allow_sheet_anchor=True,
        )

        self.assertTrue(request.allow_sheet_anchor)

    def test_result_round_trips_artifacts_and_provenance(self):
        result = ReferenceSheetResult(
            request_fingerprint="abc123",
            artifacts=(
                ReferenceArtifact(kind="sequence", path="runs/abc/sequence.mp4"),
                ReferenceArtifact(kind="sheet", path="looks/default/sheet.png"),
            ),
            provenance=ReferenceSheetProvenance(
                backend="ltx",
                profile="character_turnaround",
                seed=7,
                prompt_revision="reference-sheet-v1",
            ),
        )

        self.assertEqual(result, ReferenceSheetResult.from_dict(result.to_dict()))


if __name__ == "__main__":
    unittest.main()
