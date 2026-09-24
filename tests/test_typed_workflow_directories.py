from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).parents[1]

MOVED_WORKFLOWS = {
    "workflows/audio/audio-model/audio_song_v2.json": "e3f0a7b858b08ac34ddd94fbe71f27273832409140e1ad95d2357b33c5e8afe2",
    "workflows/image/image-model/image_detail_easyuse_startframe_v1.json": "82bcae0aea4deaa1d44012e257c04b1e3e1def8d4ce998cf0371a82c86cbe0cb",
    "workflows/image/image-model/image_edit_flux2_klein_1ref_v1.json": "b563053349eceac1e188c6d972f45e22b7128862e27f0adf1b8eb5a99c3e0259",
    "workflows/image/image-model/image_edit_flux2_klein_2ref_v1.json": "fa9513a20b012485fccaad7d90fb82b18bcfbc0ab8d86d970844b8953f6a8ede",
    "workflows/image/image-model/image_mask_sam3_actor_regions_v1.json": "c812418e296e6a38554929aea1f781a47af1e0b55c75243b3d67df8d5b38d328",
    "workflows/image/image-model/image_repair_sdxl_ipadapter_identity_v1.json": "2dc7662494e1b2a3a0208266de64be683b712c3c2cbfff8fec04733c73d6b690",
    "workflows/image/image-model/image_t2i_startframe_ideogram_director_v1.json": "cca87adb15f1e5c30eda924faaddab892f070fb0712b4d60045d27ddd87bb5cd",
    "workflows/image/image-model/image_t2i_startframe_ideogram_v1.json": "cca87adb15f1e5c30eda924faaddab892f070fb0712b4d60045d27ddd87bb5cd",
    "workflows/image/image-model/image_t2i_startframe_krea_v1.json": "eee4df82ed8e6ae96c2fc8138976839d7a179033b63b42cb3ea3c9469ddd5d7c",
    "workflows/image/image-model/image_t2i_startframe_v1.json": "87d68b51b5a8ef67ae0682c6c200754ac29bcfa05330c0fb9a098feb5ccd79ed",
    "workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json": "6bed4b802b8f1b116b397e87f17abbf04ccc6525a2378ac5e0767a6dd523bf40",
}


class TypedWorkflowDirectoryTests(unittest.TestCase):
    def test_moved_workflows_exist_with_baseline_bytes_and_valid_json(self):
        for relative, expected_hash in MOVED_WORKFLOWS.items():
            path = REPO_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(expected_hash, hashlib.sha256(path.read_bytes()).hexdigest(), relative)
            self.assertIsInstance(json.loads(path.read_text(encoding="utf-8-sig")), dict, relative)

    def test_legacy_root_workflow_paths_are_not_maintained_assets(self):
        for relative in MOVED_WORKFLOWS:
            filename = Path(relative).name
            self.assertFalse((REPO_ROOT / "workflows" / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
