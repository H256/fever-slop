from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).parents[1]

MOVED_WORKFLOWS = {
    "workflows/audio/audio-model/audio_song_v2.json": "2e896ff41503cf52f9188a1f7937a1e17e6312137aa0e24befe8dfa628d73e0c",
    "workflows/image/image-model/image_detail_easyuse_startframe_v1.json": "ad6bbe40a2596734397bf7f6a71766403df346e7ceb23da383c4106b3c75d99d",
    "workflows/image/image-model/image_edit_flux2_klein_1ref_v1.json": "351f565cda405ce0095440d13984fa9eeea12f679f11c1d9f4f42d78207d0b91",
    "workflows/image/image-model/image_edit_flux2_klein_2ref_v1.json": "2604b7dc57997da034b42944700d647a88f3d83fbac7d32e205b6aa8df322c6a",
    "workflows/image/image-model/image_mask_sam3_actor_regions_v1.json": "907c6eab5dbffae4b0ea511422beb4fa3ed4f4dbef9bd19a7d17bfb0f982e8b7",
    "workflows/image/image-model/image_repair_sdxl_ipadapter_identity_v1.json": "6b2cda1e37614825249590a2c5904efa4cb372ea4b58a1b89427e6f3451cf2d7",
    "workflows/image/image-model/image_t2i_startframe_ideogram_director_v1.json": "02920c9db946b8b93549e4b3ae3d9b5cb4655bb1baaf66c74bc12c5106b98a7e",
    "workflows/image/image-model/image_t2i_startframe_ideogram_v1.json": "02920c9db946b8b93549e4b3ae3d9b5cb4655bb1baaf66c74bc12c5106b98a7e",
    "workflows/image/image-model/image_t2i_startframe_krea_v1.json": "734e2e2f36e32c44d20fcbab873d392de2c6a29c96cb8ffbba0c5e15f3ddbad3",
    "workflows/image/image-model/image_t2i_startframe_v1.json": "cefe75fe11a88a1390a26d0c6e0817ddf02ecda577b10f36efbb60ed15aa60f3",
    "workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json": "f743b95b71414c488d72ecc54a95820cdf1df9e00057e85982c7787760c361ee",
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
