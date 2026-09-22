import base64
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.media_store import MediaStore
from feverslop.ports.project_requests import StudioPathError


def _make_store(tmp: Path, max_upload_size: int) -> MediaStore:
    def resolve(project_id: str, path: str) -> Path:
        return tmp / path

    return MediaStore(
        project_root=lambda project_id: tmp,
        resolve_project_path=resolve,
        read_json_file=lambda p: {},
        max_upload_size=max_upload_size,
    )


class MediaPersistenceAdapterTests(unittest.TestCase):
    def test_media_store_is_available_from_adapters(self):
        self.assertTrue(callable(MediaStore))

    def test_valid_upload_writes_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(Path(tmp), max_upload_size=100)
            payload = b"\x89PNG\x00test-bytes"
            data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
            result = store.write_media_data_url("p", "img.png", data_url)
            self.assertEqual(result, {"path": "img.png"})
            self.assertEqual((Path(tmp) / "img.png").read_bytes(), payload)

    def test_oversized_payload_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(Path(tmp), max_upload_size=100)
            # 1000 base64 chars -> ~750 decoded bytes, far over the 100 limit.
            data_url = "data:image/png;base64," + "A" * 1000
            with self.assertRaises(StudioPathError) as ctx:
                store.write_media_data_url("p", "img.png", data_url)
            self.assertIn("estimated", str(ctx.exception))
            # No file should have been written for the oversized upload.
            self.assertFalse((Path(tmp) / "img.png").exists())

    def test_invalid_base64_rejected_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(Path(tmp), max_upload_size=100)
            # A stray non-alphabet character ('!') that validate=False would strip.
            data_url = "data:image/png;base64," + "aGVsbG8hZm9v!"
            with self.assertRaises(StudioPathError) as ctx:
                store.write_media_data_url("p", "img.png", data_url)
            self.assertIn("Invalid base64", str(ctx.exception))
            self.assertFalse((Path(tmp) / "img.png").exists())


if __name__ == "__main__":
    unittest.main()
