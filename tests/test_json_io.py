from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from feverslop.errors import FeverSlopDataError
from feverslop.utils.io import (
    read_json,
    read_json_document,
    read_json_or_none,
    write_json_document,
)


class JsonIoTests(unittest.TestCase):
    def test_document_reader_adds_path_context_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(FeverSlopDataError, re.escape(str(path))):
                read_json_document(path)

    def test_document_writer_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "document.json"
            self.assertEqual(path, write_json_document(path, {"ok": True}))
            self.assertEqual({"ok": True}, read_json_document(path))

    def test_read_json_reads_utf8_bom_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "document.json"
            path.write_text(json.dumps({"title": "Scene"}), encoding="utf-8-sig")

            self.assertEqual({"title": "Scene"}, read_json(path))

    def test_read_json_or_none_returns_none_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.json"

            self.assertIsNone(read_json_or_none(path))

    def test_read_json_or_none_rejects_empty_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.json"
            path.write_text("", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                read_json_or_none(path)


if __name__ == "__main__":
    unittest.main()
