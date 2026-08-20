"""Tests for atomic write utilities in feverslop.utils.io."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.utils.io import atomic_write_bytes, atomic_write_json, atomic_write_text, file_is_valid


class AtomicWriteJsonTests(unittest.TestCase):

    def test_writes_valid_json_and_no_tmp_left(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            atomic_write_json(path, {"key": "value"})
            self.assertTrue(path.is_file())
            self.assertEqual([], list(Path(d).glob("*.tmp")))
            self.assertEqual({"key": "value"}, json.loads(path.read_text()))

    def test_overwrites_existing_file_safely(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            path.write_text('{"old": true}')
            atomic_write_json(path, {"new": True})
            self.assertEqual({"new": True}, json.loads(path.read_text()))

    def test_trailing_newline_on_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            atomic_write_json(path, {})
            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "deep" / "data.json"
            atomic_write_json(path, {"ok": True})
            self.assertTrue(path.is_file())

    def test_no_tmp_artifacts_after_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            atomic_write_json(path, {"before": True})
            atomic_write_json(path, {"after": True})
            self.assertEqual(0, len(list(Path(d).glob("*.tmp"))))


class AtomicWriteTextTests(unittest.TestCase):

    def test_writes_exact_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "screenplay.md"
            text = "# Title\n\nSome content"
            atomic_write_text(path, text)
            self.assertEqual(text, path.read_text())

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "file.txt"
            path.write_text("old")
            atomic_write_text(path, "new")
            self.assertEqual("new", path.read_text())

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "deep" / "file.txt"
            atomic_write_text(path, "content")
            self.assertEqual("content", path.read_text())


class AtomicWriteBytesTests(unittest.TestCase):

    def test_writes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "binary.png"
            payload = b"\x89PNG\r\n\x1a\n\x00\x01binary\xff"
            atomic_write_bytes(path, payload)
            self.assertEqual(payload, path.read_bytes())
            self.assertEqual([], list(Path(d).glob("*.tmp")))

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "file.bin"
            path.write_bytes(b"old")
            atomic_write_bytes(path, b"new")
            self.assertEqual(b"new", path.read_bytes())

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a" / "b" / "img.png"
            atomic_write_bytes(path, b"png-bytes")
            self.assertEqual(b"png-bytes", path.read_bytes())

    def test_no_tmp_artifacts_after_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.bin"
            atomic_write_bytes(path, b"one")
            atomic_write_bytes(path, b"two")
            self.assertEqual([], list(Path(d).glob("*.tmp")))
            self.assertEqual([], list(Path(d).glob("*.bin.tmp")))


class FileIsValidTests(unittest.TestCase):

    def test_missing_file_is_invalid(self):
        self.assertFalse(file_is_valid(Path("/nonexistent_path_that_does_not_exist")))

    def test_zero_byte_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.txt"
            path.write_bytes(b"")
            self.assertFalse(file_is_valid(path))

    def test_nonzero_file_is_valid(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "full.json"
            path.write_text("{}")
            self.assertTrue(file_is_valid(path))

    def test_directory_is_not_valid(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(file_is_valid(Path(d)))


class MovieArtifactWriterAtomicTests(unittest.TestCase):

    def test_write_json_is_atomic(self):
        from feverslop.adapters.movie_artifact_writer import LocalMovieArtifactWriter
        writer = LocalMovieArtifactWriter()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.json"
            writer.write_json(path, {"data": True})
            self.assertEqual({"data": True}, json.loads(path.read_text()))
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_write_text_is_atomic(self):
        from feverslop.adapters.movie_artifact_writer import LocalMovieArtifactWriter
        writer = LocalMovieArtifactWriter()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.txt"
            writer.write_text(path, "hello")
            self.assertEqual("hello", path.read_text())
