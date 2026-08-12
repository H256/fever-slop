from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.application.movie_msr_enrichment import _read_json
from feverslop.errors import FeverSlopDataError


class TestReadJsonErrorHandling(unittest.TestCase):
    """APP-2006: _read_json raises proper exceptions for all failure modes."""

    def test_missing_file_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            _read_json(Path("/nonexistent/path/to/artifact.json"))

    def test_valid_dict_json_returns_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            path = Path(f.name)
        try:
            result = _read_json(path)
            self.assertEqual({"key": "value"}, result)
        finally:
            path.unlink()

    def test_corrupted_json_raises_fever_slop_data_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{this is not valid json {{{")
            f.flush()
            path = Path(f.name)
        try:
            with self.assertRaises(FeverSlopDataError) as ctx:
                _read_json(path)
            self.assertIn(str(path), str(ctx.exception))
        finally:
            path.unlink()

    def test_empty_file_raises_fever_slop_data_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)
        try:
            with self.assertRaises(FeverSlopDataError) as ctx:
                _read_json(path)
            self.assertIn(str(path), str(ctx.exception))
        finally:
            path.unlink()

    def test_non_dict_json_raises_value_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["a", "b"], f)
            f.flush()
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                _read_json(path)
        finally:
            path.unlink()

    def test_data_error_preserves_cause(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("broken")
            f.flush()
            path = Path(f.name)
        try:
            with self.assertRaises(FeverSlopDataError) as ctx:
                _read_json(path)
            self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)
        finally:
            path.unlink()
