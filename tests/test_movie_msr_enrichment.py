from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.application.movie_artifacts import (
    _read_json as _read_json_artifacts,
)
from feverslop.application.movie_ingredients_sheets import (
    _read_json as _read_json_ingredients,
)
from feverslop.application.movie_msr_enrichment import (
    _diegetic_audio_device,
    _movie_video_prompt,
    _read_json,
    _read_json as _read_json_msr,
)
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


class TestTOCTOUReplacement(unittest.TestCase):
    """APP-2003: All _read_json helpers raise FileNotFoundError so EAFP callers get empty dict."""

    def _eafp_read_default(read_json, path, default):
        """Helper that mirrors the EAFP pattern used in all three modules."""
        try:
            return read_json(path)
        except (FileNotFoundError, IsADirectoryError):
            return default

    def test_msr_read_json_eafp_missing_returns_empty(self):
        result = TestTOCTOUReplacement._eafp_read_default(
            _read_json_msr, Path("/nonexistent"), {},
        )
        self.assertEqual(result, {})

    def test_ingredients_read_json_eafp_missing_returns_empty(self):
        result = TestTOCTOUReplacement._eafp_read_default(
            _read_json_ingredients, Path("/nonexistent"), {},
        )
        self.assertEqual(result, {})

    def test_artifacts_read_json_eafp_missing_returns_empty(self):
        result = TestTOCTOUReplacement._eafp_read_default(
            _read_json_artifacts, Path("/nonexistent"), {},
        )
        self.assertEqual(result, {})

    def test_artifacts_read_json_eafp_missing_returns_custom_default(self):
        default = {"actors": [], "locations": []}
        result = TestTOCTOUReplacement._eafp_read_default(
            _read_json_artifacts, Path("/nonexistent"), default,
        )
        self.assertEqual(result, default)

    def test_msr_read_json_eafp_existing_returns_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"foo": "bar"}, f)
            f.flush()
            path = Path(f.name)
        try:
            result = TestTOCTOUReplacement._eafp_read_default(
                _read_json_msr, path, {},
            )
            self.assertEqual(result, {"foo": "bar"})
        finally:
            path.unlink()

    def test_read_json_eafp_corrupted_json_propagates(self):
        """Non-FileNotFoundError exceptions must still propagate."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            path = Path(f.name)
        try:
            with self.assertRaises(FeverSlopDataError):
                TestTOCTOUReplacement._eafp_read_default(_read_json_msr, path, {})
        finally:
            path.unlink()


class TestDiegeticAudioHeuristic(unittest.TestCase):
    """Issue #1190: diegetic path requires an explicit device noun; no hardcoded screaming."""

    MANIFEST = {"actors": [{"id": "actor-1", "name": "Mara"}]}

    def _prompt(self, dialogue: str, action: str = "") -> str:
        shot = {
            "description": "Close-up of Mara listening",
            "action": action,
            "dialogue": dialogue,
            "actor_ids": ["actor-1"],
        }
        return _movie_video_prompt(shot, bible={}, manifest=self.MANIFEST)

    def test_bare_voice_in_dialogue_is_not_diegetic(self):
        prompt = self._prompt('Mara: Your voice is familiar.')
        self.assertIn('Mara says: "Your voice is familiar."', prompt)
        self.assertNotIn("radio", prompt.casefold())
        self.assertNotIn("screaming", prompt.casefold())

    def test_parenthetical_voice_cue_is_not_diegetic(self):
        prompt = self._prompt('(V.O.) Your voice is familiar')
        self.assertNotIn("radio", prompt.casefold())
        self.assertNotIn("screaming", prompt.casefold())
        self.assertIn("is familiar", prompt)

    def test_device_noun_still_takes_diegetic_path(self):
        prompt = self._prompt('Mara: The radio crackles to life.')
        self.assertIn("radio", prompt.casefold())
        self.assertNotIn("screaming", prompt.casefold())

    def test_radio_direction_without_screaming(self):
        prompt = self._prompt('Mara: Get me out of here, the radio said.')
        self.assertIn("The radio plays a recording of Mara's voice", prompt)
        self.assertNotIn("screaming", prompt.casefold())
        self.assertNotIn("own voice", prompt.casefold())

    def test_radio_action_phrase_preserved_without_screaming(self):
        prompt = self._prompt(
            'Mara: Get me out of here, said the radio.',
            action="Mara holds the radio to her ear",
        )
        self.assertIn('Mara holds the radio to her ear: "Get me out of here, said the radio."', prompt)
        self.assertNotIn("screaming", prompt.casefold())

    def test_device_detection_unit(self):
        self.assertEqual(_diegetic_audio_device("V.O.", "Your voice is familiar"), "")
        self.assertEqual(_diegetic_audio_device("V.O.", "A distorted voice warns her"), "")
        self.assertEqual(_diegetic_audio_device("Radio", "Stay quiet"), "radio")
        self.assertEqual(_diegetic_audio_device("Transmitter", "Signal lost"), "radio")
        self.assertEqual(_diegetic_audio_device("Speaker", "Attention everyone"), "speaker")
        self.assertEqual(_diegetic_audio_device("V.O.", "The recording begins"), "radio")
