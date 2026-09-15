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
    enrich_movie_render_plan_with_msr_prompts,
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


class _FakeMSRModules:
    """Duck-typed stand-in for MSRPromptModules.vision that records payloads."""

    def __init__(self, result):
        self._result = result
        self.payloads = []
        self.images = None

    def vision(self, payload, images):
        self.payloads.append(payload)
        self.images = list(images)
        return self._result


class TestMovieMSRFrameEndExclusive(unittest.TestCase):
    """Issue #1182: movie-MSR relay frame_end must be exclusive (== frame_count).

    Every producer/consumer treats frame_end as exclusive (render_plan_builder:541,
    movie_ingredients_sheets:199, and the MSR/LTX relay builders that compute
    length = end - start). movie_msr_enrichment previously wrote frame_count - 1,
    which dropped the last frame into gap padding and under-told the vision module
    by one frame.
    """

    BIBLE = {"runtime_constraints": {"fps": 24, "dialogue_language": "English"}}

    def _write_project(self, project_dir, manifest, shots, actor_sheet=None):
        movie = project_dir / "movie"
        refs = movie / "references"
        refs.mkdir(parents=True, exist_ok=True)
        (movie / "bible.json").write_text(json.dumps(self.BIBLE), encoding="utf-8")
        (movie / "render_plan.json").write_text(json.dumps({"shots": shots}), encoding="utf-8")
        (refs / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # Reference sheet paths are resolved relative to the project root
        # (see _movie_reference_images), so the file lives at project/actor-1.png.
        if actor_sheet is not None:
            (project_dir / actor_sheet).write_bytes(b"sheet")
        return movie

    def test_written_relay_frame_end_equals_frame_count(self):
        """The persisted msr_prompt_relay must end at frame_count (exclusive)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            manifest = {"actors": [], "locations": []}
            shots = [
                {
                    "scene": 1,
                    "shot_id": "shot-1",
                    "duration_seconds": 1.0,
                    "type": "instrumental",
                    "description": "A wide establishing shot of the harbor.",
                    "camera": "slow push in",
                    "actor_ids": [],
                },
            ]
            self._write_project(project, manifest, shots)

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project, llm=None)
            enriched = json.loads(output.read_text(encoding="utf-8"))

            relay = enriched["shots"][0]["ltx"]["msr_prompt_relay"][0]
            # duration_seconds * fps == 24 frames; frame_end must be exclusive.
            self.assertEqual(relay["frame_start"], 0)
            self.assertEqual(relay["frame_end"], 24)
            self.assertNotEqual(relay["frame_end"], 23)

            # Acceptance criterion: the shot's last frame must be covered by the
            # relay assignment under the exclusive-end convention. The last frame
            # index is frame_count - 1 == 23; with the old inclusive value
            # (frame_end == 23) it fell outside [frame_start, frame_end).
            frame_count = 24
            last_frame_index = frame_count - 1
            self.assertTrue(
                relay["frame_start"] <= last_frame_index < relay["frame_end"],
                f"last frame {last_frame_index} not covered by relay "
                f"[{relay['frame_start']}, {relay['frame_end']})",
            )

    def test_vision_payload_frame_end_equals_frame_count(self):
        """The relay frame_end handed to the vision module must be exclusive too."""
        from feverslop.prompting.msr_signatures import (
            MSRReferenceDescription,
            MSRPromptResult,
            MSRRelayPrompt,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            manifest = {
                "actors": [
                    {
                        "id": "actor-1",
                        "name": "Mara",
                        "visual_description": "a weathered face with a scarred jaw",
                        "msr_sheet_path": "actor-1.png",
                    }
                ],
                "locations": [],
            }
            shots = [
                {
                    "scene": 1,
                    "shot_id": "shot-1",
                    "duration_seconds": 1.0,
                    "type": "instrumental",
                    "description": "Mara walks along the pier.",
                    "camera": "tracking shot",
                    "reference_ids": {"actors": ["actor-1"]},
                },
            ]
            self._write_project(project, manifest, shots, actor_sheet="actor-1.png")

            result = MSRPromptResult(
                references=[
                    MSRReferenceDescription(id="actor-1", type="actor", description="a weathered face with a scarred jaw"),
                ],
                relays=[
                    MSRRelayPrompt(index=0, prompt="Mara walks along the pier; the camera tracks beside her while gulls wheel overhead."),
                ],
            )
            modules = _FakeMSRModules(result)

            output = enrich_movie_render_plan_with_msr_prompts(
                project_dir=project, llm=None, modules=modules,
            )

            self.assertEqual(len(modules.payloads), 1)
            vision_relay = modules.payloads[0]["relay_segments"][0]
            self.assertEqual(vision_relay["frame_start"], 0)
            self.assertEqual(vision_relay["frame_end"], 24)

            # The vision path also persisted the relay with the exclusive end.
            enriched = json.loads(output.read_text(encoding="utf-8"))
            written_relay = enriched["shots"][0]["ltx"]["msr_prompt_relay"][0]
            self.assertEqual(written_relay["frame_end"], 24)
            self.assertEqual(written_relay["prompt"], result.relays[0].prompt)
