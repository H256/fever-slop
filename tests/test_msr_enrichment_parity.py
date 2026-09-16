"""M-21 parity tests: Song and Movie MSR enrichment share one validation core.

Both enrichment entrypoints (``enrich_render_plan_with_msr_prompts`` for the
song pipeline and ``enrich_movie_render_plan_with_msr_prompts`` for the movie
pipeline) now delegate their ``modules.vision()`` response parsing and relay
prompt validation to the single shared function
``feverslop.application.msr_validation.validate_msr_vision_response``.

These tests assert that the same vision response over the same reference set
yields the same validated relay prompt (and the same validated reference
description) in both paths, and that an invalid relay prompt is rejected in
both. This is the "Parity test Song<->Movie over the same input" acceptance
criterion for M-21.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.application.msr_prompt_enrichment import (
    enrich_render_plan_with_msr_prompts,
)
from feverslop.application.msr_validation import (
    validate_msr_vision_response,
)
from feverslop.application.movie_msr_enrichment import (
    enrich_movie_render_plan_with_msr_prompts,
)
from feverslop.prompting.msr_signatures import (
    MSRRelayPrompt,
    MSRPromptResult,
    MSRReferenceDescription,
)
from tests.fakellm import FakeLLM

_VALID_PROMPT = "Mara enters the archive and looks toward the desk."
_DESCRIPTION = "a weathered face with a scarred jaw"


def _shared_vision_response(*, prompt: str = _VALID_PROMPT) -> MSRPromptResult:
    return MSRPromptResult(
        references=[
            MSRReferenceDescription(id="mara", type="actor", description=_DESCRIPTION),
        ],
        relays=[MSRRelayPrompt(index=0, prompt=prompt)],
    )


class _ConfiguredLLM(FakeLLM):
    """A FakeLLM that satisfies the ``_dspy_modules`` capability check."""

    model = "fake-model"
    client = object()


class _FakeModules:
    """Duck-typed MSRPromptModules that returns a fixed vision response."""

    def __init__(self, response: MSRPromptResult):
        self._response = response

    def vision(self, payload, image_paths):
        return self._response

    def segments(self, payload):
        return MSRPromptResult(references=[], relays=[])


def _song_scene() -> dict:
    return {
        "scene": 1,
        "metadata": {"type": "instrumental"},
        "references": {
            "actor_reference_descriptions": [
                {"id": "mara", "name": "Mara", "visual_description": _DESCRIPTION},
            ],
            "actor_msr_paths": ["mara.png"],
        },
        "ltx": {
            "prompt_relay": [
                {"frame_start": 0, "frame_end": 24, "state": "instrumental"},
            ],
        },
    }


def _movie_project(project: Path) -> None:
    movie = project / "movie"
    refs = movie / "references"
    refs.mkdir(parents=True, exist_ok=True)
    (movie / "bible.json").write_text(json.dumps({}), encoding="utf-8")
    (movie / "render_plan.json").write_text(
        json.dumps({"shots": [
            {
                "scene": 1,
                "shot_id": "shot-1",
                "duration_seconds": 1.0,
                "type": "instrumental",
                "description": "Mara enters the archive.",
                "reference_ids": {"actors": ["mara"]},
            },
        ]}),
        encoding="utf-8",
    )
    (refs / "manifest.json").write_text(
        json.dumps({"actors": [
            {"id": "mara", "name": "Mara", "msr_sheet_path": "mara.png"},
        ], "locations": []}),
        encoding="utf-8",
    )
    (project / "mara.png").write_bytes(b"sheet")


class TestSongMovieParity(unittest.TestCase):
    """M-21 acceptance criterion: parity over the same vision input."""

    def _song_enrich(self, temp: Path, response: MSRPromptResult) -> str:
        plan = temp / "render_plan.json"
        plan.write_text(json.dumps([_song_scene()]), encoding="utf-8")
        (temp / "mara.png").write_bytes(b"sheet")
        with patch(
            "feverslop.application.msr_prompt_enrichment.MSRPromptModules",
            return_value=_FakeModules(response),
        ):
            out = enrich_render_plan_with_msr_prompts(plan, temp / "out.json", llm=_ConfiguredLLM("unused"))
        return json.loads(out.read_text(encoding="utf-8"))[0]["ltx"]["msr_prompt_relay"][0]["prompt"]

    def _movie_enrich(self, temp: Path, response: MSRPromptResult) -> str:
        _movie_project(temp)
        out = enrich_movie_render_plan_with_msr_prompts(
            project_dir=temp, llm=None, modules=_FakeModules(response),
        )
        return json.loads(out.read_text(encoding="utf-8"))["shots"][0]["ltx"]["msr_prompt_relay"][0]["prompt"]

    def test_same_vision_response_yields_same_relay_prompt(self):
        response = _shared_vision_response()
        with tempfile.TemporaryDirectory() as song_dir, tempfile.TemporaryDirectory() as movie_dir:
            song = self._song_enrich(Path(song_dir), response)
            movie = self._movie_enrich(Path(movie_dir), response)
            self.assertEqual(song, movie)
            self.assertEqual(song, _VALID_PROMPT)

    def test_invalid_relay_prompt_is_rejected_in_both_paths(self):
        bad = MSRPromptResult(
            references=[MSRReferenceDescription(id="mara", type="actor", description=_DESCRIPTION)],
            relays=[MSRRelayPrompt(index=0, prompt="Please preserve same subject and keep identity.")],
        )
        with tempfile.TemporaryDirectory() as song_dir, tempfile.TemporaryDirectory() as movie_dir:
            self.assertNotIn("preserve same subject", self._song_enrich(Path(song_dir), bad).lower())
            self.assertNotIn("preserve same subject", self._movie_enrich(Path(movie_dir), bad).lower())

    def test_shared_validator_accepts_and_rejects_consistently(self):
        relays = [{"frame_start": 0, "frame_end": 24, "state": "instrumental"}]
        pairs = {("mara", "actor")}
        ok = validate_msr_vision_response(
            _shared_vision_response().model_dump(), expected_pairs=pairs, relays=relays,
        )
        self.assertIsNotNone(ok)
        descriptions, prompts = ok
        self.assertEqual(prompts[0], _VALID_PROMPT)
        self.assertEqual(descriptions[("mara", "actor")], _DESCRIPTION)
        bad = validate_msr_vision_response(
            {"references": [{"id": "mara", "type": "actor", "description": _DESCRIPTION}],
             "relays": [{"index": 0, "prompt": "preserve same subject"}]},
            expected_pairs=pairs, relays=relays,
        )
        self.assertIsNone(bad)

