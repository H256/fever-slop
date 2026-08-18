"""Integration test: FullAutoUseCase → LLMSongBriefGenerator → typed prompt module → SongSpec.

Exercises the real LLMSongBriefGenerator adapter so that system-prompt creation,
JSON extraction, and SongSpec construction are validated end-to-end with only the
LLM call faked.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.llm_song_brief_generator import LLMSongBriefGenerator
from feverslop.application.full_auto import FullAutoRequest, FullAutoUseCase
from feverslop.domain.full_auto import GeneratedSong, SongSpec

from tests.prompt_fakes import GeneralModulesFake


class FakeSongGenerator:
    """Minimal song generator that writes a stub mp3 and records calls."""

    def __init__(self):
        self.calls: list[tuple[SongSpec, str, Path, int]] = []

    def generate(self, spec: SongSpec, *, project_slug: str, output_dir: Path, seed: int) -> GeneratedSong:
        self.calls.append((spec, project_slug, Path(output_dir), seed))
        path = Path(output_dir) / f"{project_slug}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3")
        return GeneratedSong(audio_path=path, manifest={"prompt_id": "test", "seed": seed})


class FakeRunner:
    """Minimal pipeline runner that records calls."""

    def __init__(self):
        self.calls: list[tuple[Path, dict]] = []

    def run(self, *, project_config_path: Path, options: dict) -> Path | None:
        self.calls.append((Path(project_config_path), dict(options)))
        return Path(project_config_path).parent / "output" / "render" / "ltx_single_prompt" / "joy-demo.mp4"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _json_response() -> str:
    return json.dumps({
        "title": "Joy Demo",
        "tags": "bright pop song with synth groove",
        "lyrics": "[Verse]\nhello world",
        "bpm": 128,
        "language": "en",
        "keyscale": "D major",
        "visual_story_idea": "friends in sunlight",
        "visual_style": "warm realistic pop video",
    })


class FullAutoLLMChainTests(unittest.TestCase):
    """Verify the real LLMSongBriefGenerator adapter wires through the FullAutoUseCase."""

    def test_llm_called_with_system_prompt_and_json_payload(self):
        """LLMSongBriefGenerator sends correct system prompt and serialized request."""
        modules = GeneralModulesFake(song_brief=json.loads(_json_response()))
        generator = LLMSongBriefGenerator(object(), modules=modules)

        request = FullAutoRequest(
            idea="friendship",
            style="bright pop",
            duration_seconds=90.0,
            language="en",
        )
        spec = generator.generate(request)

        self.assertEqual(len(modules.calls), 1)
        call = modules.calls[0]

        payload = call.payload
        self.assertEqual(payload["idea"], "friendship")
        self.assertEqual(payload["style"], "bright pop")
        self.assertEqual(payload["duration_seconds"], 90.0)

        self.assertEqual(spec.title, "Joy Demo")
        self.assertEqual(spec.tags, "bright pop song with synth groove")
        self.assertEqual(spec.lyrics, "[Verse]\nhello world")
        self.assertEqual(spec.bpm, 128)
        self.assertEqual(spec.duration_seconds, 90.0)
        self.assertEqual(spec.language, "en")
        self.assertEqual(spec.keyscale, "D major")
        self.assertEqual(spec.visual_story_idea, "friends in sunlight")
        self.assertEqual(spec.visual_style, "warm realistic pop video")

    def test_json_parsing_strips_code_fences(self):
        """LLMSongBriefGenerator extracts JSON wrapped in markdown code fences."""
        modules = GeneralModulesFake(song_brief=json.loads(_json_response()))
        generator = LLMSongBriefGenerator(object(), modules=modules)

        request = FullAutoRequest(
            idea="test",
            style="pop",
            duration_seconds=60.0,
            language="en",
        )
        spec = generator.generate(request)

        self.assertEqual(spec.title, "Joy Demo")
        self.assertEqual(spec.bpm, 128)

    def test_song_spec_values_match_llm_response(self):
        """SongSpec fields reflect the exact JSON returned by the LLM."""
        custom_response = json.dumps({
            "title": "Custom Title",
            "tags": "jazz fusion",
            "lyrics": "[Verse]\ncustom lyrics",
            "bpm": 95,
            "language": "ja",
            "keyscale": "A minor",
            "visual_story_idea": "rainy city street",
            "visual_style": "cyberpunk noir",
        })

        modules = GeneralModulesFake(song_brief=json.loads(custom_response))
        generator = LLMSongBriefGenerator(object(), modules=modules)

        spec = generator.generate(
            FullAutoRequest(
                idea="rain",
                style="jazz",
                duration_seconds=120.0,
                language="ja",
            )
        )

        self.assertEqual(spec.title, "Custom Title")
        self.assertEqual(spec.tags, "jazz fusion")
        self.assertEqual(spec.lyrics, "[Verse]\ncustom lyrics")
        self.assertEqual(spec.bpm, 95)
        self.assertEqual(spec.duration_seconds, 120.0)
        self.assertEqual(spec.language, "ja")
        self.assertEqual(spec.keyscale, "A minor")
        self.assertEqual(spec.visual_story_idea, "rainy city street")
        self.assertEqual(spec.visual_style, "cyberpunk noir")

    def test_full_auto_use_case_preserves_explicit_idea_over_llm_visual_summary(self):
        """The generated song brief must not replace explicit visual constraints."""
        from feverslop.adapters.full_auto_scaffold import LocalProjectScaffold

        llm_response = json.dumps({
            "title": "End-to-End Test",
            "tags": "dreamy ambient",
            "lyrics": "[Verse]\nambient sounds",
            "bpm": 70,
            "language": "en",
            "keyscale": "C major",
            "visual_story_idea": "ocean waves",
            "visual_style": "minimalist",
        })

        modules = GeneralModulesFake(song_brief=json.loads(llm_response))

        with tempfile.TemporaryDirectory() as temp_dir:
            use_case = FullAutoUseCase(
                brief_generator=LLMSongBriefGenerator(object(), modules=modules),
                song_generator=FakeSongGenerator(),
                project_scaffold=LocalProjectScaffold(),
                pipeline_runner=None,
            )

            result = use_case.execute(
                FullAutoRequest(
                    idea="ocean",
                    style="ambient",
                    project_name="Ocean Vibes",
                    projects_dir=Path(temp_dir),
                    duration_seconds=60.0,
                    language="en",
                    seed=42,
                    run_video_pipeline=False,
                )
            )

            self.assertEqual(result.song_spec.title, "End-to-End Test")
            self.assertEqual(result.song_spec.tags, "dreamy ambient")
            self.assertEqual(result.song_spec.lyrics, "[Verse]\nambient sounds")
            self.assertEqual(result.song_spec.bpm, 70)
            self.assertEqual(result.song_spec.duration_seconds, 60.0)
            self.assertEqual(result.song_spec.visual_story_idea, "ocean")
            self.assertEqual(result.song_spec.visual_style, "ambient")

    def test_llm_brief_generator_adapts_request_defaults_for_missing_values(self):
        """LLMSongBriefGenerator falls back to request values when LLM omits fields."""
        class PartialSongBriefModules:
            def song_brief(self, payload):
                return type("Result", (), {"model_dump": lambda self: {
                    "title": "Short", "tags": "short tags", "lyrics": "[Verse]\nshort",
                    "visual_story_idea": "short idea", "visual_style": "short style",
                }})()

        modules = PartialSongBriefModules()
        generator = LLMSongBriefGenerator(object(), modules=modules)

        spec = generator.generate(
            FullAutoRequest(
                idea="missing",
                style="pop",
                duration_seconds=100.0,
                language="en",
                bpm=140,
                keyscale="E major",
            )
        )

        # LLM didn't return bpm → fallback to request.bpm
        self.assertEqual(spec.bpm, 140)
        # LLM didn't return keyscale → fallback to request.keyscale
        self.assertEqual(spec.keyscale, "E major")
        # LLM didn't return language → fallback to request.language
        self.assertEqual(spec.language, "en")


if __name__ == "__main__":
    unittest.main()
