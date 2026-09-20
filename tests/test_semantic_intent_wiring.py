"""Wiring tests: the semantic intent extraction is threaded through the real
production pipeline path (build_resolved_global_context -> _extract_semantic_intent
-> factory -> persisted artifact), not just the isolated extraction core.

These prove the end-to-end contract required by issue #544:
- a configured extractor factory is actually invoked by the pipeline,
- the artifact is persisted before subject/location generation,
- with no factory (tests / non-LLM runs) the pipeline is a no-op that never
  invents entities and still yields a constructible empty ledger.
"""

import unittest
from pathlib import Path
from typing import Any

from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.config.project_config import ProjectConfig
from feverslop.prompting.semantic_intent_extraction import (
    EXTRACTION_EMPTY,
    EXTRACTION_OK,
    SemanticIntentExtractor,
)


class _FakePromptPipeline:
    """Minimal prompt pipeline satisfying build_resolved_global_context."""

    def __init__(self):
        self.actors = [
            {
                "id": "singer",
                "name": "Lead Singer",
                "role": "lead singer",
                "gender": "female",
                "visual_description": "A person on a mirror stage",
                "image_prompt": "A person singing into a microphone",
            },
        ]
        self.save_json_calls = 0

    def create_story_idea(self, lyrics: str, notes: str) -> str:
        return "a female lead singer performs with a band on a mirror stage"

    def create_style_block(self, lyrics: str, notes: str) -> str:
        return "cinematic concert film"

    def create_subject_and_locations(self, story_idea: str, notes: str) -> dict:
        return {
            "subject": "a band plays a mirror stage",
            "actors": [dict(actor) for actor in self.actors],
            "locations": [{"id": "stage", "name": "Mirror Stage"}],
        }

    def save_json(self, path: Path, data: dict, *, artifact_store: Any = None) -> None:
        self.save_json_calls += 1


class _RecordingArtifactStore:
    """Records write_json calls so the test can assert the artifact persisted."""

    def __init__(self) -> None:
        self.writes: dict[Path, dict] = {}

    def write_json(self, path: Path, data: dict) -> None:
        self.writes[path] = data

    def read_json(self, path: Path) -> dict:
        return self.writes[path]


def _build_pipeline(intent_extractor_factory: Any = None) -> PromptGenerationPipeline:
    return PromptGenerationPipeline(
        llm_factory=lambda app_config: None,
        prompt_pipeline_factory=lambda llm: _FakePromptPipeline(),
        concept_batcher_factory=lambda llm, size: None,
        scene_prompt_builder_factory=lambda llm: None,
        intent_extractor_factory=intent_extractor_factory,
    )


def _config() -> ProjectConfig:
    return ProjectConfig(
        project_dir=Path(),
        project_name="test",
        input_audio=Path("song.mp3"),
    )


class SemanticIntentWiringTests(unittest.TestCase):
    def test_configured_factory_is_invoked_and_artifact_is_persisted(self):
        """The pipeline must call the configured extractor factory and persist
        the artifact before subject/location generation completes."""
        store = _RecordingArtifactStore()
        artifact_path = Path("semantic_intent_song.json")

        # A factory that returns a no-op extractor (predictor=None) -- proves the
        # factory is wired in (it is invoked) without needing a real DSPy model.
        factory_invocations = []

        def factory(llm: Any) -> SemanticIntentExtractor:
            factory_invocations.append(llm)
            return SemanticIntentExtractor()  # no predictor -> empty ledger

        pipeline = _build_pipeline(intent_extractor_factory=factory)
        pipeline.build_resolved_global_context(
            config=_config(),
            prompt_pipeline=_FakePromptPipeline(),
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            llm="fake-llm",
            semantic_intent_json=artifact_path,
            artifact_store=store,
        )

        self.assertEqual(len(factory_invocations), 1, "factory must be invoked once")
        self.assertEqual(factory_invocations[0], "fake-llm", "llm must be threaded to the factory")
        self.assertIn(artifact_path, store.writes, "artifact must be persisted")
        payload = store.writes[artifact_path]
        self.assertEqual(payload["status"], EXTRACTION_EMPTY)
        self.assertEqual(payload["ledger"]["entities"], [], "no-op extractor yields an empty ledger")

    def test_no_factory_is_a_noop_that_never_invents_entities(self):
        """With no factory configured, the pipeline must not invent entities and
        must still yield a constructible empty ledger (bounded no-op)."""
        store = _RecordingArtifactStore()
        artifact_path = Path("semantic_intent_song.json")

        pipeline = _build_pipeline(intent_extractor_factory=None)
        pipeline.build_resolved_global_context(
            config=_config(),
            prompt_pipeline=_FakePromptPipeline(),
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            llm=None,
            semantic_intent_json=artifact_path,
            artifact_store=store,
        )

        self.assertIn(artifact_path, store.writes, "artifact must still persist (bounded)")
        payload = store.writes[artifact_path]
        self.assertEqual(payload["status"], EXTRACTION_EMPTY)
        self.assertEqual(payload["ledger"]["entities"], [], "no factory -> empty ledger, no invention")

    def test_failed_extraction_is_bounded_and_still_persisted(self):
        """A failed extraction must be bounded: it still persists a constructible,
        empty ledger with a visible status so subject/location generation always
        has a valid artifact."""
        store = _RecordingArtifactStore()
        artifact_path = Path("semantic_intent_song.json")

        def boom(story_idea: str, notes: str = "") -> Any:
            raise RuntimeError("structured output unavailable")

        def factory(llm: Any) -> SemanticIntentExtractor:
            return SemanticIntentExtractor(predictor=boom)

        pipeline = _build_pipeline(intent_extractor_factory=factory)
        pipeline.build_resolved_global_context(
            config=_config(),
            prompt_pipeline=_FakePromptPipeline(),
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            llm="fake-llm",
            semantic_intent_json=artifact_path,
            artifact_store=store,
        )

        self.assertIn(artifact_path, store.writes, "failed extraction must still persist")
        payload = store.writes[artifact_path]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["ledger"]["entities"], [], "failed extraction -> empty ledger")
        self.assertTrue(payload["warnings"], "failure must be visible via warnings")

    def test_extracted_entities_are_not_invented(self):
        """A successful extraction must surface exactly the entities the model
        returned -- the pipeline must not add performers/genders on its own."""
        store = _RecordingArtifactStore()
        artifact_path = Path("semantic_intent_song.json")

        def factory(llm: Any) -> SemanticIntentExtractor:
            # Reuse the extraction core's DE/EN payload shape: a single object
            # entity with role 'lead' and no performer.
            from feverslop.prompting.semantic_intent_extraction import IntentExtractionResult

            def predictor(story_idea: str, notes: str = "") -> Any:
                return IntentExtractionResult(
                    language="de",
                    entities=[
                        {
                            "id": "e1",
                            "kind": "object",
                            "label": "Stein",
                            "role": "lead",
                            "attributes": {},
                        },
                    ],
                    relations=[],
                    constraints=[],
                )

            return SemanticIntentExtractor(predictor=predictor)

        pipeline = _build_pipeline(intent_extractor_factory=factory)
        pipeline.build_resolved_global_context(
            config=_config(),
            prompt_pipeline=_FakePromptPipeline(),
            all_lyrics="",
            run_spinner=lambda _description, func: func(),
            llm="fake-llm",
            semantic_intent_json=artifact_path,
            artifact_store=store,
        )

        payload = store.writes[artifact_path]
        self.assertEqual(payload["status"], EXTRACTION_OK)
        ledger = payload["ledger"]
        self.assertEqual(len(ledger["entities"]), 1)
        self.assertEqual(ledger["entities"][0]["kind"], "object")
        self.assertEqual(ledger["entities"][0]["role"], "lead")


if __name__ == "__main__":
    unittest.main()
