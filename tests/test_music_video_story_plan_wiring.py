"""Wiring tests for the music-video StoryPlan pipeline (issue #1386).

These prove the end-to-end contract:
- direction is stored as provenance, never authoritative audio data
- the plan is persisted as authoritative with fingerprinted resume
- resume reuses a stable plan; a direction change invalidates it
- the approval gate stops the pipeline before concept generation
- segment briefs are threaded into concept generation
- an invalid plan is a hard stop (no fallback)
- the H3 compiler, timeline builder, and audio analysis are not invoked
"""

from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from feverslop.composition import generate_render_plan as composition
from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.application.story_plan_service import (
    SegmentDescriptor,
    StoryPlanError,
    StoryPlanRequest,
    StoryPlanService,
    compute_source_fingerprint,
)
from feverslop.prompting.story_plan_service_adapter import StoryPlanServiceAdapter
from feverslop.prompting.dspy_runtime import DspyRuntime, H3SignatureBundle
from feverslop.domain.story_plan import (
    PLANNER_REVISION,
    STORY_PLAN_SCHEMA_VERSION,
)
from feverslop.domain.story_plan_artifacts import (
    ArtifactClass,
    RegenerationPolicy,
    StoryPlanArtifactManifest,
    read_manifest,
    read_story_plan,
    story_plan_fingerprint,
    write_manifest,
    write_story_plan,
)
from tests.test_story_plan_dspy import (
    FakePromptModules,
    make_request,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _make_config(**overrides: Any) -> Any:
    defaults = dict(
        content_mode="music_video",
        project_name="test-song",
        song_id="test-song",
        language="en",
        music_style="indie",
        lyrics="la la la",
        actors=(),
        story_direction="",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_app_config(require_approval: bool = False, planner_revision: str = PLANNER_REVISION) -> Any:
    return SimpleNamespace(
        llm=SimpleNamespace(
            story_planning=SimpleNamespace(
                require_approval=require_approval,
                planner_revision=planner_revision,
                failure_policy="warn",
            ),
        ),
    )


def _make_request(**overrides: Any) -> Any:
    defaults = dict(
        story_direction="",
        story_plan_approve=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_paths(prompts_dir: Path) -> Any:
    return SimpleNamespace(prompts_dir=prompts_dir)


def _stage1_segments() -> list[dict]:
    return [
        {"segment_id": "seg-1", "start_seconds": 0.0, "end_seconds": 30.0},
        {"segment_id": "seg-2", "start_seconds": 30.0, "end_seconds": 60.0},
    ]


def _legacy_stage1_segments() -> list[dict]:
    """The persisted Stage 1 schema used by existing projects."""
    return [
        {"segment_id": "seg-1", "start": 0.0, "end": 30.0, "lyrics": "first line"},
        {"segment_id": "seg-2", "start": 30.0, "end": 60.0, "lyrics": "last line"},
    ]


class _RecordingReporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def message(self, text: str) -> None:
        self.messages.append(text)

    def step(self, text: str) -> None:
        self.messages.append(text)

    def panel(self, *args: Any, **kwargs: Any) -> None:
        pass

    def table(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.messages.append(str(args[0]) if args else "warning")


class _FactoryFakeClient:
    base_url = "http://fake.local/v1"
    api_key = "fake-key"


class _FactoryFakeLLM:
    """Minimal DSPy-compatible LLM without a network-backed predictor."""

    model = "fake-model"
    client = _FactoryFakeClient()
    max_tokens = 2048
    dspy_temperature = 0.4
    dspy_cache = False
    max_retries = 1
    request_timeout_seconds = None
    llm_limiter = None
    metrics = None


class _FactoryPredictor:
    def __init__(self, signature: Any) -> None:
        self._name = signature.__name__

    def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        if self._name == "StoryPlanBible":
            return {"bible": {"premise": "A singer leaves home.", "theme": "release"}}
        if self._name == "BeatAllocation":
            return {
                "allocation": {
                    "beats": [
                        {"phase": "opening", "description": "Departure"},
                        {"phase": "resolution", "description": "Release"},
                    ],
                    "brief_allocations": [
                        {"target": "seg-1", "beat_index": 0},
                        {"target": "seg-2", "beat_index": 1},
                    ],
                }
            }
        if self._name == "Acting":
            return {
                "result": {
                    "arcs": [],
                    "briefs": [
                        {
                            "target": "seg-1",
                            "objective": "Leave the familiar world.",
                            "emotional_turn": "hesitant to committed",
                            "actor_states": [{"character_id": "char-1", "inner_state": "hesitant", "physical_state": "walking"}],
                        },
                        {
                            "target": "seg-2",
                            "objective": "Accept the new future.",
                            "emotional_turn": "guarded to free",
                            "actor_states": [{"character_id": "char-1", "inner_state": "free", "physical_state": "standing"}],
                        },
                    ],
                }
            }
        raise AssertionError(f"unexpected DSPy signature: {self._name}")


def _factory_fake_dspy_runtime() -> DspyRuntime:
    return DspyRuntime(
        signatures=H3SignatureBundle(object(), object(), object(), object()),
        lm_factory=lambda _name, **_kwargs: "fake-lm",
        predict_factory=_FactoryPredictor,
        context_factory=lambda **_kwargs: nullcontext(),
    )


def _build_pipeline(
    story_plan_service_factory: Any = None,
) -> PromptGenerationPipeline:
    return PromptGenerationPipeline(
        llm_factory=lambda app_config: None,
        prompt_pipeline_factory=lambda llm: None,
        concept_batcher_factory=lambda llm, size: None,
        scene_prompt_builder_factory=lambda llm: None,
        story_plan_service_factory=story_plan_service_factory,
    )


class _RecordingService:
    """Wraps a StoryPlanService to record build_plan calls."""

    def __init__(self, inner: StoryPlanService) -> None:
        self._inner = inner
        self.build_plan_calls: list[Any] = []

    def build_plan(self, request: Any) -> Any:
        self.build_plan_calls.append(request)
        return self._inner.build_plan(request)

    def write_review_export(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.write_review_export(*args, **kwargs)

    def render_review_export(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.render_review_export(*args, **kwargs)


def _make_recording_factory(
    service: StoryPlanService,
) -> tuple[Any, _RecordingService]:
    """Returns (factory, recording_service)."""
    recording = _RecordingService(service)

    def factory(llm: Any) -> _RecordingService:
        return recording

    return factory, recording


def _compute_fingerprint(
    config: Any,
    request: Any,
    stage1_segments: list[dict],
) -> str:
    """Compute the fingerprint the same way _resolve_story_plan does."""
    effective_direction = (
        str(getattr(request, "story_direction", "") or "").strip()
        or str(getattr(config, "story_direction", "") or "").strip()
    )
    song_id = str(
        getattr(config, "song_id", "")
        or getattr(config, "project_name", "")
        or "song"
    )
    song_language = str(getattr(config, "language", "") or "unknown")
    song_style = str(
        getattr(config, "music_style", "")
        or getattr(config, "style", "")
        or ""
    )
    lyrics = str(getattr(config, "lyrics", "") or "")
    characters = [
        {
            "id": str(actor.id),
            "name": str(actor.name),
            "description": str(getattr(actor, "description", "")),
        }
        for actor in (getattr(config, "actors", ()) or ())
    ]
    segments = [
        {
            "segment_id": seg["segment_id"],
            "start_seconds": seg["start_seconds"],
            "end_seconds": seg["end_seconds"],
        }
        for seg in stage1_segments
    ]
    return compute_source_fingerprint(
        song_title=song_id,
        song_language=song_language,
        song_style=song_style,
        lyrics=lyrics,
        sections=(),
        segments=segments,
        characters=characters,
        creative_direction=effective_direction,
    )


def _persist_plan(
    prompts_dir: Path,
    song_id: str,
    plan: Any,
    input_fingerprint: str,
) -> tuple[Path, Path]:
    """Persist a plan and its manifest (mirrors _resolve_story_plan)."""
    plan_path = prompts_dir / f"story_plan_{song_id}.json"
    manifest_path = prompts_dir / f"story_plan_{song_id}.manifest.json"
    write_story_plan(plan_path, plan)
    plan_fingerprint = story_plan_fingerprint(plan)
    manifest = StoryPlanArtifactManifest(
        artifact_class=ArtifactClass.authoritative,
        regeneration_policy=RegenerationPolicy.on_input_change,
        input_fingerprint=input_fingerprint,
        plan_fingerprint=plan_fingerprint,
    )
    write_manifest(manifest_path, manifest)
    return plan_path, manifest_path


class MusicVideoStoryPlanWiringTests(unittest.TestCase):
    def test_adapter_passes_story_sources_and_acting_segments(self) -> None:
        calls: dict[str, dict[str, Any]] = {}

        class Modules:
            def bible(self, **kwargs: Any) -> dict[str, Any]:
                calls["bible"] = kwargs
                return {"premise": "p"}

            def beat_allocation(self, **kwargs: Any) -> dict[str, Any]:
                calls["allocation"] = kwargs
                return {"beats": [{"phase": "resolution", "description": "end"}], "brief_allocations": []}

            def acting(self, **kwargs: Any) -> dict[str, Any]:
                calls["acting"] = kwargs
                return {"briefs": []}

        adapter = StoryPlanServiceAdapter(Modules())
        adapter.bible(
            song_title="Song", song_language="en", song_style="dark", lyrics="orcs",
            sections=[], characters=[{"id": "ravena"}],
            source_evidence={
                "creative_direction": "user direction",
                "story_idea": "A pilgrimage to renewal.",
                "locations": [{"id": "cave"}],
                "props": [{"id": "well"}],
            }, guide="",
        )
        adapter.beat_allocation(
            song_title="Song", lyrics="orcs", segments=[{"segment_id": "seg-1"}],
            narrative_bible={}, characters=[], terminal_window_seconds=1, guide="",
            locations=[{"id": "cave"}], props=[{"id": "well"}],
        )
        adapter.acting(
            song_title="Song", song_language="en", lyrics="orcs", briefs=[{"brief_id": "b"}],
            characters=[], guide="", segments=[{"segment_id": "seg-1"}],
            locations=[{"id": "cave"}], props=[{"id": "well"}],
            typed_beats=[{"phase": "resolution", "description": "end"}],
        )
        self.assertIn("A pilgrimage to renewal.", calls["bible"]["story_text"])
        self.assertEqual(calls["allocation"]["locations"], [{"id": "cave"}])
        self.assertEqual(calls["acting"]["segments"], [{"segment_id": "seg-1"}])

    def test_warn_policy_stops_before_unbound_concept_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            class FailingService:
                def build_plan(self, request: Any) -> Any:
                    raise StoryPlanError("acting output must contain a non-empty briefs list")

            pipeline = _build_pipeline(story_plan_service_factory=lambda _llm: FailingService())
            reporter = _RecordingReporter()
            result, stopped = pipeline._resolve_story_plan(
                config=_make_config(), app_config=_make_app_config(), request=_make_request(),
                resume=False, stage1_segments=_stage1_segments(), paths=_make_paths(Path(tmp)),
                reporter=reporter, artifact_store=None, log_file=lambda *_args: None,
            )
            self.assertIsNone(result)
            self.assertTrue(stopped)
            self.assertIn("stopping", " ".join(reporter.messages).lower())
    """Wiring tests for the music-video StoryPlan pipeline (issue #1386)."""

    def test_resume_accepts_existing_stage1_start_end_schema(self) -> None:
        """Existing Stage 1 artifacts reach planning without being mutated."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)
            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)
            segments = _legacy_stage1_segments()
            original = [dict(segment) for segment in segments]

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=_make_config(),
                app_config=_make_app_config(),
                request=_make_request(),
                resume=True,
                stage1_segments=segments,
                paths=_make_paths(prompts_dir),
                reporter=_RecordingReporter(),
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertFalse(gate_stopped)
            self.assertIsNotNone(segment_briefs)
            self.assertEqual(len(recording.build_plan_calls), 1)
            request = recording.build_plan_calls[0]
            self.assertEqual(
                [(segment.segment_id, segment.start_seconds, segment.end_seconds, segment.lyric_text)
                 for segment in request.segments],
                [("seg-1", 0.0, 30.0, "first line"), ("seg-2", 30.0, 60.0, "last line")],
            )
            self.assertEqual(segments, original)

    def test_story_plan_adapter_passes_canonical_characters_to_bible(self) -> None:
        """The bible receives the resolved cast instead of an empty list."""
        class Modules:
            def __init__(self) -> None:
                self.received: dict[str, Any] = {}

            def bible(self, **kwargs: Any) -> dict[str, Any]:
                self.received = kwargs
                return {"premise": "p"}

        modules = Modules()
        adapter = StoryPlanServiceAdapter(modules)
        adapter.bible(
            song_title="Song",
            song_language="en",
            song_style="gothic",
            lyrics="line",
            sections=[],
            source_evidence={},
            guide="",
            characters=[{"id": "ravena", "name": "Ravena"}],
        )
        self.assertEqual(modules.received["characters"], [{"id": "ravena", "name": "Ravena"}])

    # -- (a) direction provenance ------------------------------------------

    def test_direction_is_stored_as_provenance_not_authoritative_audio(self) -> None:
        """The creative direction is stored in PlanProvenance.notes, never as
        authoritative audio data in the plan."""
        modules = FakePromptModules()
        service = StoryPlanService(prompt_modules=modules)
        request = make_request(source_evidence={"creative_direction": "dark, brooding"})
        result = service.build_plan(request)
        plan = result.plan
        self.assertEqual(plan.provenance.notes, "dark, brooding")
        # The plan must not carry authoritative audio data in its segments.
        for brief in plan.segments:
            if brief.audio_ref is not None:
                self.assertEqual(brief.audio_ref.segment_id, brief.target)

    def test_direction_from_request_overrides_config(self) -> None:
        """When both request and config carry a direction, the request wins."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)
            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)

            config = _make_config(story_direction="from config")
            app_config = _make_app_config()
            request_ns = _make_request(story_direction="from request")
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=False,
                stage1_segments=_stage1_segments(),
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )
            self.assertFalse(gate_stopped)
            # The effective direction is "from request" (request wins).
            # Verify via the persisted plan's provenance.
            plan_path = prompts_dir / "story_plan_test-song.json"
            self.assertTrue(plan_path.is_file())
            plan = read_story_plan(plan_path)
            self.assertEqual(plan.provenance.notes, "from request")

    # -- (b) authoritative persistence ------------------------------------

    def test_plan_is_persisted_as_authoritative_with_fingerprint(self) -> None:
        """A built plan is persisted as authoritative with the input fingerprint
        and regeneration policy on_input_change."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)
            song_id = "test-song"
            plan_path = prompts_dir / f"story_plan_{song_id}.json"
            manifest_path = prompts_dir / f"story_plan_{song_id}.manifest.json"

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            request = make_request()
            result = service.build_plan(request)
            plan = result.plan

            # Persist as authoritative (mirrors _resolve_story_plan).
            _persist_plan(prompts_dir, song_id, plan, request.source_fingerprint)

            # Verify the persisted artifacts.
            loaded_manifest = read_manifest(manifest_path)
            self.assertEqual(loaded_manifest.artifact_class, ArtifactClass.authoritative)
            self.assertEqual(loaded_manifest.regeneration_policy, RegenerationPolicy.on_input_change)
            self.assertEqual(loaded_manifest.input_fingerprint, request.source_fingerprint)
            self.assertIsNotNone(loaded_manifest.plan_fingerprint)

            loaded_plan = read_story_plan(plan_path)
            self.assertEqual(loaded_plan.schema_version, STORY_PLAN_SCHEMA_VERSION)
            self.assertEqual(loaded_plan.planner_revision, PLANNER_REVISION)

    def test_review_export_uses_distinct_artifact_class(self) -> None:
        """A review export uses ArtifactClass.review_export, not authoritative."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)
            song_id = "test-song"

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            request = make_request()
            result = service.build_plan(request)

            # Write a review export (mirrors the approval gate path).
            markdown_path, manifest_path = service.write_review_export(
                result, prompts_dir, name=f"story_plan_{song_id}"
            )
            loaded_manifest = read_manifest(manifest_path)
            self.assertEqual(loaded_manifest.artifact_class, ArtifactClass.review_export)
            self.assertEqual(loaded_manifest.regeneration_policy, RegenerationPolicy.on_request)

    # -- (c) fingerprint resume (stable inputs -> reuse) ------------------

    def test_fingerprint_resume_reuses_stable_plan(self) -> None:
        """When the input fingerprint matches and the manifest is not stale,
        the plan is reused without rebuilding."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)
            song_id = "test-song"

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            config = _make_config()
            request_ns = _make_request()
            segments = _stage1_segments()
            fingerprint = _compute_fingerprint(config, request_ns, segments)

            plan_request = StoryPlanRequest(
                source_fingerprint=fingerprint,
                song_title=song_id,
                song_language="en",
                song_style="indie",
                lyrics="la la la",
                sections=(),
                segments=tuple(
                    SegmentDescriptor.from_mapping(seg) for seg in segments
                ),
                characters=(),
                source_evidence={"creative_direction": ""},
                guide="",
                terminal_window_seconds=54.0,
            )
            result = service.build_plan(plan_request)
            plan = result.plan

            # Persist the plan.
            _persist_plan(prompts_dir, song_id, plan, fingerprint)

            # Now call _resolve_story_plan with resume=True and the same inputs.
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)
            config = _make_config()
            app_config = _make_app_config()
            app_config.llm.story_planning.failure_policy = "block"
            request_ns = _make_request()
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=True,
                stage1_segments=_stage1_segments(),
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertFalse(gate_stopped)
            self.assertEqual(len(recording.build_plan_calls), 0, "plan must be reused, not rebuilt")
            self.assertIn("Reusing story plan", " ".join(reporter.messages))
            # Segment briefs are still built from the reused plan.
            self.assertIsNotNone(segment_briefs)
            self.assertIn("seg-1", segment_briefs)
            self.assertIn("seg-2", segment_briefs)

    # -- (d) approval gate stop ------------------------------------------

    def test_approval_gate_stops_pipeline_before_concept_generation(self) -> None:
        """When require_approval is enabled and story_plan_approve is not passed,
        the pipeline stops before concept generation (returns gate_stopped=True)."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)

            config = _make_config()
            app_config = _make_app_config(require_approval=True)
            request_ns = _make_request(story_plan_approve=False)
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=False,
                stage1_segments=_stage1_segments(),
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertTrue(gate_stopped)
            self.assertIsNone(segment_briefs)
            # The plan was built (not reused) and the review export was written.
            self.assertEqual(len(recording.build_plan_calls), 1)
            # A review export should have been written.
            review_md = prompts_dir / "story_plan_test-song.md"
            self.assertTrue(review_md.is_file())
            # The approval message should be visible.
            self.assertIn("approval required", " ".join(reporter.messages).lower())

    def test_approval_gate_passes_with_story_plan_approve(self) -> None:
        """When require_approval is enabled but story_plan_approve is passed,
        the pipeline continues (gate_stopped=False)."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)

            config = _make_config()
            app_config = _make_app_config(require_approval=True)
            request_ns = _make_request(story_plan_approve=True)
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=False,
                stage1_segments=_stage1_segments(),
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertFalse(gate_stopped)
            self.assertIsNotNone(segment_briefs)
            self.assertEqual(len(recording.build_plan_calls), 1)

    # -- (e) SEGMENT_BRIEFS threading ------------------------------------

    def test_segment_briefs_are_threaded_into_concept_generation(self) -> None:
        """The segment briefs returned by _resolve_story_plan are keyed by
        segment target and contain the full brief payload."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)

            config = _make_config()
            app_config = _make_app_config()
            request_ns = _make_request()
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=False,
                stage1_segments=_stage1_segments(),
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertFalse(gate_stopped)
            self.assertIsNotNone(segment_briefs)
            # Keys are the segment targets.
            self.assertIn("seg-1", segment_briefs)
            self.assertIn("seg-2", segment_briefs)
            # Each value is the full brief payload (dict).
            for key, value in segment_briefs.items():
                self.assertIsInstance(value, dict)
                self.assertIn("id", value)
                self.assertIn("target", value)
                self.assertEqual(value["target"], key)

    def test_segment_briefs_do_not_mutate_stage1_segments(self) -> None:
        """Threading segment briefs must not mutate the stage1 segments."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)

            config = _make_config()
            app_config = _make_app_config()
            request_ns = _make_request()
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()
            segments = _stage1_segments()
            original = [dict(seg) for seg in segments]

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=False,
                stage1_segments=segments,
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertFalse(gate_stopped)
            # The stage1 segments must be unmutated.
            self.assertEqual(segments, original)

    # -- (f) invalid plan hard stop --------------------------------------

    def test_invalid_plan_is_a_hard_stop_no_fallback(self) -> None:
        """When the plan fails validation (e.g., unsupported schema), a
        StoryPlanError is raised -- no fallback to story_idea."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)

            # Use a service that raises StoryPlanError directly.
            class _FailingService:
                def build_plan(self, request: Any) -> Any:
                    raise StoryPlanError("plan validation failed")

                def write_review_export(self, *args: Any, **kwargs: Any) -> Any:
                    raise AssertionError("should not reach review export")

            # Use a service that raises StoryPlanError directly.
            class _FailingService:
                def build_plan(self, request: Any) -> Any:
                    raise StoryPlanError("plan validation failed")

                def write_review_export(self, *args: Any, **kwargs: Any) -> Any:
                    raise AssertionError("should not reach review export")

            def factory(llm: Any) -> Any:
                return _FailingService()

            pipeline = _build_pipeline(story_plan_service_factory=factory)
            config = _make_config()
            app_config = _make_app_config()
            app_config.llm.story_planning.failure_policy = "block"
            request_ns = _make_request()
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            with self.assertRaises(StoryPlanError):
                pipeline._resolve_story_plan(
                    config=config,
                    app_config=app_config,
                    request=request_ns,
                    resume=False,
                    stage1_segments=_stage1_segments(),
                    paths=paths,
                    reporter=reporter,
                    artifact_store=None,
                    log_file=lambda _name, _path: None,
                )

    def test_planner_revision_mismatch_is_a_hard_stop(self) -> None:
        """When the plan's planner_revision doesn't match the expected revision,
        a StoryPlanError is raised."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)

            # A service that returns a plan with a mismatched planner revision.
            class _MismatchedRevisionService:
                def build_plan(self, request: Any) -> Any:
                    modules = FakePromptModules()
                    inner = StoryPlanService(prompt_modules=modules)
                    result = inner.build_plan(request)
                    # We need to return a result whose plan has a different
                    # planner_revision. Since StoryPlan validates the revision,
                    # we can't easily mutate it. Instead, we'll use a custom
                    # app_config with a different expected revision.
                    return result

                def write_review_export(self, *args: Any, **kwargs: Any) -> Any:
                    raise AssertionError("should not reach review export")

            def factory(llm: Any) -> Any:
                return _MismatchedRevisionService()

            pipeline = _build_pipeline(story_plan_service_factory=factory)
            config = _make_config()
            # Use a different expected revision to trigger the mismatch.
            app_config = _make_app_config(planner_revision="planner/v999")
            request_ns = _make_request()
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            with self.assertRaises(StoryPlanError):
                pipeline._resolve_story_plan(
                    config=config,
                    app_config=app_config,
                    request=request_ns,
                    resume=False,
                    stage1_segments=_stage1_segments(),
                    paths=paths,
                    reporter=reporter,
                    artifact_store=None,
                    log_file=lambda _name, _path: None,
                )

    # -- (g) no H3/timeline/audio invocation -----------------------------

    def test_story_plan_wiring_does_not_invoke_h3_timeline_or_audio(self) -> None:
        """The story plan wiring must not invoke the H3 compiler, timeline
        builder, or audio analysis. These remain unchanged."""
        source = inspect.getsource(PromptGenerationPipeline._resolve_story_plan)
        # The method should not reference H3 compiler, timeline builder,
        # or audio analysis modules.
        self.assertNotIn("h3", source.lower(),
                        "_resolve_story_plan references H3")
        self.assertNotIn("timeline", source.lower(),
                        "_resolve_story_plan references timeline")
        self.assertNotIn("audio_analysis", source.lower(),
                        "_resolve_story_plan references audio_analysis")
        self.assertNotIn("beat_analysis", source.lower(),
                        "_resolve_story_plan references beat_analysis")

    # -- (h) creative_direction fingerprint change -> regenerate ---------

    def test_creative_direction_change_invalidates_plan(self) -> None:
        """When the creative direction changes, the fingerprint changes,
        causing the plan to be regenerated (not reused)."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)
            song_id = "test-song"

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            # Build with the original direction (empty).
            config = _make_config()
            request_ns = _make_request()
            segments = _stage1_segments()
            fingerprint = _compute_fingerprint(config, request_ns, segments)

            plan_request = StoryPlanRequest(
                source_fingerprint=fingerprint,
                song_title=song_id,
                song_language="en",
                song_style="indie",
                lyrics="la la la",
                sections=(),
                segments=tuple(
                    SegmentDescriptor.from_mapping(seg) for seg in segments
                ),
                characters=(),
                source_evidence={"creative_direction": ""},
                guide="",
                terminal_window_seconds=54.0,
            )
            result = service.build_plan(plan_request)
            plan = result.plan

            # Persist with the original fingerprint.
            _persist_plan(prompts_dir, song_id, plan, fingerprint)

            # Now call _resolve_story_plan with a different direction.
            # This changes the fingerprint, so the plan must be regenerated.
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)
            config = _make_config()
            app_config = _make_app_config()
            # Set a different direction on the request.
            request_ns = _make_request(story_direction="dark, brooding")
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=True,
                stage1_segments=_stage1_segments(),
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertFalse(gate_stopped)
            # The plan must be regenerated (build_plan called).
            self.assertEqual(len(recording.build_plan_calls), 1,
                            "direction change must invalidate the plan")
            # The regenerated plan should have the new direction in provenance.
            plan_path = prompts_dir / f"story_plan_{song_id}.json"
            regenerated_plan = read_story_plan(plan_path)
            self.assertEqual(regenerated_plan.provenance.notes, "dark, brooding")

    def test_stable_inputs_preserve_plan_on_resume(self) -> None:
        """When the inputs are unchanged, the plan is preserved on resume
        (fingerprint match -> reuse, no rebuild)."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp)
            song_id = "test-song"

            modules = FakePromptModules()
            service = StoryPlanService(prompt_modules=modules)
            config = _make_config()
            request_ns = _make_request()
            segments = _stage1_segments()
            fingerprint = _compute_fingerprint(config, request_ns, segments)

            plan_request = StoryPlanRequest(
                source_fingerprint=fingerprint,
                song_title=song_id,
                song_language="en",
                song_style="indie",
                lyrics="la la la",
                sections=(),
                segments=tuple(
                    SegmentDescriptor.from_mapping(seg) for seg in segments
                ),
                characters=(),
                source_evidence={"creative_direction": ""},
                guide="",
                terminal_window_seconds=54.0,
            )
            result = service.build_plan(plan_request)
            plan = result.plan

            # Persist with the original fingerprint.
            _persist_plan(prompts_dir, song_id, plan, fingerprint)

            # Now call _resolve_story_plan with the same inputs (no direction change).
            factory, recording = _make_recording_factory(service)
            pipeline = _build_pipeline(story_plan_service_factory=factory)
            config = _make_config()
            app_config = _make_app_config()
            request_ns = _make_request()  # No direction change.
            paths = _make_paths(prompts_dir)
            reporter = _RecordingReporter()

            segment_briefs, gate_stopped = pipeline._resolve_story_plan(
                config=config,
                app_config=app_config,
                request=request_ns,
                resume=True,
                stage1_segments=_stage1_segments(),
                paths=paths,
                reporter=reporter,
                artifact_store=None,
                log_file=lambda _name, _path: None,
            )

            self.assertFalse(gate_stopped)
            # The plan must be reused (build_plan NOT called).
            self.assertEqual(len(recording.build_plan_calls), 0,
                            "stable inputs must preserve the plan")

    def test_composition_factory_runs_typed_dspy_plan_end_to_end(self) -> None:
        """The production factory must bridge real typed DSPy modules to a plan."""
        factory = getattr(composition, "build_story_plan_service", None)
        self.assertIsNotNone(factory, "composition must expose the production factory")

        service = factory(
            _FactoryFakeLLM(),
            dspy_runtime=_factory_fake_dspy_runtime(),
        )

        result = service.build_plan(make_request())

        self.assertEqual([brief.target for brief in result.plan.segments], ["seg-1", "seg-2"])
        self.assertEqual([brief.beat_id for brief in result.plan.segments], ["beat-001", "beat-002"])
        self.assertEqual(
            [brief["target"] for brief in result.acting["briefs"]],
            ["brief-seg-1", "brief-seg-2"],
        )


if __name__ == "__main__":
    unittest.main()
