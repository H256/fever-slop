"""Fake-DSPy tests for the story-plan prompt modules and service (issue #1385).

The prompt modules are exercised through a fake ``DspyRuntime`` (no real
DSPy LM, no network); the service is exercised through a structural fake
prompt-modules object (the service is pure application code).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from feverslop.application.story_plan_service import (
    STORY_PLAN_PRODUCER,
    SegmentDescriptor,
    StoryPlanError,
    StoryPlanRequest,
    StoryPlanService,
    validate_acting,
    validate_beat_allocation,
    validate_story_plan_payload,
)
from feverslop.prompting.dspy_runtime import DspyRuntime, H3SignatureBundle
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.story_plan_modules import StoryPlanPromptModules
from feverslop.prompting.story_plan_signatures import (
    BeatAllocationResult,
    SegmentBriefDraft,
    StoryBibleResult,
    build_story_plan_signature_bundle,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_FP = _sha("source-1385")


# -- fixtures ---------------------------------------------------------------


def make_request(**overrides: Any) -> StoryPlanRequest:
    defaults = dict(
        source_fingerprint=_FP,
        song_title="Demo Song",
        song_language="en",
        song_style="indie",
        lyrics="la la la",
        sections=({"section": "verse"},),
        segments=(
            SegmentDescriptor(segment_id="seg-1", start_seconds=0.0, end_seconds=30.0),
            SegmentDescriptor(segment_id="seg-2", start_seconds=30.0, end_seconds=60.0),
        ),
        characters=({"id": "char-1", "name": "Singer", "is_singer": True},),
        source_evidence={"creative_direction": "Keep it intimate."},
        guide="story plan guide",
        terminal_window_seconds=45.0,
        story_idea="A singer travels through a haunted mountain to find renewal.",
        locations=(
            {"id": "cave", "name": "Cave"},
            {"id": "well", "name": "Well Grotto"},
        ),
        props=({"id": "well", "name": "Well"},),
    )
    defaults.update(overrides)
    return StoryPlanRequest(**defaults)


def make_request_3seg() -> StoryPlanRequest:
    return make_request(
        segments=(
            SegmentDescriptor(segment_id="seg-1", start_seconds=0.0, end_seconds=20.0),
            SegmentDescriptor(segment_id="seg-2", start_seconds=20.0, end_seconds=40.0),
            SegmentDescriptor(segment_id="seg-3", start_seconds=40.0, end_seconds=60.0),
        ),
        terminal_window_seconds=45.0,
    )


def valid_bible() -> dict:
    return {
        "sections": [{"title": "Opening", "summary": "A singer steps onto the stage."}],
        "narrative_facts": ["The song is about leaving."],
    }


def valid_allocation() -> dict:
    return {
        "briefs": [
            {
                "brief_id": "brief-1",
                "segment_id": "seg-1",
                "start_seconds": 0.0,
                "end_seconds": 30.0,
                "beat_indices": [0],
                "required": ["beat-0"],
                "forbidden": [],
            },
            {
                "brief_id": "brief-2",
                "segment_id": "seg-2",
                "start_seconds": 30.0,
                "end_seconds": 60.0,
                "beat_indices": [1],
                "required": [],
                "forbidden": ["beat-9"],
            },
        ]
    }


def acting_for_brief_ids(*brief_ids: str) -> dict:
    briefs = [
        {
            "brief_id": brief_id,
            "objective": f"Objective for {brief_id}.",
            "emotional_turn": "Nervous to determined.",
            "actor_states": [
                {
                    "character_id": "char-1",
                    "inner_state": "nervous",
                    "physical_state": "standing",
                }
            ],
        }
        for brief_id in brief_ids
    ]
    return {
        "briefs": briefs,
        "character_arcs": [
            {
                "character_id": "char-1",
                "character_name": "Singer",
                "arc_summary": "Nervous to relieved",
            }
        ],
    }


def valid_acting() -> dict:
    return acting_for_brief_ids("brief-1", "brief-2")


def duplicate_target_allocation() -> dict:
    return {
        "briefs": [
            {
                "brief_id": "brief-1",
                "segment_id": "seg-1",
                "start_seconds": 0.0,
                "end_seconds": 20.0,
                "beat_indices": [0],
                "required": [],
                "forbidden": [],
            },
            {
                "brief_id": "brief-2",
                "segment_id": "seg-1",
                "start_seconds": 0.0,
                "end_seconds": 20.0,
                "beat_indices": [1],
                "required": [],
                "forbidden": [],
            },
            {
                "brief_id": "brief-3",
                "segment_id": "seg-2",
                "start_seconds": 20.0,
                "end_seconds": 40.0,
                "beat_indices": [2],
                "required": [],
                "forbidden": [],
            },
            {
                "brief_id": "brief-4",
                "segment_id": "seg-3",
                "start_seconds": 40.0,
                "end_seconds": 60.0,
                "beat_indices": [3],
                "required": [],
                "forbidden": [],
            },
        ]
    }


def valid_repair_plan() -> dict:
    return {
        "mode": "music_video",
        "source_fingerprint": _FP,
        "provenance": {
            "producer": "stale-producer",
            "source_refs": ["stale"],
            "notes": "stale",
        },
        "characters": [{"id": "char-1", "name": "Singer", "is_singer": True}],
        "beats": [],
        "arcs": [],
        "segments": [
            {
                "id": "brief-1",
                "target": "seg-1",
                "vocal_presentation": "offscreen",
                "audio_ref": {"segment_id": "seg-1", "fingerprint": _sha(f"seg-1:{_FP}")},
            },
            {
                "id": "brief-2",
                "target": "seg-2",
                "vocal_presentation": "offscreen",
                "audio_ref": {"segment_id": "seg-2", "fingerprint": _sha(f"seg-2:{_FP}")},
            },
        ],
    }


# -- fakes -------------------------------------------------------------------


class _FakeClient:
    base_url = "http://fake.local/v1"
    api_key = "fake-key"


class _FakeDspyLLM:
    """LLM double exposing the attributes the modules/runtime read."""

    def __init__(self) -> None:
        self.model = "fake-model"
        self.client = _FakeClient()
        self.max_tokens = 2048
        self.dspy_temperature = 0.4
        self.dspy_cache = False
        self.max_retries = 1
        self.request_timeout_seconds = None
        self.llm_limiter = None
        self.metrics = None


class _FakePredictor:
    def __init__(self, signature: Any) -> None:
        self.signature = signature
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        name = self.signature.__name__
        if name == "StoryPlanBible":
            return {"bible": {"premise": "p", "theme": "t"}}
        if name == "BeatAllocation":
            return {"allocation": {"beats": [{"phase": "opening", "description": "d"}]}}
        if name == "Acting":
            return {"result": {"briefs": [], "arcs": []}}
        return {"plan": {"mode": "music_video"}}


def _make_runtime() -> tuple[DspyRuntime, list, list]:
    lm_calls: list[tuple[str, float | None, int | None]] = []
    predictors: list[_FakePredictor] = []

    def lm_factory(name: str, **kwargs: Any) -> str:
        lm_calls.append((name, kwargs.get("temperature"), kwargs.get("max_tokens")))
        return "lm"

    def predict_factory(signature: Any) -> _FakePredictor:
        predictor = _FakePredictor(signature)
        predictors.append(predictor)
        return predictor

    runtime = DspyRuntime(
        signatures=H3SignatureBundle(object(), object(), object(), object()),
        lm_factory=lm_factory,
        predict_factory=predict_factory,
        context_factory=lambda **kwargs: nullcontext(kwargs),
    )
    return runtime, lm_calls, predictors


def _predictor(predictors: list[_FakePredictor], name: str) -> _FakePredictor:
    return next(p for p in predictors if p.signature.__name__ == name)


class FakePromptModules:
    """Structural double for the four story-plan job methods."""

    def __init__(
        self,
        *,
        bible: dict | None = None,
        allocation: dict | None = None,
        acting: dict | None = None,
        repair: dict | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.outputs = {
            "bible": bible if bible is not None else valid_bible(),
            "beat_allocation": allocation if allocation is not None else valid_allocation(),
            "acting": acting if acting is not None else valid_acting(),
            "repair": repair if repair is not None else valid_repair_plan(),
        }

    def bible(self, **kwargs: Any) -> dict:
        self.calls.append(("bible", dict(kwargs)))
        return self.outputs["bible"]

    def beat_allocation(self, **kwargs: Any) -> dict:
        self.calls.append(("beat_allocation", dict(kwargs)))
        return self.outputs["beat_allocation"]

    def acting(self, **kwargs: Any) -> dict:
        self.calls.append(("acting", dict(kwargs)))
        return self.outputs["acting"]

    def repair(self, **kwargs: Any) -> dict:
        self.calls.append(("repair", dict(kwargs)))
        return self.outputs["repair"]


class StrictPromptModules:
    """Records interface access and rejects any method outside the four jobs."""

    _ALLOWED = frozenset({"bible", "beat_allocation", "acting", "repair"})

    def __init__(self, outputs: dict[str, Any]) -> None:
        object.__setattr__(self, "_outputs", outputs)
        object.__setattr__(self, "accessed", [])

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._ALLOWED:
            raise AttributeError(f"service touched unexpected interface: {name}")
        self.accessed.append(name)
        return self._outputs[name]


# -- prompt module tests -------------------------------------------------------


class StoryPlanModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.lm_calls, self.predictors = _make_runtime()
        self.modules = StoryPlanPromptModules(_FakeDspyLLM(), dspy_runtime=self.runtime)

    def test_bible_payload_allowlist(self) -> None:
        self.modules.bible(
            story_text="A singer leaves.",
            creative_direction="Keep it intimate.",
            characters=[{"id": "char-1", "name": "Singer", "alignment": ["a", "b"]}],
            locations=[{"id": "loc-1", "name": "Stage"}],
            props=[{"id": "prop-1", "name": "Mic"}],
        )
        kwargs = _predictor(self.predictors, "StoryPlanBible").calls[0]
        self.assertEqual(
            sorted(kwargs),
            [
                "characters",
                "config",
                "creative_direction",
                "guide",
                "locations",
                "props",
                "story_text",
            ],
        )
        self.assertIsInstance(kwargs["guide"], str)
        self.assertTrue(kwargs["guide"])

    def test_beat_allocation_payload_allowlist(self) -> None:
        self.modules.beat_allocation(
            creative_direction="cd",
            bible={"premise": "p"},
            characters=[],
            locations=[],
            props=[],
        )
        kwargs = _predictor(self.predictors, "BeatAllocation").calls[0]
        self.assertEqual(
            sorted(kwargs),
            [
                "bible",
                "characters",
                "config",
                "creative_direction",
                "guide",
                "locations",
                "props",
            ],
        )

    def test_acting_payload_allowlist(self) -> None:
        self.modules.acting(
            creative_direction="cd",
            bible={"premise": "p"},
            beats=[{"id": "beat-1", "phase": "opening"}],
            segments=[{"segment_id": "seg-1"}],
            characters=[],
            locations=[],
            props=[],
        )
        kwargs = _predictor(self.predictors, "Acting").calls[0]
        self.assertEqual(
            sorted(kwargs),
            [
                "beats",
                "bible",
                "characters",
                "config",
                "creative_direction",
                "expected_brief_ids",
                "guide",
                "locations",
                "props",
                "segments",
            ],
        )

    def test_module_compacts_acoustic_evidence_from_inputs(self) -> None:
        self.modules.bible(
            story_text="A singer leaves.",
            creative_direction="Keep it intimate.",
            characters=[{"id": "char-1", "name": "Singer", "alignment": ["a", "b"]}],
            locations=[],
            props=[],
        )
        bible_payload = json.dumps(_predictor(self.predictors, "StoryPlanBible").calls[0])
        self.assertNotIn("alignment", bible_payload)
        allocation_payload = json.dumps(
            _predictor(self.predictors, "BeatAllocation").calls
        )
        self.assertNotIn("vocal_events", allocation_payload)
        self.assertNotIn("word_timestamps", allocation_payload)

    def test_module_config_uses_policy_max_tokens_per_job(self) -> None:
        self.modules.bible(
            story_text="t",
            creative_direction="cd",
            characters=[],
            locations=[],
            props=[],
        )
        self.modules.beat_allocation(
            creative_direction="cd",
            bible={"premise": "p"},
            characters=[],
            locations=[],
            props=[],
        )
        self.modules.acting(
            creative_direction="cd",
            bible={"premise": "p"},
            beats=[],
            segments=[],
            characters=[],
            locations=[],
            props=[],
        )
        self.assertEqual(
            _predictor(self.predictors, "StoryPlanBible").calls[0]["config"],
            {"max_tokens": 4096},
        )
        self.assertEqual(
            _predictor(self.predictors, "BeatAllocation").calls[0]["config"],
            {"max_tokens": 4096},
        )
        self.assertEqual(
            _predictor(self.predictors, "Acting").calls[0]["config"],
            {"max_tokens": 1536},
        )

    def test_module_resolves_per_task_temperature(self) -> None:
        # __init__ creates one LM per job in _BUNDLE_TASK_NAMES order:
        # acting, beat_allocation, bible. The per-task values come
        # from DEFAULT_TASK_TEMPERATURES, never the global dspy_temperature.
        self.assertEqual(
            [(name, temperature) for name, temperature, _ in self.lm_calls],
            [
                ("openai/fake-model", 0.6),  # story_plan_acting
                ("openai/fake-model", 0.2),  # story_plan_beat_allocation
                ("openai/fake-model", 0.2),  # story_plan_bible
            ],
        )
        self.assertNotIn(0.4, [t for _, t, _ in self.lm_calls])

    def test_module_bounds_the_dspy_lm_not_just_predictor_config(self) -> None:
        """DSPy must receive the bounded budget as its LM default.

        Some providers ignore a nested predictor ``config`` when creating a
        completion.  Leaving the LM at the application's 65k default turns a
        malformed small-model acting response into a multi-minute request.
        """
        self.assertEqual(
            [max_tokens for _, _, max_tokens in self.lm_calls],
            [1536, 4096, 4096],
        )

    def test_module_requires_configured_dspy_llm(self) -> None:
        with self.assertRaises(RuntimeError):
            StoryPlanPromptModules(object(), dspy_runtime=self.runtime)

        class NoModel:
            client = _FakeClient()

        with self.assertRaises(RuntimeError):
            StoryPlanPromptModules(NoModel(), dspy_runtime=self.runtime)

        class NoClient:
            model = "fake-model"

        with self.assertRaises(RuntimeError):
            StoryPlanPromptModules(NoClient(), dspy_runtime=self.runtime)


class StoryPlanSignatureBundleTests(unittest.TestCase):
    def test_dspy_structured_transport_uses_json_objects(self) -> None:
        """DSPy transports model JSON; the service owns Pydantic validation."""
        bundle = build_story_plan_signature_bundle()
        self.assertEqual(dict[str, Any], bundle["bible"].output_fields["bible"].annotation)
        self.assertEqual(dict[str, Any], bundle["beat_allocation"].input_fields["bible"].annotation)
        self.assertEqual(dict[str, Any], bundle["beat_allocation"].output_fields["allocation"].annotation)
        self.assertEqual(dict[str, Any], bundle["acting"].input_fields["bible"].annotation)
        self.assertEqual(dict[str, Any], bundle["acting"].output_fields["result"].annotation)

    def test_signature_bundle_covers_three_small_jobs(self) -> None:
        bundle = build_story_plan_signature_bundle()
        self.assertEqual(sorted(bundle), ["acting", "beat_allocation", "bible"])
        self.assertIn("guide", bundle["bible"].input_fields)
        self.assertIn("story_text", bundle["bible"].input_fields)
        self.assertIn("creative_direction", bundle["bible"].input_fields)
        self.assertIn("bible", bundle["bible"].output_fields)
        self.assertIn("allocation", bundle["beat_allocation"].output_fields)
        self.assertIn("result", bundle["acting"].output_fields)


class StoryPlanGuideTests(unittest.TestCase):
    def test_story_plan_guides_are_bundled_markdown_resources(self) -> None:
        for name in (
            "story-plan-bible",
            "story-plan-beat-allocation",
            "story-plan-acting",
        ):
            guide = load_markdown_guide(name)
            self.assertIsInstance(guide, str)
            self.assertTrue(guide.strip())
        self.assertTrue(load_markdown_guide("story-plan-bible").startswith("# Story Plan Bible"))


class StoryPlanTypedContractTests(unittest.TestCase):
    def test_bible_result_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            StoryBibleResult(premise="p", surprise="x")
        ok = StoryBibleResult(premise="p")
        self.assertEqual(ok.character_notes, {})

    def test_segment_brief_draft_forbids_audio_and_render_fields(self) -> None:
        base = {"target": "seg-1", "vocal_presentation": "offscreen"}
        SegmentBriefDraft.model_validate(base)
        for key in (
            "audio_ref",
            "audio_features",
            "camera",
            "shot_type",
            "timestamps",
            "lyrics",
        ):
            with self.assertRaises(ValidationError):
                SegmentBriefDraft.model_validate({**base, key: "x"})

    def test_beat_allocation_result_requires_at_least_one_beat(self) -> None:
        with self.assertRaises(ValidationError):
            BeatAllocationResult(beats=[])
        ok = BeatAllocationResult(beats=[{"phase": "opening", "description": "d"}])
        self.assertEqual(len(ok.beats), 1)


# -- service tests -------------------------------------------------------------


class StoryPlanServiceTests(unittest.TestCase):
    def test_service_deterministically_binds_every_segment_to_a_small_beat_sheet(self) -> None:
        """The LLM writes the arc; Python owns the complete scene binding."""
        modules = FakePromptModules(
            allocation={
                "beats": [
                    {"phase": "opening", "description": "The journey begins."},
                    {"phase": "development", "description": "The danger closes in."},
                    {"phase": "resolution", "description": "Renewal in the well."},
                ]
            },
            acting=acting_for_brief_ids("brief-seg-1", "brief-seg-2"),
        )

        result = StoryPlanService(prompt_modules=modules).build_plan(make_request())

        self.assertEqual(
            [(brief.target, brief.beat_id) for brief in result.plan.segments],
            [("seg-1", "beat-001"), ("seg-2", "beat-003")],
        )
        allocation_call = next(payload for name, payload in modules.calls if name == "beat_allocation")
        self.assertNotIn("segments", allocation_call)
        self.assertNotIn("repair", [name for name, _ in modules.calls])

    def test_service_reports_visible_progress_for_each_creative_job(self) -> None:
        class RecordingReporter:
            def __init__(self) -> None:
                self.progress: list[str] = []

            def step(self, _title: str) -> None:
                pass

            def message(self, _text: str) -> None:
                pass

            def warning(self, _text: str, *, title: str | None = None) -> None:
                pass

            def table(self, _title: str, _columns: list[str], _rows: list[list[str]]) -> None:
                pass

            def run_progress(self, description: str, func: Any) -> Any:
                self.progress.append(description)
                return func()

        reporter = RecordingReporter()
        StoryPlanService(prompt_modules=FakePromptModules(), reporter=reporter).build_plan(make_request())

        self.assertEqual(
            reporter.progress,
            [
                "Story plan - reading the story",
                "Story plan - shaping the arc",
                "Story plan - writing acting",
            ],
        )

    def test_service_reports_acting_batch_and_total_scene_progress(self) -> None:
        class RecordingReporter:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def step(self, _title: str) -> None:
                pass

            def message(self, text: str) -> None:
                self.messages.append(text)

            def warning(self, _text: str, *, title: str | None = None) -> None:
                pass

            def table(self, _title: str, _columns: list[str], _rows: list[list[str]]) -> None:
                pass

            def run_progress(self, _description: str, func: Any) -> Any:
                return func()

        class PerBatchPromptModules(FakePromptModules):
            def acting(self, **kwargs: Any) -> dict:
                return acting_for_brief_ids(*kwargs["expected_brief_ids"])

        reporter = RecordingReporter()
        StoryPlanService(
            prompt_modules=PerBatchPromptModules(),
            reporter=reporter,
            acting_batch_size=1,
        ).build_plan(make_request())

        self.assertTrue(
            any("Story plan - acting: 0/2" in message for message in reporter.messages)
        )
        self.assertTrue(
            any("Story plan - acting: 1/2" in message and "batch 1/2" in message
                for message in reporter.messages)
        )
        self.assertTrue(
            any("Story plan - acting: 2/2" in message and "batch 2/2" in message
                for message in reporter.messages)
        )

    def test_pydantic_bible_output_is_accepted_from_typed_dspy(self) -> None:
        """DSPy may deserialize an annotated output before the service sees it."""
        modules = FakePromptModules(
            bible=StoryBibleResult(
                premise="A singer leaves home.",
                theme="release",
            )
        )

        result = StoryPlanService(prompt_modules=modules).build_plan(make_request())

        self.assertEqual([brief.target for brief in result.plan.segments], ["seg-1", "seg-2"])

    def test_typed_dspy_outputs_produce_beats_and_segment_briefs(self) -> None:
        """The service consumes the public typed planning DTOs directly."""
        modules = FakePromptModules(
            allocation={
                "beats": [
                    {"phase": "opening", "description": "The journey begins."},
                    {"phase": "resolution", "description": "The journey resolves."},
                ],
                "brief_allocations": [
                    {"target": "seg-1", "beat_index": 0},
                    {"target": "seg-2", "beat_index": 1},
                ],
            },
            acting={
                "arcs": [],
                "briefs": [
                    {
                        "target": "seg-1",
                        "vocal_presentation": "on_screen",
                        "objective": "Invite the audience into the journey.",
                        "emotional_turn": "Guarded to hopeful.",
                        "actor_states": [{"character_id": "char-1", "state": "guarded"}],
                    },
                    {
                        "target": "seg-2",
                        "vocal_presentation": "offscreen",
                        "objective": "Release the tension.",
                        "emotional_turn": "Hopeful to free.",
                        "actor_states": [{"character_id": "char-1", "state": "free"}],
                    },
                ],
            },
        )

        result = StoryPlanService(prompt_modules=modules).build_plan(make_request())

        self.assertEqual([beat.id for beat in result.plan.beats], ["beat-001", "beat-002"])
        self.assertEqual([brief.target for brief in result.plan.segments], ["seg-1", "seg-2"])
        self.assertEqual(result.plan.segments[0].audio_ref.segment_id, "seg-1")
        self.assertEqual(
            "Invite the audience into the journey.",
            result.plan.segments[0].objective,
        )
        self.assertEqual("Guarded to hopeful.", result.plan.segments[0].emotional_turn)
        self.assertEqual("guarded", result.plan.segments[0].actor_states[0].state)

    def test_story_plan_bindings_override_creative_location_choices(self) -> None:
        modules = FakePromptModules(
            allocation={
                "beats": [
                    {"phase": "opening", "description": "Caves", "location_id": "cave", "character_ids": ["char-1"]},
                    {"phase": "resolution", "description": "Well", "location_id": "well", "character_ids": ["char-1"]},
                ],
                "brief_allocations": [
                    {"target": "seg-1", "beat_index": 0},
                    {"target": "seg-2", "beat_index": 1},
                ],
            },
            acting={
                "briefs": [
                    {"target": "seg-1", "location_id": "well", "objective": "Move.", "emotional_turn": "fear to resolve", "actor_states": [{"character_id": "char-1", "state": "fearful"}]},
                    {"target": "seg-2", "location_id": "cave", "objective": "Arrive.", "emotional_turn": "resolve to calm", "actor_states": [{"character_id": "char-1", "state": "calm"}]},
                ],
            },
        )
        result = StoryPlanService(prompt_modules=modules).build_plan(make_request())
        self.assertEqual([brief.location_id for brief in result.plan.segments], ["cave", "well"])

    def test_missing_actor_states_are_filled_from_locked_beat_cast(self) -> None:
        modules = FakePromptModules(
            allocation={
                "beats": [
                    {"phase": "opening", "description": "Caves", "character_ids": ["char-1"]},
                    {"phase": "resolution", "description": "Well", "character_ids": ["char-1"]},
                ],
                "brief_allocations": [
                    {"target": "seg-1", "beat_index": 0},
                    {"target": "seg-2", "beat_index": 1},
                ],
            },
            acting={
                "briefs": [
                    {"target": "seg-1", "objective": "Move.", "emotional_turn": "fear to resolve"},
                    {"target": "seg-2", "objective": "Arrive.", "emotional_turn": "resolve to calm", "actor_states": [{"character_id": "char-1", "state": "calm"}]},
                ],
            },
        )

        result = StoryPlanService(prompt_modules=modules).build_plan(make_request())

        first = result.acting["briefs"][0]
        self.assertEqual(first["actor_states"], [{"character_id": "char-1", "state": "present"}])

    def test_acting_is_split_into_bounded_batches_and_merged_in_order(self) -> None:
        class BatchedModules(FakePromptModules):
            def __init__(self, allocation: dict) -> None:
                super().__init__(allocation=allocation)
                self.acting_targets: list[list[str]] = []

            def acting(self, **kwargs: Any) -> dict:
                targets = [brief["segment_id"] for brief in kwargs["briefs"]]
                self.acting_targets.append(targets)
                return {
                    "briefs": [
                        {
                            "target": target,
                            "objective": f"Objective {target}",
                            "emotional_turn": "fear to resolve",
                            "actor_states": [{"character_id": "char-1", "state": "present"}],
                        }
                        for target in targets
                    ]
                }

        allocation = {
            "beats": [
                {"phase": "opening", "description": "A", "character_ids": ["char-1"]},
                {"phase": "development", "description": "B", "character_ids": ["char-1"]},
                {"phase": "resolution", "description": "C", "character_ids": ["char-1"]},
            ],
            "brief_allocations": [
                {"target": "seg-1", "beat_index": 0},
                {"target": "seg-2", "beat_index": 1},
                {"target": "seg-3", "beat_index": 2},
            ],
        }
        modules = BatchedModules(allocation)
        result = StoryPlanService(
            prompt_modules=modules, acting_batch_size=2
        ).build_plan(make_request_3seg())

        self.assertEqual(modules.acting_targets, [["seg-1", "seg-2"], ["seg-3"]])
        self.assertEqual(
            [brief["target"] for brief in result.acting["briefs"]],
            ["brief-seg-1", "brief-seg-2", "brief-seg-3"],
        )

    def test_acting_retries_only_missing_briefs(self) -> None:
        class MissingOnceModules(FakePromptModules):
            def __init__(self, allocation: dict) -> None:
                super().__init__(allocation=allocation)
                self.targets: list[list[str]] = []

            def acting(self, **kwargs: Any) -> dict:
                targets = [brief["segment_id"] for brief in kwargs["briefs"]]
                self.targets.append(targets)
                returned = targets[:1] if len(self.targets) == 1 else targets
                return {
                    "briefs": [
                        {
                            "target": target,
                            "objective": f"Objective {target}",
                            "emotional_turn": "fear to resolve",
                            "actor_states": [{"character_id": "char-1", "state": "present"}],
                        }
                        for target in returned
                    ]
                }

        allocation = {
            "beats": [
                {"phase": "opening", "description": "A", "character_ids": ["char-1"]},
                {"phase": "resolution", "description": "B", "character_ids": ["char-1"]},
            ],
            "brief_allocations": [
                {"target": "seg-1", "beat_index": 0},
                {"target": "seg-2", "beat_index": 1},
            ],
        }
        modules = MissingOnceModules(allocation)
        result = StoryPlanService(
            prompt_modules=modules, acting_batch_size=2
        ).build_plan(make_request())

        self.assertEqual(modules.targets, [["seg-1", "seg-2"], ["seg-2"]])
        self.assertEqual(len(result.acting["briefs"]), 2)

    def test_valid_plan_skips_repair(self) -> None:
        modules = FakePromptModules()
        service = StoryPlanService(prompt_modules=modules)
        result = service.build_plan(make_request())
        self.assertEqual(
            [name for name, _ in modules.calls],
            ["bible", "beat_allocation", "acting"],
        )
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(
            sorted(dict(modules.calls[0][1])),
            [
                "characters",
                "guide",
                "locations",
                "lyrics",
                "props",
                "sections",
                "song_language",
                "song_style",
                "song_title",
                "source_evidence",
                "story_idea",
            ],
        )
        self.assertEqual(
            sorted(dict(modules.calls[1][1])),
            [
                "characters",
                "guide",
                "locations",
                "lyrics",
                "narrative_bible",
                "props",
                "segment_count",
                "song_title",
                "terminal_window_seconds",
            ],
        )
        self.assertEqual(
            sorted(dict(modules.calls[2][1])),
            [
                "briefs",
                "characters",
                "expected_brief_ids",
                "guide",
                "locations",
                "lyrics",
                "props",
                "segments",
                "song_language",
                "song_title",
                "typed_beats",
            ],
        )
        self.assertEqual(validate_story_plan_payload(result.plan.model_dump()), [])

    def test_service_only_touches_the_four_job_interface(self) -> None:
        outputs = {
            "bible": lambda **kwargs: valid_bible(),
            "beat_allocation": lambda **kwargs: valid_allocation(),
            "acting": lambda **kwargs: valid_acting(),
            "repair": lambda **kwargs: valid_repair_plan(),
        }
        modules = StrictPromptModules(outputs)
        service = StoryPlanService(prompt_modules=modules)
        service.build_plan(make_request())
        self.assertEqual(modules.accessed, ["bible", "beat_allocation", "acting"])

    def test_invalid_candidate_fails_without_a_full_plan_repair(self) -> None:
        modules = FakePromptModules(
            allocation=duplicate_target_allocation(),
            acting=acting_for_brief_ids("brief-1", "brief-2", "brief-3", "brief-4"),
        )
        service = StoryPlanService(prompt_modules=modules)
        with self.assertRaises(StoryPlanError) as ctx:
            service.build_plan(make_request_3seg())
        self.assertIn("deterministic assembly", str(ctx.exception))
        self.assertNotIn("repair", [name for name, _ in modules.calls])

    def test_second_invalid_repair_result_raises_story_plan_error(self) -> None:
        bad_repair = valid_repair_plan()
        del bad_repair["segments"][0]["audio_ref"]
        modules = FakePromptModules(
            allocation=duplicate_target_allocation(),
            acting=acting_for_brief_ids("brief-1", "brief-2", "brief-3", "brief-4"),
            repair=bad_repair,
        )
        service = StoryPlanService(prompt_modules=modules)
        with self.assertRaises(StoryPlanError) as ctx:
            service.build_plan(make_request_3seg())
        self.assertIn("deterministic assembly", str(ctx.exception))
        self.assertIn(
            "plan_validation_failed",
            [diagnostic["code"] for diagnostic in ctx.exception.diagnostics],
        )

    def test_assembled_plan_derives_provenance_from_user_direction(self) -> None:
        modules = FakePromptModules()
        service = StoryPlanService(prompt_modules=modules)
        result = service.build_plan(make_request())
        self.assertEqual(result.plan.provenance.producer, STORY_PLAN_PRODUCER)
        self.assertEqual(result.plan.provenance.source_refs, ["Demo Song"])
        self.assertEqual(result.plan.provenance.notes, "Keep it intimate.")

    def test_service_candidate_invents_no_ids_and_assigns_deterministic_audio_refs(self) -> None:
        request = make_request()
        modules = FakePromptModules()
        service = StoryPlanService(prompt_modules=modules)
        result = service.build_plan(request)
        segment_ids = {segment.segment_id for segment in request.segments}
        character_ids = {character["id"] for character in request.characters}
        for segment in result.plan.segments:
            self.assertIn(segment.target, segment_ids)
            self.assertEqual(segment.audio_ref.segment_id, segment.target)
            self.assertEqual(
                segment.audio_ref.fingerprint,
                _sha(f"{segment.target}:{request.source_fingerprint}"),
            )
        for character in result.plan.characters:
            self.assertIn(character.id, character_ids)
        self.assertEqual(result.plan.beats, [])
        self.assertEqual(result.plan.arcs, [])

    def test_service_forbidden_render_keys_recorded_in_diagnostics(self) -> None:
        bible_with_camera = valid_bible()
        bible_with_camera["camera"] = "wide shot"
        modules = FakePromptModules(bible=bible_with_camera)
        service = StoryPlanService(prompt_modules=modules)
        result = service.build_plan(make_request())
        render_diagnostics = [
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic["code"] == "forbidden_keys" and diagnostic.get("job") == "bible"
        ]
        self.assertEqual(len(render_diagnostics), 1)
        self.assertEqual(render_diagnostics[0]["keys"], ["camera"])

    def test_service_forbidden_audio_data_keys_recorded_in_diagnostics(self) -> None:
        acting_with_audio = valid_acting()
        acting_with_audio["audio_features"] = {"spectral": True}
        modules = FakePromptModules(acting=acting_with_audio)
        service = StoryPlanService(prompt_modules=modules)
        result = service.build_plan(make_request())
        audio_diagnostics = [
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic["code"] == "forbidden_keys" and diagnostic.get("job") == "acting"
        ]
        self.assertEqual(len(audio_diagnostics), 1)
        self.assertEqual(audio_diagnostics[0]["keys"], ["audio_features"])


class StoryPlanRequestValidationTests(unittest.TestCase):
    def test_request_validate_requires_terminal_window_inside_span(self) -> None:
        with self.assertRaises(StoryPlanError):
            make_request(terminal_window_seconds=0).validate()
        with self.assertRaises(StoryPlanError):
            make_request(terminal_window_seconds=60.0).validate()
        make_request().validate()


class StoryPlanValidatorTests(unittest.TestCase):
    def test_validate_beat_allocation_enforces_required_forbidden_constraints(self) -> None:
        segments = [SegmentDescriptor(segment_id="seg-1", start_seconds=0.0, end_seconds=30.0)]
        allocation = {
            "briefs": [
                {
                    "brief_id": "brief-1",
                    "segment_id": "seg-1",
                    "start_seconds": 0.0,
                    "end_seconds": 30.0,
                    "beat_indices": [0],
                    "required": ["beat-0", 1],
                    "forbidden": [],
                }
            ]
        }
        codes = {
            diagnostic["code"]
            for diagnostic in validate_beat_allocation(allocation, segments, 15.0)
        }
        self.assertIn("invalid_constraints", codes)

        bad_beats = {
            "briefs": [
                {
                    "brief_id": "brief-1",
                    "segment_id": "seg-1",
                    "start_seconds": 0.0,
                    "end_seconds": 30.0,
                    "beat_indices": [0, 1.5],
                    "required": [],
                    "forbidden": [],
                }
            ]
        }
        codes = {
            diagnostic["code"]
            for diagnostic in validate_beat_allocation(bad_beats, segments, 15.0)
        }
        self.assertIn("invalid_beats", codes)

    def test_validate_acting_requires_structured_objective_turn_and_states(self) -> None:
        allocation = {"briefs": [{"brief_id": "brief-1"}, {"brief_id": "brief-2"}]}
        acting = {
            "briefs": [
                {
                    "brief_id": "brief-1",
                    "objective": "   ",
                    "emotional_turn": "",
                    "actor_states": [],
                }
            ]
        }
        codes = {
            diagnostic["code"] for diagnostic in validate_acting(allocation, acting)
        }
        self.assertIn("missing_objective", codes)
        self.assertIn("missing_emotional_turn", codes)
        self.assertIn("missing_actor_states", codes)
        self.assertIn("uncovered_acting_briefs", codes)

        stateless = {
            "briefs": [
                {
                    "brief_id": "brief-1",
                    "objective": "o",
                    "emotional_turn": "t",
                    "actor_states": [{"inner_state": "nervous"}],
                }
            ]
        }
        codes = {
            diagnostic["code"]
            for diagnostic in validate_acting(
                {"briefs": [{"brief_id": "brief-1"}]}, stateless
            )
        }
        self.assertIn("missing_character_id", codes)


class StoryPlanTypedActingTests(unittest.TestCase):
    def test_typed_acting_result_is_structured_not_free_prose(self) -> None:
        modules = FakePromptModules()
        service = StoryPlanService(prompt_modules=modules)
        result = service.build_plan(make_request())
        self.assertEqual(sorted(result.acting), ["arcs", "briefs"])
        for brief in result.acting["briefs"]:
            self.assertIsInstance(brief["objective"], str)
            self.assertIsInstance(brief["emotional_turn"], str)
            for state in brief["actor_states"]:
                self.assertEqual(sorted(state), ["character_id", "state"])
        for arc in result.acting["arcs"]:
            self.assertEqual(
                sorted(arc), ["beat_ids", "character_id", "from_state", "to_state"]
            )


class StoryPlanReviewExportTests(unittest.TestCase):
    def test_render_review_export_is_markdown_and_persisted_as_review_export(self) -> None:
        modules = FakePromptModules()
        service = StoryPlanService(prompt_modules=modules)
        result = service.build_plan(make_request())
        markdown = service.render_review_export(result)
        self.assertTrue(markdown.startswith("# Story Plan Review"))
        with self.assertRaises(json.JSONDecodeError):
            json.loads(markdown)
        with tempfile.TemporaryDirectory() as tmp:
            markdown_path, manifest_path = service.write_review_export(result, Path(tmp))
            self.assertEqual(markdown_path.name, "story-plan.md")
            self.assertEqual(manifest_path.name, "story-plan.manifest.json")
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), markdown)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_class"], "review_export")
            self.assertEqual(manifest["regeneration_policy"], "on_request")
            self.assertEqual(manifest["input_fingerprint"], _FP)
            expected_plan_fingerprint = hashlib.sha256(
                json.dumps(result.plan.model_dump(mode="json"), sort_keys=True).encode(
                    "utf-8"
                )
            ).hexdigest()
            self.assertEqual(manifest["plan_fingerprint"], expected_plan_fingerprint)


if __name__ == "__main__":
    unittest.main()
