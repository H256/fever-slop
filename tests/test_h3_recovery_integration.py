"""Real compiler/checkpoint recovery with only the model boundary replaced."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from feverslop.adapters.h3_prompt_checkpoints import H3PromptCheckpointStore
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.dspy_h3_models import MusicIntent, PlannedShot, ResolvedPromptPlan
from feverslop.prompting.dspy_h3_prompt_builder import DspyH3PromptBuilder


class RecordingPlanner:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        return SimpleNamespace(plan=ResolvedPromptPlan(
            creative_intent="A performer waits.",
            style_opening="Live-action cinematic imagery uses cool practical lighting.",
            shots=[PlannedShot(
                shot_number=1, start_seconds=0,
                end_seconds=request["duration_seconds"],
                description="A performer folds a letter.",
                camera_behavior="slow dolly left",
            )],
            overall_soundscape="Quiet room tone.", music_intent=MusicIntent.NONE,
        ))


class H3RecoveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "prompts.json"
        self.planner = RecordingPlanner()
        self.segments = [
            {"segment_id": "blocked", "scene": 1, "start": 0, "end": 2,
             "duration": 2, "locked_facts": [{
                 "category": "wardrobe", "key": "hero",
                 "value": "wearing a scarlet velvet cape", "source_id": "cast:hero",
             }]},
            {"segment_id": "valid", "scene": 2, "start": 2, "end": 4, "duration": 2},
        ]

    def run_batch(self, *, segments=None, replan=False):
        # New instances on every run ensure that reuse comes from disk, not memory.
        builder = DspyH3PromptBuilder(self.planner)
        with patch.object(builder, "_deterministic_fallback",
                          wraps=builder._deterministic_fallback) as fallback:
            builder.build_all_h3_prompts(
                stage1_segments=self.segments if segments is None else segments,
                concept_prompts={"blocked": "A performer waits.", "valid": "A performer waits."},
                scene_details={}, global_context={}, mode="r2v",
                output_json_path=self.output, artifact_store=JsonArtifactStore(),
                checkpoint_store=H3PromptCheckpointStore(self.root),
                reuse_checkpoints=not replan,
                preserve_existing_aggregate=segments is not None,
            )
        return json.loads(self.output.read_text(encoding="utf-8")), fallback.call_count

    def assert_blocked_budget(self, result):
        self.assertEqual("blocked", result["readiness"]["status"])
        self.assertFalse(result["prompt_contract"]["valid"])
        self.assertIn("h3.fact.missing", result["readiness"]["reason_codes"])
        self.assertEqual(["generate", "repair", "fallback"],
                         [attempt["stage"] for attempt in result["readiness"]["attempts"]])

    def test_exhausted_scene_resumes_three_times_without_model_or_fallback_calls(self):
        results, fallbacks = self.run_batch()
        self.assert_blocked_budget(results[0])
        self.assertEqual("ready", results[1]["readiness"]["status"])
        self.assertTrue(results[1]["prompt_contract"]["valid"])
        self.assertEqual(3, len(self.planner.requests))
        self.assertEqual(1, fallbacks)
        original = deepcopy(results)
        for resume in range(3):
            with self.subTest(resume=resume):
                results, fallbacks = self.run_batch()
                self.assertEqual(3, len(self.planner.requests))
                self.assertEqual(0, fallbacks)
                self.assertEqual(original, results)

    def test_changed_timing_opens_one_budget_and_preserves_independent_scene(self):
        original, _ = self.run_batch()
        self.segments[0].update(end=3, duration=3)
        results, fallbacks = self.run_batch()
        self.assert_blocked_budget(results[0])
        self.assertEqual(5, len(self.planner.requests))
        self.assertEqual(1, fallbacks)
        self.assertNotEqual(original[0]["readiness"]["input_fingerprint"],
                            results[0]["readiness"]["input_fingerprint"])
        self.assertEqual(original[1], results[1])
        resumed, fallbacks = self.run_batch()
        self.assertEqual(results, resumed)
        self.assertEqual(5, len(self.planner.requests))
        self.assertEqual(0, fallbacks)

    def test_explicit_scene_replan_opens_one_revision_and_preserves_other_scene(self):
        original, _ = self.run_batch()
        results, fallbacks = self.run_batch(segments=[self.segments[0]], replan=True)
        self.assert_blocked_budget(results[0])
        self.assertEqual(5, len(self.planner.requests))
        self.assertEqual(1, fallbacks)
        self.assertEqual(original[0]["readiness"]["input_fingerprint"],
                         results[0]["readiness"]["input_fingerprint"])
        self.assertEqual(original[0]["readiness"]["attempt_revision"] + 1,
                         results[0]["readiness"]["attempt_revision"])
        self.assertEqual(original[1], results[1])
        resumed, fallbacks = self.run_batch()
        self.assertEqual(results, resumed)
        self.assertEqual(5, len(self.planner.requests))
        self.assertEqual(0, fallbacks)

    def test_override_is_semantically_checked_and_can_correct_blocked_scene(self):
        original, _ = self.run_batch()
        self.segments[0]["h3_prompt_override"] = original[1]["prompt"]
        rejected, fallbacks = self.run_batch()
        self.assertEqual("blocked", rejected[0]["readiness"]["status"])
        self.assertFalse(rejected[0]["prompt_contract"]["valid"])
        self.assertIn("h3.fact.missing", rejected[0]["readiness"]["reason_codes"])
        self.assertEqual(3, len(self.planner.requests))
        self.assertEqual(0, fallbacks)
        self.assertNotEqual(original[0]["readiness"]["input_fingerprint"],
                            rejected[0]["readiness"]["input_fingerprint"])
        self.segments[0]["h3_prompt_override"] = original[1]["prompt"].replace(
            "A performer folds a letter.",
            "A performer wearing a scarlet velvet cape folds a letter.",
        )
        corrected, fallbacks = self.run_batch()
        self.assertEqual("ready", corrected[0]["readiness"]["status"])
        self.assertTrue(corrected[0]["prompt_contract"]["valid"])
        self.assertEqual("user_override", corrected[0]["prompt_provenance"]["source"])
        self.assertEqual(3, len(self.planner.requests))
        self.assertEqual(0, fallbacks)
        self.assertEqual(original[1], corrected[1])
        resumed, fallbacks = self.run_batch()
        self.assertEqual(corrected, resumed)
        self.assertEqual(3, len(self.planner.requests))
        self.assertEqual(0, fallbacks)

    def test_override_refreshes_changed_cast_and_location_evidence(self):
        original, _ = self.run_batch()
        self.segments[1]["references"] = {"actor_ids": ["new-actor"], "location_id": "new-location"}
        self.segments[1]["h3_prompt_override"] = original[1]["prompt"]
        results, _ = self.run_batch()
        self.assertEqual(4, len(self.planner.requests))
        facts = results[1]["sections"]["facts"]["facts"]
        self.assertTrue(any(f["category"] == "cast" and f["value"] == "new-actor" for f in facts))
        self.assertTrue(any(f["category"] == "location" and f["value"] == "new-location" for f in facts))
        self.assertEqual(original[0], results[0])

    def test_valid_legacy_checkpoint_is_validated_without_model_calls(self):
        original, _ = self.run_batch()
        path = self.root / "output/render/scenes/scene_0002/h3_prompt.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        checkpoint["generated"].pop("readiness")
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        resumed, fallbacks = self.run_batch()
        self.assertEqual(3, len(self.planner.requests))
        self.assertEqual(0, fallbacks)
        self.assertEqual(original[1]["prompt"], resumed[1]["prompt"])
        self.assertEqual("ready", resumed[1]["readiness"]["status"])

    def test_compiler_contract_error_gets_one_corrective_attempt(self):
        from feverslop.prompting.deterministic_h3_compiler import DeterministicH3Compiler
        from feverslop.prompting.prompt_contract_validation import PromptContractError, PromptContractIssue
        original_compile = DeterministicH3Compiler.compile
        calls = []
        def compile_once_invalid(compiler, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise PromptContractError([PromptContractIssue("h3.test.invalid", "plan", "Repair plan")])
            return original_compile(compiler, **kwargs)
        with patch.object(DeterministicH3Compiler, "compile", compile_once_invalid):
            results, fallbacks = self.run_batch(segments=[self.segments[1]])
        self.assertEqual(2, len(self.planner.requests))
        self.assertEqual(0, fallbacks)
        self.assertEqual("ready", results[0]["readiness"]["status"])
        self.assertEqual(["generate", "repair"], [a["stage"] for a in results[0]["readiness"]["attempts"]])

    def test_blocked_checkpoint_reaches_render_and_finalization_gates(self):
        from feverslop.adapters.comfyui_minimax_h3_r2v_backend import ComfyUIMiniMaxH3R2VBackend
        from feverslop.adapters.comfyui_minimax_h3_t2v_backend import ComfyUIMiniMaxH3T2VBackend
        from feverslop.composition.stage_runners import _run_mux_original_audio_stage

        results, _ = self.run_batch()
        # A blocked scene must stop before accessing a backend or postprocessor.
        for backend_type in (ComfyUIMiniMaxH3R2VBackend, ComfyUIMiniMaxH3T2VBackend):
            with self.subTest(backend=backend_type.__name__):
                backend = object.__new__(backend_type)
                with self.assertRaisesRegex(ValueError, "incomplete; blocked scenes: blocked"):
                    backend._validate_scene(results[0])
        state = SimpleNamespace(plan_for_next_step=self.output)
        with self.assertRaisesRegex(ValueError, "incomplete; blocked scenes: blocked"):
            _run_mux_original_audio_stage(state)


if __name__ == "__main__":
    unittest.main()
