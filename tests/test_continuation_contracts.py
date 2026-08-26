import unittest
from pathlib import Path

from feverslop.domain.audio_timing_contract import AudioTimingWindow, validate_audio_timing_windows
from feverslop.domain.continuation_intent import continuation_intents_from_plan
from feverslop.domain.cutless_boundaries import (
    CutlessBoundary,
    CutlessBoundaryDiagnostic,
    build_cutless_assembly_plan,
    validate_cutless_chain,
)
from feverslop.adapters.cutless_assembly import CutlessAssemblyService
from feverslop.prompting.dspy_h3_models import CreativeShotPayload, PromptJudgeResult, PromptPlan
from feverslop.prompting.creative_field_repair import repair_creative_fields, repair_creative_payloads


class ContinuationContractsTests(unittest.TestCase):
    def test_planner_intent_and_absolute_audio_windows(self):
        intent = continuation_intents_from_plan([{"action_id": "a", "requires_continuation": True}])[0]
        self.assertTrue(intent.requires_continuation)
        validate_audio_timing_windows([AudioTimingWindow(0, 4), AudioTimingWindow(4, 8)], song_duration=8)

    def test_cutless_chain_and_field_scoped_repair(self):
        digest = "a" * 64
        validate_cutless_chain([CutlessBoundary("s1", "s2", digest)], ["s1", "s2"])
        self.assertEqual({"camera": "pan", "action": "walk"}, repair_creative_fields(
            {"camera": "shake", "action": "walk"}, ["camera"], {"camera": "pan"}))

    def test_repairs_only_addressed_creative_payload_fields(self):
        payload = CreativeShotPayload(
            shot_id="shot-0001", visible_action="walks", performance="calm", camera_behavior="shake",
        )
        repaired = repair_creative_payloads(
            [payload],
            [{"shot_id": "shot-0001", "field": "camera_behavior", "issue_code": "camera.invalid", "repair_instruction": "Use a slow pan."}],
            {("shot-0001", "camera_behavior"): "slow pan"},
        )
        self.assertEqual("slow pan", repaired[0].camera_behavior)
        self.assertEqual("walks", repaired[0].visible_action)
        self.assertEqual("calm", repaired[0].performance)

    def test_judge_can_describe_field_addressable_issues(self):
        result = PromptJudgeResult(
            verdict="bad",
            field_issues=[{
                "shot_id": "shot-0001", "field": "camera_behavior",
                "issue_code": "camera.invalid", "repair_instruction": "Use a slow pan.",
            }],
        )
        self.assertEqual("shot-0001", result.field_issues[0].shot_id)
        self.assertEqual("camera.invalid", result.field_issues[0].issue_code)

    def test_cutless_plan_trims_only_proven_duplicate_boundary_frame(self):
        digest = "a" * 64
        boundaries = [CutlessBoundary("s1", "s2", digest)]
        diagnostics = [CutlessBoundaryDiagnostic(
            predecessor_segment_id="s1",
            successor_segment_id="s2",
            predecessor_last_frame_sha256=digest,
            successor_first_frame_sha256=digest,
            similarity=1.0,
            timing_delta_frames=0,
        )]

        plan = build_cutless_assembly_plan(
            ["s1", "s2"], boundaries, diagnostics, duplicate_policy="reject",
        )

        self.assertEqual(("s1", "s2"), plan.segment_ids)
        self.assertEqual(("s2",), plan.trim_first_frame_segments)
        self.assertEqual("accept", plan.outcome)
        self.assertFalse(plan.crossfade)

    def test_cutless_plan_policy_rejects_discontinuous_boundary(self):
        diagnostics = [CutlessBoundaryDiagnostic(
            predecessor_segment_id="s1",
            successor_segment_id="s2",
            predecessor_last_frame_sha256="a" * 64,
            successor_first_frame_sha256="b" * 64,
            similarity=0.2,
            timing_delta_frames=1,
        )]

        with self.assertRaisesRegex(ValueError, "cutless boundary rejected"):
            build_cutless_assembly_plan(
                ["s1", "s2"],
                [CutlessBoundary("s1", "s2", "a" * 64)],
                diagnostics,
                duplicate_policy="reject",
            )

    def test_cutless_plan_warns_without_mutating_source_segments(self):
        diagnostic = CutlessBoundaryDiagnostic(
            predecessor_segment_id="s1",
            successor_segment_id="s2",
            predecessor_last_frame_sha256="a" * 64,
            successor_first_frame_sha256="b" * 64,
            similarity=0.2,
            timing_delta_frames=1,
        )
        plan = build_cutless_assembly_plan(
            ["s1", "s2"], [CutlessBoundary("s1", "s2", "a" * 64)],
            [diagnostic], duplicate_policy="warn",
        )
        self.assertEqual("warn", plan.outcome)
        self.assertEqual((), plan.trim_first_frame_segments)
        self.assertEqual(("s1", "s2"), plan.segment_ids)

    def test_cutless_assembly_trims_derived_successor_and_hard_cuts(self):
        class FakePostprocessor:
            def __init__(self):
                self.trim_specs = []
                self.concat_args = None

            def trim_clip(self, spec):
                self.trim_specs.append(spec)
                return spec.output_file

            def write_concat_list(self, clips, output_file):
                self.written_clips = list(clips)
                self.concat_list = output_file
                return output_file

            def concat_clips(self, concat_list, output_file, **kwargs):
                self.concat_args = (concat_list, output_file, kwargs)
                return output_file

        digest = "a" * 64
        plan = build_cutless_assembly_plan(
            ["s1", "s2"], [CutlessBoundary("s1", "s2", digest)],
            [CutlessBoundaryDiagnostic("s1", "s2", digest, digest, 1.0, 0)],
        )
        postprocessor = FakePostprocessor()
        output = CutlessAssemblyService(postprocessor).assemble(
            {"s1": Path("s1.mp4"), "s2": Path("s2.mp4")}, plan,
            Path("movie.mp4"), segment_frame_counts={"s1": 10, "s2": 11},
        )

        self.assertEqual(Path("movie.mp4"), output)
        self.assertEqual(1, len(postprocessor.trim_specs))
        self.assertEqual(1, postprocessor.trim_specs[0].trim_front_frames)
        self.assertEqual(10, postprocessor.trim_specs[0].keep_frames)
        self.assertTrue(postprocessor.concat_args[2]["video_only"])
        self.assertTrue(postprocessor.concat_args[2]["reencode"])
        self.assertEqual(20, postprocessor.concat_args[2]["frame_count"])

    def test_prompt_plan_carries_semantic_continuation_intent(self):
        plan = PromptPlan(
            creative_intent="one continuous action",
            overall_soundscape="wind",
            music_intent="none",
            continuation_intents=[{
                "action_id": "raise-lantern",
                "requires_continuation": True,
                "desired_duration_seconds": 24.0,
            }],
        )

        intent = plan.continuation_intents[0]
        self.assertEqual("raise-lantern", intent.action_id)
        self.assertTrue(intent.requires_continuation)
        self.assertEqual(24.0, intent.desired_duration_seconds)

    def test_prompt_plan_preserves_explicit_hard_cut_as_non_continuation(self):
        plan = PromptPlan(
            creative_intent="two shots",
            overall_soundscape="silence",
            music_intent="none",
            shots=[{
                "shot_number": 1,
                "description": "The door opens.",
                "hard_cut_after": True,
            }],
            continuation_intents=[{
                "action_id": "door-open",
                "requires_continuation": False,
            }],
        )

        self.assertTrue(plan.shots[0].hard_cut_after)
        self.assertFalse(plan.continuation_intents[0].requires_continuation)


if __name__ == "__main__":
    unittest.main()
