import math
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.domain.srt import SrtScene
from feverslop.domain.scene_duration_limits import (
    resolve_scene_duration_policy,
    validate_render_frame_budget,
)
from feverslop.errors import FeverSlopValidationError
from feverslop.pipeline.scene_duration_enforcer import (
    enforce_scene_srt_file,
    parse_scene_srt,
    validate_scene_durations,
    write_scene_srt,
)


class SceneDurationLimitTests(unittest.TestCase):
    def resolve(self, **overrides):
        arguments = {
            "requested_min_seconds": 2.0,
            "requested_max_seconds": 30.0,
            "fps": 24,
            "preroll_frames": 50,
            "tail_frames": 25,
            "round_render_frames_to_8n1": True,
            "workflow_limits": {"video.json": 18.0},
            "workflow_paths": (Path("workflows/video.json"),),
            "default_max_render_duration_seconds": None,
        }
        arguments.update(overrides)
        return resolve_scene_duration_policy(**arguments)

    def test_clamps_24fps_original_profile_to_18_second_render_budget(self):
        result = self.resolve()

        self.assertEqual(433, result.max_render_frames)
        self.assertEqual(358, result.max_scene_frames)
        self.assertEqual(14.916, result.effective_max_seconds)
        self.assertEqual(2.0, result.effective_min_seconds)
        self.assertTrue(result.clamped)
        self.assertEqual("video.json", result.limiting_workflow)

    def test_effective_cap_survives_split_srt_write_and_read_without_an_extra_frame(self):
        policy = self.resolve()
        store = JsonArtifactStore()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_srt = temp / "input.srt"
            output_srt = temp / "output.srt"
            write_scene_srt(
                input_srt,
                [SrtScene(scene=1, start=0.0, end=29.833, text="Scene 1")],
                artifact_store=store,
            )

            enforce_scene_srt_file(
                input_srt=input_srt,
                output_srt=output_srt,
                min_duration=policy.effective_min_seconds,
                max_duration=policy.effective_max_seconds,
                artifact_store=store,
            )
            repaired = parse_scene_srt(output_srt)

        self.assertEqual(
            [],
            validate_scene_durations(
                repaired,
                min_duration=policy.effective_min_seconds,
                max_duration=policy.effective_max_seconds,
            ),
        )
        self.assertTrue(
            all(round(scene.duration * policy.fps) <= policy.max_scene_frames for scene in repaired)
        )

    def test_millisecond_cap_validation_is_stable_at_nonzero_srt_timestamps(self):
        policy = self.resolve()
        store = JsonArtifactStore()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scene.srt"
            write_scene_srt(
                path,
                [
                    SrtScene(
                        scene=1,
                        start=3600.0,
                        end=3600.0 + policy.effective_max_seconds,
                        text="Scene 1",
                    )
                ],
                artifact_store=store,
            )
            repaired = parse_scene_srt(path)

        self.assertEqual(
            [],
            validate_scene_durations(
                repaired,
                min_duration=policy.effective_min_seconds,
                max_duration=policy.effective_max_seconds,
            ),
        )
        self.assertLessEqual(round(repaired[0].duration * policy.fps), policy.max_scene_frames)

    def test_resolves_frame_exact_limits_at_16_and_50_fps(self):
        at_16 = self.resolve(fps=16)
        at_50 = self.resolve(fps=50)

        self.assertEqual((289, 214), (at_16.max_render_frames, at_16.max_scene_frames))
        self.assertAlmostEqual(13.375, at_16.effective_max_seconds)
        self.assertEqual((897, 822), (at_50.max_render_frames, at_50.max_scene_frames))
        self.assertAlmostEqual(16.44, at_50.effective_max_seconds)

    def test_does_not_round_render_budget_when_rounding_is_disabled(self):
        result = self.resolve(
            fps=50,
            preroll_frames=0,
            tail_frames=0,
            round_render_frames_to_8n1=False,
        )

        self.assertEqual(901, result.max_render_frames)
        self.assertEqual(901, result.max_scene_frames)

    def test_decimal_interval_boundary_does_not_lose_a_frame(self):
        result = self.resolve(
            requested_min_seconds=0.1,
            requested_max_seconds=1.0,
            fps=50,
            preroll_frames=0,
            tail_frames=0,
            round_render_frames_to_8n1=False,
            workflow_limits={"video.json": 0.58},
        )

        self.assertEqual(30, result.max_render_frames)
        self.assertEqual(30, result.max_scene_frames)

    def test_huge_finite_duration_does_not_leak_overflow_error(self):
        result = self.resolve(
            fps=50,
            preroll_frames=0,
            tail_frames=0,
            round_render_frames_to_8n1=False,
            workflow_limits={"video.json": 1e308},
        )

        self.assertGreater(result.max_render_frames, 0)
        validate_render_frame_budget(
            scene_number=1,
            render_frame_count=1,
            fps=result.fps,
            workflow_path="video.json",
            max_render_frames=result.max_render_frames,
            max_render_duration_seconds=result.max_render_duration_seconds,
        )

    def test_uses_strictest_limit_when_auto_mode_has_two_workflows(self):
        result = self.resolve(
            preroll_frames=0,
            tail_frames=0,
            round_render_frames_to_8n1=False,
            workflow_limits={"RELAY.JSON": 24.0, "single.json": 18.0},
            workflow_paths=(Path("elsewhere/relay.json"), Path("SINGLE.JSON")),
        )

        self.assertEqual("single.json", result.limiting_workflow)
        self.assertEqual(18.041, result.effective_max_seconds)

    def test_uses_default_for_workflow_without_explicit_override(self):
        result = self.resolve(
            workflow_limits={"other.json": 24.0},
            workflow_paths=(Path("unconfigured.json"),),
            default_max_render_duration_seconds=18.0,
        )

        self.assertEqual("unconfigured.json", result.limiting_workflow)
        self.assertEqual(18.0, result.max_render_duration_seconds)

    def test_clamps_minimum_when_cap_falls_below_requested_minimum(self):
        result = self.resolve(
            requested_min_seconds=20.0,
            workflow_limits={},
            workflow_paths=(),
            default_max_render_duration_seconds=18.0,
        )

        self.assertEqual(result.effective_max_seconds, result.effective_min_seconds)

    def test_preserves_requested_values_when_no_limit_is_configured(self):
        result = self.resolve(
            workflow_limits={},
            workflow_paths=(Path("unconfigured.json"),),
        )

        self.assertEqual((2.0, 30.0), (result.effective_min_seconds, result.effective_max_seconds))
        self.assertIsNone(result.max_render_frames)
        self.assertIsNone(result.max_scene_frames)
        self.assertFalse(result.clamped)

    def test_preserves_requested_maximum_when_it_is_below_cap(self):
        result = self.resolve(requested_max_seconds=10.0)

        self.assertEqual(10.0, result.effective_max_seconds)
        self.assertFalse(result.clamped)

    def test_rejects_invalid_fps_ranges_and_nonfinite_values(self):
        invalid_overrides = (
            {"fps": 0},
            {"fps": -1},
            {"requested_min_seconds": 31.0},
            {"requested_min_seconds": 0.0},
            {"requested_max_seconds": math.inf},
            {"default_max_render_duration_seconds": math.nan},
            {"workflow_limits": {"video.json": 0.0}},
        )

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(FeverSlopValidationError):
                    self.resolve(**overrides)

    def test_rejects_render_budget_that_cannot_fit_overhead_and_one_scene_frame(self):
        with self.assertRaisesRegex(FeverSlopValidationError, "cannot fit"):
            self.resolve(
                preroll_frames=50,
                tail_frames=25,
                workflow_limits={"video.json": 3.0},
            )

    def test_render_frame_guard_accepts_boundary(self):
        validate_render_frame_budget(
            scene_number=4,
            render_frame_count=433,
            fps=24,
            workflow_path=Path("workflows/video.json"),
            max_render_frames=433,
            max_render_duration_seconds=18.0,
        )

    def test_validate_scene_durations_float_consistency(self):
        """No ms rounding mismatch: boundary durations pass with float comparison."""
        min_duration = 2.0
        scene = SrtScene(scene=1, start=0.0005, end=min_duration + 0.0005)
        errors = validate_scene_durations(
            [scene],
            min_duration=min_duration,
            max_duration=30.0,
        )
        self.assertEqual([], errors)

    def test_render_frame_guard_rejects_over_budget_with_actionable_message(self):
        with self.assertRaises(FeverSlopValidationError) as raised:
            validate_render_frame_budget(
                scene_number=4,
                render_frame_count=457,
                fps=24,
                workflow_path=Path("workflows/video.json"),
                max_render_frames=433,
                max_render_duration_seconds=18.0,
            )

        self.assertEqual(
            "Scene 4 requires 457 render frames (19.000s), but video.json is limited "
            "to 433 frames (18.000s). Regenerate the render plan with the active workflow limit.",
            str(raised.exception),
        )

    def test_render_frame_guard_reports_allowed_seconds_from_frame_interval(self):
        with self.assertRaises(FeverSlopValidationError) as raised:
            validate_render_frame_budget(
                scene_number=2,
                render_frame_count=901,
                fps=50,
                workflow_path="rounded.json",
                max_render_frames=897,
                max_render_duration_seconds=18.0,
            )

        self.assertIn("901 render frames (18.000s)", str(raised.exception))
        self.assertIn("897 frames (17.920s)", str(raised.exception))

    def test_render_frame_guard_rejects_inconsistent_duration_and_frame_cap(self):
        with self.assertRaisesRegex(
            FeverSlopValidationError,
            "max_render_frames exceeds max_render_duration_seconds",
        ):
            validate_render_frame_budget(
                scene_number=2,
                render_frame_count=30,
                fps=50,
                workflow_path="video.json",
                max_render_frames=31,
                max_render_duration_seconds=0.58,
            )


if __name__ == "__main__":
    unittest.main()
