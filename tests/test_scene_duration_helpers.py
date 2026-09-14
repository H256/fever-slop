import os
import tempfile
import unittest

from feverslop.domain.srt import SrtScene, parse_srt_text
from feverslop.pipeline.scene_duration_enforcer import (
    enforce_scene_duration_constraints,
    merge_remaining_short_scenes,
    merge_short_scenes,
    renumber_scenes,
    split_long_scenes,
    validate_scene_durations,
)


class SceneDurationHelperTests(unittest.TestCase):
    def test_parse_srt_text_is_filesystem_independent(self):
        blocks = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        self.assertEqual([(1, "Hello")], [(block.index, block.text) for block in blocks])

    def test_split_long_scenes_returns_legal_chunks(self):
        result = split_long_scenes([SrtScene(scene=7, start=0.0, end=9.0, text="A")], max_duration=4.0)

        self.assertEqual([7, 7, 7], [scene.scene for scene in result])
        self.assertTrue(all(scene.duration <= 4.0 for scene in result))
        self.assertEqual(0.0, result[0].start)
        self.assertEqual(9.0, result[-1].end)

    def test_merge_short_scenes_merges_tail_into_previous_scene(self):
        scenes = [
            SrtScene(scene=1, start=0.0, end=2.0, text="A"),
            SrtScene(scene=2, start=2.0, end=2.5, text="B"),
        ]

        result = merge_short_scenes(scenes, min_duration=1.0, max_duration=4.0)

        self.assertEqual(1, len(result))
        self.assertEqual(2.5, result[0].end)

    def test_renumber_scenes_rewrites_scene_numbers_and_empty_text(self):
        result = renumber_scenes([SrtScene(scene=99, start=0, end=1, text="")])

        self.assertEqual(1, result[0].scene)
        self.assertEqual("Scene 1", result[0].text)


class MergeRemainingShortScenesTests(unittest.TestCase):
    def test_merge_remaining_short_scenes_terminates_when_max_below_twice_min(self):
        """Terminates when min_duration=2, max_duration=3 and scenes are 1.5s each."""
        scenes = [
            SrtScene(scene=1, start=0.0, end=1.5, text="A"),
            SrtScene(scene=2, start=1.5, end=3.0, text="B"),
            SrtScene(scene=3, start=3.0, end=4.5, text="C"),
            SrtScene(scene=4, start=4.5, end=6.0, text="D"),
        ]
        # This should terminate quickly and return a result without hanging
        result = merge_remaining_short_scenes(
            scenes, min_duration=2.0, max_duration=3.0,
        )
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)

    def test_merge_remaining_short_scenes_normal_still_converges(self):
        """Normal inputs where merging fixes short scenes still work."""
        scenes = [
            SrtScene(scene=1, start=0.0, end=3.0, text="A"),
            SrtScene(scene=2, start=3.0, end=3.5, text="B"),
            SrtScene(scene=3, start=3.5, end=6.0, text="C"),
        ]
        result = merge_remaining_short_scenes(
            scenes, min_duration=2.0, max_duration=4.0,
        )
        self.assertTrue(all(s.duration >= 2.0 for s in result[:-1]))


class ValidateSceneDurationsContractTests(unittest.TestCase):
    """Round-trip contract: enforcer output must pass the validator.

    The validator must not raise false-positive "too short" errors for the
    enforcer's documented best-effort output when the constraints are
    infeasible (e.g. a clamped render budget that leaves no scene at or
    above min_duration). It must still reject genuine violations.
    """

    def test_short_single_scene_is_exempt(self):
        # The entire song is shorter than min_duration: a single short scene
        # is the enforcer's expected output and must be accepted.
        scenes = [SrtScene(scene=1, start=0.0, end=1.5, text="A")]
        self.assertEqual(
            [],
            validate_scene_durations(scenes, min_duration=2.0, max_duration=10.0),
        )

    def test_multi_scene_infeasible_tail_is_exempt(self):
        # Regression for the old ``is_last and len(scenes) == 1`` guard, which
        # only ever freed a single scene, so any multi-scene list with a short
        # scene was rejected even when the band was infeasible. When min == max
        # the enforcer cannot make every scene reach min_duration, so its
        # best-effort multi-scene output must be accepted.
        scenes = [
            SrtScene(scene=1, start=0.0, end=1.5, text="A"),
            SrtScene(scene=2, start=1.5, end=3.0, text="B"),
        ]
        self.assertEqual(
            [],
            validate_scene_durations(scenes, min_duration=2.0, max_duration=2.0),
        )

    def test_short_tail_in_feasible_band_is_rejected(self):
        # Under the feasibility model a short scene is only exempt when the
        # band is infeasible. A 0.8s tail with a 5.8s total is feasible (it
        # could have been one 5.8s scene), so the enforcer should have merged
        # it; flagging the short tail here is correct.
        scenes = [
            SrtScene(scene=1, start=0.0, end=5.0, text="A"),
            SrtScene(scene=2, start=5.0, end=5.8, text="B"),
        ]
        errors = validate_scene_durations(scenes, min_duration=2.0, max_duration=10.0)
        self.assertEqual(
            ["Scene 2 too short: 0.800s < 2.000s"],
            errors,
        )

    def test_short_middle_scene_is_rejected(self):
        # A short scene that is not the tail is a genuine enforcer violation.
        scenes = [
            SrtScene(scene=1, start=0.0, end=1.0, text="A"),
            SrtScene(scene=2, start=1.0, end=4.0, text="B"),
            SrtScene(scene=3, start=4.0, end=7.0, text="C"),
        ]
        errors = validate_scene_durations(scenes, min_duration=2.0, max_duration=10.0)
        self.assertEqual(
            ["Scene 1 too short: 1.000s < 2.000s"],
            errors,
        )

    def test_too_long_scene_is_rejected(self):
        scenes = [SrtScene(scene=1, start=0.0, end=15.0, text="A")]
        errors = validate_scene_durations(scenes, min_duration=2.0, max_duration=10.0)
        self.assertEqual(
            ["Scene 1 too long: 15.000s > 10.000s"],
            errors,
        )

    def test_allow_single_short_tail_false_rejects_tail(self):
        scenes = [
            SrtScene(scene=1, start=0.0, end=5.0, text="A"),
            SrtScene(scene=2, start=5.0, end=5.8, text="B"),
        ]
        errors = validate_scene_durations(
            scenes,
            min_duration=2.0,
            max_duration=10.0,
            allow_single_short_tail=False,
        )
        self.assertEqual(
            ["Scene 2 too short: 0.800s < 2.000s"],
            errors,
        )

    def test_infeasible_degenerate_band_does_not_raise_false_positive(self):
        # When min == max, no split can make every scene reach min_duration
        # (this is the clamped render-budget case in production). The enforcer
        # therefore emits short scenes; the validator must not flag them.
        # min == max == 2.0 with a 3.0s song -> enforcer produces two 1.5s
        # scenes, both below the 2.0s minimum.
        scenes = [
            SrtScene(scene=1, start=0.0, end=1.5, text="A"),
            SrtScene(scene=2, start=1.5, end=3.0, text="B"),
        ]
        self.assertEqual(
            [],
            validate_scene_durations(scenes, min_duration=2.0, max_duration=2.0),
        )

    def test_round_trip_enforcer_output_passes_validator(self):
        # Enforcer contract: its output must always validate, across a range
        # of realistic and clamped duration bands.
        cases = [
            # (scene durations, min_duration, max_duration)
            ([5.0, 5.0], 2.0, 10.0),
            ([5.0, 0.8], 2.0, 10.0),
            ([3.0, 1.0], 2.0, 10.0),
            ([1.5, 1.5], 2.0, 2.0),  # degenerate: infeasible band
            ([8.0, 2.0, 1.0], 2.0, 10.0),
            ([14.0, 14.0, 14.0], 14.0, 14.0),  # clamped near-degenerate
        ]
        for durations, min_duration, max_duration in cases:
            with self.subTest(durations=durations, min_duration=min_duration, max_duration=max_duration):
                start = 0.0
                scenes = []
                for index, duration in enumerate(durations, start=1):
                    scenes.append(SrtScene(scene=index, start=start, end=start + duration, text=f"S{index}"))
                    start += duration
                repaired = enforce_scene_duration_constraints(
                    scenes,
                    min_duration=min_duration,
                    max_duration=max_duration,
                )
                self.assertEqual(
                    [],
                    validate_scene_durations(
                        repaired,
                        min_duration=min_duration,
                        max_duration=max_duration,
                    ),
                )

    def test_enforcer_output_has_no_short_scene_when_band_is_feasible(self):
        # Enforcer contract: when the total coverage CAN be split into scenes
        # that all respect min_duration (a feasible band), the enforcer must
        # not leave any scene below min_duration. Covers the tight band
        # (max < 2*min) where the old merge/split pass over-split and
        # scattered short scenes remained, plus realistic bands.
        cases = [
            # (scene durations, min_duration, max_duration)
            ([2.0, 2.0, 1.6], 2.0, 3.0),
            ([2.0, 2.0, 1.8], 2.0, 3.0),
            ([2.0, 2.0, 2.0, 1.5], 2.0, 3.0),
            ([2.5, 2.5, 1.0], 2.0, 3.0),
            ([2.0, 1.9], 2.0, 4.0),
            ([4.0, 1.2], 2.0, 10.0),
        ]
        for durations, min_duration, max_duration in cases:
            with self.subTest(durations=durations, min_duration=min_duration, max_duration=max_duration):
                start = 0.0
                scenes = []
                for index, duration in enumerate(durations, start=1):
                    scenes.append(SrtScene(scene=index, start=start, end=start + duration, text=f"S{index}"))
                    start += duration
                repaired = enforce_scene_duration_constraints(
                    scenes,
                    min_duration=min_duration,
                    max_duration=max_duration,
                )
                # Every scene must respect both bounds in a feasible band.
                for scene in repaired:
                    self.assertGreaterEqual(scene.duration, min_duration - 1e-6, "short scene leaked")
                    self.assertLessEqual(scene.duration, max_duration + 1e-6, "long scene leaked")
                # Coverage must be preserved.
                self.assertAlmostEqual(scenes[0].start, repaired[0].start)
                self.assertAlmostEqual(scenes[-1].end, repaired[-1].end)


class SrtDomainTests(unittest.TestCase):
    def test_parse_srt_timestamp_basic(self):
        from feverslop.domain.srt import parse_srt_timestamp
        self.assertAlmostEqual(0.0, parse_srt_timestamp("00:00:00,000"))
        self.assertAlmostEqual(61.5, parse_srt_timestamp("00:01:01,500"))

    def test_format_srt_timestamp_roundtrip(self):
        from feverslop.domain.srt import format_srt_timestamp, parse_srt_timestamp
        original = "01:23:45,678"
        seconds = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(seconds)
        self.assertEqual(original, formatted)

    def test_parse_srt_blocks_empty_file(self):
        from feverslop.domain.srt import parse_srt_blocks
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        result = parse_srt_blocks(path)
        os.unlink(path)
        self.assertEqual([], result)

    def test_parse_srt_blocks_valid_content(self):
        from feverslop.domain.srt import parse_srt_blocks
        content = "1\n00:00:00,000 --> 00:00:02,500\nHello\n\n2\n00:00:02,500 --> 00:00:05,000\nWorld"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        result = parse_srt_blocks(path)
        os.unlink(path)
        self.assertEqual(2, len(result))
        self.assertEqual(1, result[0].index)
        self.assertAlmostEqual(0.0, result[0].start)
        self.assertEqual("Hello", result[0].text)


if __name__ == "__main__":
    unittest.main()
