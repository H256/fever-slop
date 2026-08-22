import math
import unittest


class AuditDomainValidationTests(unittest.TestCase):
    def test_timeline_segment_rejects_invalid_bounds_and_non_finite_values(self):
        from feverslop.domain.timeline import TimelineSegment

        with self.assertRaises(ValueError):
            TimelineSegment(start=5.0, end=2.0, kind="vocals")
        with self.assertRaises(ValueError):
            TimelineSegment(start=math.nan, end=2.0, kind="vocals")

    def test_srt_timestamp_accepts_semicolon_millisecond_separator(self):
        from feverslop.domain.srt import parse_srt_timestamp

        self.assertAlmostEqual(61.5, parse_srt_timestamp("00:01:01;500"))


class MovieProjectValidationTests(unittest.TestCase):
    def _valid_project(self, **overrides):
        from feverslop.domain.movie import (
            CinematicShot,
            MovieBible,
            MovieProject,
            StoryArch,
        )

        story_arch = StoryArch(title="Test", premise="Test", beats=("beat",))
        bible = MovieBible(
            title="Test",
            premise="Test",
            story_arch=story_arch,
            actors=(),
            locations=(),
            continuity=(),
            style_constraints=(),
            runtime_constraints={},
        )
        kwargs = dict(
            slug="test-slug",
            name="Test Name",
            bible=bible,
            story_arch=story_arch,
            shots=(
                CinematicShot(
                    shot_id="s1",
                    description="Test shot",
                    duration_seconds=5.0,
                    camera="static",
                    action="test",
                    expression="neutral",
                    location="Test location",
                ),
            ),
            duration_seconds=12.0,
            width=1280,
            height=704,
            mode="scaffold",
            config=None,
        )
        kwargs.update(overrides)
        return MovieProject(**kwargs)

    def test_rejects_blank_slug(self):
        with self.assertRaisesRegex(ValueError, "slug"):
            self._valid_project(slug="   ")

    def test_rejects_blank_name(self):
        with self.assertRaisesRegex(ValueError, "name"):
            self._valid_project(name="")

    def test_rejects_empty_shots(self):
        with self.assertRaisesRegex(ValueError, "shots"):
            self._valid_project(shots=())

    def test_rejects_blank_mode(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            self._valid_project(mode=" ")

    def test_accepts_valid_project(self):
        project = self._valid_project(config={"a": 1})
        self.assertEqual(project.slug, "test-slug")
        self.assertEqual(len(project.shots), 1)
        self.assertEqual(project.config, {"a": 1})


class RenderSceneMissingKeyTests(unittest.TestCase):
    def test_missing_scene_key_raises_data_error(self):
        from feverslop.domain.render_plan import RenderScene
        from feverslop.errors import FeverSlopDataError

        with self.assertRaisesRegex(FeverSlopDataError, "'scene'"):
            RenderScene(data={}).scene_number

    def test_scene_number_reads_int(self):
        from feverslop.domain.render_plan import RenderScene

        self.assertEqual(RenderScene.from_dict({"scene": 3}).scene_number, 3)

    def test_from_dict_stays_lenient_without_scene(self):
        from feverslop.domain.render_plan import RenderScene

        scene = RenderScene.from_dict({})
        self.assertEqual(scene.to_dict(), {})


class ReferenceImageTypeTests(unittest.TestCase):
    def test_accepts_actor_and_location(self):
        from pathlib import Path

        from feverslop.domain.vision_references import ReferenceImage

        for reference_type in ("actor", "location"):
            ReferenceImage(id="ref-1", type=reference_type, path=Path("/tmp/ref.png"))

    def test_rejects_unknown_type(self):
        from pathlib import Path

        from feverslop.domain.vision_references import ReferenceImage

        with self.assertRaisesRegex(ValueError, "prop"):
            ReferenceImage(id="ref-1", type="prop", path=Path("/tmp/ref.png"))

    def test_rejects_empty_type(self):
        from pathlib import Path

        from feverslop.domain.vision_references import ReferenceImage

        with self.assertRaises(ValueError):
            ReferenceImage(id="ref-1", type="", path=Path("/tmp/ref.png"))

    def test_valid_reference_types_constant(self):
        from feverslop.domain.vision_references import VALID_REFERENCE_TYPES

        self.assertEqual(VALID_REFERENCE_TYPES, frozenset({"actor", "location"}))


class SrtSceneDurationTests(unittest.TestCase):
    def test_rejects_end_before_start(self):
        from feverslop.domain.srt import SrtScene

        with self.assertRaisesRegex(ValueError, "end must be"):
            SrtScene(scene=1, start=5.0, end=2.0, text="bad")

    def test_rejects_negative_start(self):
        from feverslop.domain.srt import SrtScene

        with self.assertRaisesRegex(ValueError, "start"):
            SrtScene(scene=1, start=-1.0, end=5.0, text="bad")

    def test_zero_duration_is_allowed(self):
        from feverslop.domain.srt import SrtScene

        scene = SrtScene(scene=1, start=2.0, end=2.0, text="zero")
        self.assertEqual(scene.duration, 0.0)

    def test_duration_is_end_minus_start(self):
        from feverslop.domain.srt import SrtScene

        scene = SrtScene(scene=1, start=1.0, end=4.5, text="span")
        self.assertEqual(scene.duration, 3.5)


if __name__ == "__main__":
    unittest.main()
