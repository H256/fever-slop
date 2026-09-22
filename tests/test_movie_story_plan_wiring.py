import json
import tempfile
import unittest
from pathlib import Path

from feverslop.application.movie_artifacts import ensure_movie_bible
from feverslop.composition.movie_pipeline import _resolve_movie_story_plan
from feverslop.domain.movie import (
    story_plan_shot_durations,
    story_plan_to_cinematic_shots,
    story_plan_to_scene_cards,
    story_plan_to_shot_cards,
    story_plan_to_screenplay_scenes,
)
from feverslop.domain.story_plan import StoryMode
from feverslop.domain.story_plan_artifacts import (
    ArtifactClass,
    manifest_matches_story_plan,
    read_manifest,
    read_story_plan,
    story_plan_fingerprint,
)


def _write_render_plan(path: Path, shots: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "title": "Test Movie",
                "resolution": {"width": 1280, "height": 704},
                "duration_seconds": 30.0,
                "shots": shots,
            }
        ),
        encoding="utf-8",
    )


def _write_config(path: Path, actors: list[dict], locations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"actors": actors, "locations": locations}), encoding="utf-8")


class TestMovieStoryPlanWiring(unittest.TestCase):
    def _make_project(self, temp_dir: str) -> Path:
        project = Path(temp_dir)
        _write_render_plan(project / "movie" / "render_plan.json", [
            {
                "shot_id": "shot_0001",
                "description": "A witch watches the sky.",
                "actor_ids": ["witch"],
                "location_id": "tower",
                "duration_seconds": 5.0,
            },
            {
                "shot_id": "shot_0002",
                "description": "The witch walks down the stairs.",
                "actor_ids": ["witch"],
                "location_id": "tower",
                "duration_seconds": 10.0,
            },
            {
                "shot_id": "shot_0003",
                "description": "The witch meets the knight.",
                "actor_ids": ["witch", "knight"],
                "location_id": "gate",
                "duration_seconds": 15.0,
            },
        ])
        _write_config(
            project / "config.json",
            actors=[{"id": "witch", "name": "The Witch"}, {"id": "knight", "name": "The Knight"}],
            locations=[{"id": "tower", "name": "The Tower"}, {"id": "gate", "name": "The Gate"}],
        )
        return project

    def test_resolve_builds_authoritative_narrative_film_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._make_project(temp_dir)
            result_path = _resolve_movie_story_plan(project, project / "movie" / "render_plan.json")

            self.assertEqual(project / "movie" / "plan.json", result_path)
            self.assertTrue(result_path.is_file())
            self.assertTrue((project / "movie" / "plan.manifest.json").is_file())

            plan = read_story_plan(result_path)
            self.assertIs(StoryMode.narrative_film, plan.mode)
            # Segment targets come from the render plan shot IDs.
            self.assertEqual(["shot_0001", "shot_0002", "shot_0003"], [brief.target for brief in plan.segments])
            # The plan groups shots into beats by location (tower + gate).
            self.assertEqual(2, len(plan.beats))

            manifest = read_manifest(project / "movie" / "plan.manifest.json")
            self.assertIs(ArtifactClass.authoritative, manifest.artifact_class)
            self.assertTrue(manifest_matches_story_plan(manifest, plan))

    def test_tampered_plan_is_not_reused_when_manifest_fingerprint_differs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._make_project(temp_dir)
            render_plan_path = project / "movie" / "render_plan.json"
            plan_path = _resolve_movie_story_plan(project, render_plan_path)

            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["segments"][0]["visual_direction"] = "Tampered direction."
            plan_path.write_text(json.dumps(payload), encoding="utf-8")

            _resolve_movie_story_plan(project, render_plan_path)

            restored = read_story_plan(plan_path)
            self.assertEqual("A witch watches the sky.", restored.segments[0].visual_direction)

    def test_projection_manifest_records_the_canonical_plan_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._make_project(temp_dir)
            plan_path = _resolve_movie_story_plan(project, project / "movie" / "render_plan.json")
            plan = read_story_plan(plan_path)

            ensure_movie_bible(project)

            projection_manifest = read_manifest(project / "movie" / "bible.json.manifest.json")
            self.assertEqual(story_plan_fingerprint(plan), projection_manifest.plan_fingerprint)

    def test_legacy_shot_without_id_or_flat_references_remains_plannable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            _write_render_plan(project / "movie" / "render_plan.json", [
                {
                    "description": "The witch enters the tower.",
                    "reference_ids": {"actors": ["witch"], "location": "tower"},
                }
            ])
            _write_config(
                project / "config.json",
                actors=[{"id": "witch", "name": "The Witch"}],
                locations=[{"id": "tower", "name": "The Tower"}],
            )

            plan = read_story_plan(
                _resolve_movie_story_plan(project, project / "movie" / "render_plan.json")
            )

            self.assertEqual("shot_0001", plan.segments[0].target)
            self.assertEqual(["witch"], plan.segments[0].character_ids)
            self.assertEqual("tower", plan.segments[0].location_id)

    def test_stale_projection_manifest_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._make_project(temp_dir)
            plan_path = _resolve_movie_story_plan(project, project / "movie" / "render_plan.json")
            plan = read_story_plan(plan_path)
            ensure_movie_bible(project)
            manifest_path = project / "movie" / "bible.json.manifest.json"

            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["plan_fingerprint"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

            ensure_movie_bible(project)

            restored_manifest = read_manifest(manifest_path)
            self.assertEqual(story_plan_fingerprint(plan), restored_manifest.plan_fingerprint)

    def test_plan_is_reused_on_fingerprint_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._make_project(temp_dir)
            plan_path = project / "movie" / "plan.json"
            _resolve_movie_story_plan(project, project / "movie" / "render_plan.json")
            first_mtime = plan_path.stat().st_mtime_ns
            _resolve_movie_story_plan(project, project / "movie" / "render_plan.json")
            second_mtime = plan_path.stat().st_mtime_ns

            # No rewrite on the second call => the plan was reused.
            self.assertEqual(first_mtime, second_mtime)

    def test_plan_is_rebuilt_on_render_plan_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._make_project(temp_dir)
            render_plan_path = project / "movie" / "render_plan.json"
            plan_path = project / "movie" / "plan.json"
            _resolve_movie_story_plan(project, render_plan_path)
            first_targets = [brief.target for brief in read_story_plan(plan_path).segments]

            # Add a new shot to the render plan => fingerprint mismatch.
            data = json.loads(render_plan_path.read_text(encoding="utf-8"))
            data["shots"].append(
                {
                    "shot_id": "shot_0004",
                    "description": "The knight leaves.",
                    "actor_ids": ["knight"],
                    "location_id": "gate",
                    "duration_seconds": 5.0,
                }
            )
            render_plan_path.write_text(json.dumps(data), encoding="utf-8")
            _resolve_movie_story_plan(project, render_plan_path)
            second_targets = [brief.target for brief in read_story_plan(plan_path).segments]

            self.assertEqual(["shot_0001", "shot_0002", "shot_0003"], first_targets)
            self.assertEqual(["shot_0001", "shot_0002", "shot_0003", "shot_0004"], second_targets)

    def test_piece1_adapters_consume_the_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._make_project(temp_dir)
            render_plan_path = project / "movie" / "render_plan.json"
            plan_path = _resolve_movie_story_plan(project, render_plan_path)
            plan = read_story_plan(plan_path)
            render_plan = json.loads(render_plan_path.read_text(encoding="utf-8"))

            # Durations come from the render plan, keyed by the plan's shot IDs.
            self.assertEqual(
                {"shot_0001": 5.0, "shot_0002": 10.0, "shot_0003": 15.0},
                story_plan_shot_durations(plan, render_plan),
            )
            # Cinematic shots: plan shot IDs + render plan durations.
            shots = story_plan_to_cinematic_shots(plan, render_plan)
            self.assertEqual(["shot_0001", "shot_0002", "shot_0003"], [shot.shot_id for shot in shots])
            self.assertEqual([5.0, 10.0, 15.0], [shot.duration_seconds for shot in shots])
            # Scene cards: one per beat (grouped by location).
            self.assertEqual(2, len(story_plan_to_scene_cards(plan)))
            # Shot cards: one per segment.
            self.assertEqual(3, len(story_plan_to_shot_cards(plan)))
            # Screenplay scenes: one per beat.
            self.assertEqual(2, len(story_plan_to_screenplay_scenes(plan)))


if __name__ == "__main__":
    unittest.main()
