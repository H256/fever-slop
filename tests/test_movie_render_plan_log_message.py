"""Regression test for issue 1312 (AUDIT APP-002).

``ensure_movie_render_plan_matches_bible`` only verifies that the render plan
file exists (the render plan is the canonical source; the bible is derived
from it). The caller in ``movie_pipeline._run`` therefore must not claim it is
"syncing" the render plan with the bible. This test drives the real caller
branch and asserts the emitted stage message is accurate.
"""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from feverslop.composition import movie_pipeline


def _write_render_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "title": "Test Movie",
                "resolution": {"width": 1280, "height": 704},
                "duration_seconds": 30.0,
                "shots": [
                    {
                        "shot_id": "shot_0001",
                        "description": "A witch watches the sky.",
                        "actor_ids": ["witch"],
                        "location_id": "tower",
                        "duration_seconds": 5.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class _CapturingReporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def step(self, title: str) -> None:
        self.messages.append(title)

    def file(self, label: str, path: Path) -> None:
        self.messages.append(f"{label}: {path}")

    def message(self, text: str) -> None:
        self.messages.append(text)

    def warning(self, text: str, *, title: str | None = None) -> None:
        self.messages.append(text)

    def panel(self, text: str, *, title: str | None = None) -> None:
        self.messages.append(text)


class TestMovieRenderPlanLogMessage(unittest.TestCase):
    def test_render_plan_stage_reports_existence_not_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            _write_render_plan(project / "movie" / "render_plan.json")
            (project / "movie" / "references").mkdir(parents=True, exist_ok=True)
            (project / "movie" / "references" / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            (project / "movie" / "bible.json").write_text(
                json.dumps({"actors": [], "locations": []}), encoding="utf-8"
            )

            args = movie_pipeline.build_movie_arg_parser().parse_args(
                [str(project), "--skip-movie-bible", "--skip-movie-references"]
            )
            config = {"movie_video_workflow": "msr", "reference_backend": "local"}

            reporter = _CapturingReporter()
            sentinel = object()

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        movie_pipeline, "_active_reporter", reporter
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        movie_pipeline, "_resolve_movie_story_plan",
                        return_value=project / "movie" / "plan.json",
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        movie_pipeline, "write_movie_reference_manifest_from_bible",
                        return_value=project / "movie" / "references" / "manifest.json",
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        movie_pipeline, "movie_references_ready", return_value=True
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        movie_pipeline, "_run_msr_workflow", return_value=sentinel
                    )
                )
                for name in (
                    "ensure_movie_story_design",
                    "ensure_movie_screenplay",
                    "ensure_movie_narrative_plan",
                    "ensure_movie_scene_cards",
                    "ensure_movie_shot_cards",
                    "ensure_movie_continuity_plan",
                ):
                    stack.enter_context(
                        mock.patch.object(
                            movie_pipeline, name,
                            return_value=project / "movie" / f"{name}.json",
                        )
                    )
                result = movie_pipeline._run(args, config)

        self.assertIs(result, sentinel)
        joined = "\n".join(reporter.messages)
        self.assertIn("ensuring render plan exists", joined)
        self.assertNotIn("syncing render plan with bible", joined)


if __name__ == "__main__":
    unittest.main()
