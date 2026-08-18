import unittest
from unittest.mock import Mock, patch

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from feverslop.utils.rich_progress import build_progress


class RichProgressTests(unittest.TestCase):
    def test_backend_neutral_video_scene_progress_label(self):
        from feverslop.composition import stage_runners

        self.assertEqual("Rendering video scenes", stage_runners.VIDEO_SCENE_PROGRESS_LABEL)

    def test_build_progress_uses_standard_pipeline_columns(self):
        progress = build_progress(console=Console(record=True))

        self.assertIsInstance(progress, Progress)
        self.assertEqual(
            [
                TextColumn,
                BarColumn,
                TextColumn,
                TaskProgressColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            ],
            [type(column) for column in progress.columns],
        )
        self.assertTrue(all(isinstance(column, ProgressColumn) for column in progress.columns))

    def test_classic_reporter_uses_shared_progress_factory(self):
        from feverslop.composition import stage_runners

        progress = Mock()
        with patch.object(stage_runners, "build_progress", create=True, return_value=progress) as factory:
            reporter = stage_runners.RenderProgressReporter("Rendering", 1)
            reporter.task_id = object()
            reporter.update(None, completed=1, total=1)

        factory.assert_called_once()
        progress.update.assert_called_once_with(reporter.task_id, completed=1)

    def test_movie_reporter_uses_shared_progress_factory(self):
        from feverslop.composition import movie_pipeline

        progress = Mock()
        with patch.object(movie_pipeline, "build_progress", create=True, return_value=progress) as factory:
            reporter = movie_pipeline.MovieStageProgressReporter({"Movie complete"})
            reporter.task_id = object()
            reporter.advance("Movie complete")

        factory.assert_called_once()
        progress.update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
