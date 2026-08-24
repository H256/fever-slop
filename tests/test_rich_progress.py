import unittest
import json
import tempfile
from pathlib import Path
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

    def test_storyboard_cli_uses_shared_progress_factory(self):
        import render_storyboard

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "render_plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            progress = Mock()
            progress.__enter__ = Mock(return_value=progress)
            progress.__exit__ = Mock(return_value=False)
            use_case = Mock()
            def execute(request):
                request.on_frame_complete(Path("scene.png"), 1, 1)
                return [Path("scene.png")]

            use_case.execute.side_effect = execute

            argv = [
                "render_storyboard.py",
                "--render-plan", str(plan),
                "--workflow", str(root / "workflow.json"),
                "--output-dir", str(root / "output"),
            ]
            with patch.object(render_storyboard, "build_progress", create=True, return_value=progress) as factory, \
                    patch.object(render_storyboard.AppConfig, "load", return_value=Mock()), \
                    patch.object(render_storyboard, "build_render_storyboard_use_case", return_value=use_case), \
                    patch("sys.argv", argv):
                render_storyboard.main()

            factory.assert_called_once_with(console=render_storyboard.console)
            progress.add_task.assert_called_once_with("Rendering storyboard startframes", total=1)

    def test_trim_cli_uses_shared_progress_factory(self):
        import tools.trim_existing_ltx_clips as trim_cli

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "render_plan.json"
            plan.write_text(json.dumps([{"scene": 1, "fps": 24, "frame_count": 12}]), encoding="utf-8")
            raw_dir = root / "raw"
            raw_dir.mkdir()
            raw_file = raw_dir / "scene_0001_raw.mp4"
            raw_file.write_bytes(b"")
            output_dir = root / "output"
            progress = Mock()
            progress.__enter__ = Mock(return_value=progress)
            progress.__exit__ = Mock(return_value=False)
            processor = Mock()
            processor.write_concat_list.return_value = output_dir / "concat_list.txt"

            argv = [
                "trim_existing_ltx_clips.py",
                "--render-plan", str(plan),
                "--raw-dir", str(raw_dir),
                "--output-dir", str(output_dir),
            ]
            with patch.object(trim_cli, "build_progress", create=True, return_value=progress) as factory, \
                    patch.object(trim_cli, "VideoPostProcessor", return_value=processor), \
                    patch("sys.argv", argv):
                trim_cli.main()

            factory.assert_called_once_with(console=trim_cli.console)
            progress.add_task.assert_called_once_with("Trimming clips", total=1)


if __name__ == "__main__":
    unittest.main()
