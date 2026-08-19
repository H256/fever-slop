import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from feverslop.adapters.movie_visual import (
    PLACEHOLDER_FFMPEG_TIMEOUT_SECONDS,
    LocalMovieVisualAdapter,
    write_local_placeholder_clip,
)
from feverslop.errors import FeverSlopAdaptationError


class WriteLocalPlaceholderClipTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = Path(self._tmp.name) / "movieproj"
        self.project.mkdir()

    @patch("feverslop.adapters.movie_visual.subprocess.run")
    def test_placeholder_clip_passes_timeout_and_captures_stderr(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        path = self.project / "output" / "scene_0001.mp4"

        write_local_placeholder_clip(path)

        mock_run.assert_called_once()
        self.assertEqual(
            mock_run.call_args.kwargs,
            {
                "check": True,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
                "text": True,
                "timeout": PLACEHOLDER_FFMPEG_TIMEOUT_SECONDS,
            },
        )
        argv = mock_run.call_args.args[0]
        self.assertEqual(argv[0], "ffmpeg")
        self.assertIn("color=c=black:s=16x16:r=1:d=1.000", argv)
        self.assertEqual(argv[-1], str(path))

    @patch("feverslop.adapters.movie_visual.subprocess.run")
    def test_placeholder_clip_timeout_raises_adaptation_error(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(
            ["ffmpeg"], PLACEHOLDER_FFMPEG_TIMEOUT_SECONDS
        )
        path = self.project / "output" / "timeout.mp4"

        with self.assertRaises(FeverSlopAdaptationError) as ctx:
            write_local_placeholder_clip(path)

        message = str(ctx.exception)
        self.assertIn("timed out", message)
        self.assertIn(str(path), message)

    @patch("feverslop.adapters.movie_visual.subprocess.run")
    def test_placeholder_clip_failure_includes_stderr(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["ffmpeg"], stderr="lavfi: not found"
        )
        path = self.project / "output" / "failed.mp4"

        with self.assertRaises(FeverSlopAdaptationError) as ctx:
            write_local_placeholder_clip(path)

        message = str(ctx.exception)
        self.assertIn("lavfi: not found", message)
        self.assertIn(str(path), message)


class LocalMovieVisualAdapterRenderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = Path(self._tmp.name) / "movieproj"
        self.project.mkdir()

    def _write_render_plan(self) -> Path:
        plan_path = self.project / "render_plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "shots": [
                        {"scene": 1, "duration_seconds": 1.5},
                        {"scene": 2, "duration_seconds": 2.0},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return plan_path

    @patch("feverslop.adapters.movie_visual.subprocess.run")
    def test_local_adapter_render_movie_generates_scene_and_final_clips(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        plan_path = self._write_render_plan()

        result = LocalMovieVisualAdapter().render_movie(
            project_dir=self.project,
            render_plan_path=plan_path,
        )

        self.assertEqual(
            result,
            self.project / "output" / "movie" / f"{self.project.name}.mp4",
        )
        self.assertEqual(mock_run.call_count, 3)
        argvs = [call.args[0] for call in mock_run.call_args_list]
        self.assertIn("d=1.500", " ".join(argvs[0]))
        self.assertIn("d=2.000", " ".join(argvs[1]))

    @patch("feverslop.adapters.movie_visual.subprocess.run")
    def test_local_adapter_render_movie_surfaces_ffmpeg_timeout_as_adaptation_error(
        self,
        mock_run,
    ):
        mock_run.side_effect = subprocess.TimeoutExpired(
            ["ffmpeg"], PLACEHOLDER_FFMPEG_TIMEOUT_SECONDS
        )
        plan_path = self._write_render_plan()

        with self.assertRaises(FeverSlopAdaptationError):
            LocalMovieVisualAdapter().render_movie(
                project_dir=self.project,
                render_plan_path=plan_path,
            )


if __name__ == "__main__":
    unittest.main()
