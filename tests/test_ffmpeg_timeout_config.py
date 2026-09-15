"""Config surface for the FFmpeg timeout (#857).

``ComfyUIConfig.ffmpeg_timeout_seconds`` is the single canonical knob; the
composition paths must thread it into the postprocessor so the value a user
writes in ``app_config.json`` actually reaches the FFmpeg subprocess.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class FFMPEGTimeoutConfigTest(unittest.TestCase):
    def _load(self, raw):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            return AppConfig.load(config_path)

    def test_default_is_600_when_unspecified(self):
        from feverslop.config.app_config import AppConfig
        from feverslop.domain.postprocessing import FFMPEG_TIMEOUT_SECONDS

        self.assertEqual(FFMPEG_TIMEOUT_SECONDS, 600.0)
        config = AppConfig.load(Path("/nonexistent/does-not-exist.json"))
        self.assertEqual(config.comfyui.ffmpeg_timeout_seconds, 600)

    def test_configured_value_round_trips(self):
        config = self._load({"comfyui": {"ffmpeg_timeout_seconds": 900}})
        self.assertEqual(config.comfyui.ffmpeg_timeout_seconds, 900)

    def test_invalid_timeout_is_rejected(self):
        from feverslop.config.app_config import AppConfig

        for raw in (0, -5, "soon"):
            with self.subTest(raw=raw):
                config_path = self._config_path({"comfyui": {"ffmpeg_timeout_seconds": raw}})
                with self.assertRaises(ValueError) as err:
                    AppConfig.load(config_path)
                self.assertIn("ffmpeg_timeout_seconds", str(err.exception))

    @staticmethod
    def _config_path(raw):
        import json as _json

        path = Path(tempfile.mkdtemp()) / "app_config.json"
        path.write_text(_json.dumps(raw), encoding="utf-8")
        return path


class FFMPEGTimeoutPropagationTest(unittest.TestCase):
    def test_video_postprocessor_respects_configured_timeout(self):
        from feverslop.adapters.video_postprocessor import VideoPostProcessor
        from feverslop.domain.postprocessing import FFMPEG_TIMEOUT_SECONDS

        processor = VideoPostProcessor(ffmpeg_timeout_seconds=900)
        self.assertEqual(processor.ffmpeg_timeout_seconds, 900)
        # Default (no explicit value) is the canonical 600 constant.
        self.assertEqual(
            VideoPostProcessor().ffmpeg_timeout_seconds, FFMPEG_TIMEOUT_SECONDS
        )

    def test_final_video_postprocessor_propagates_configured_timeout(self):
        from feverslop.adapters.video_postprocessor import final_video_postprocessor
        from feverslop.domain.postprocessing import FFMPEG_TIMEOUT_SECONDS

        self.assertEqual(final_video_postprocessor().ffmpeg_timeout_seconds, FFMPEG_TIMEOUT_SECONDS)
        self.assertEqual(
            final_video_postprocessor(timeout_seconds=42).ffmpeg_timeout_seconds, 42
        )

    def test_video_backend_threads_configured_timeout_into_postprocessor(self):
        from unittest.mock import MagicMock

        from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend

        backend = ComfyUIVideoRenderBackend(
            client=MagicMock(),
            ltx_workflow_path="workflows/video/ltx_25/i2v.json",
            output_dir="output",
            ffmpeg_timeout_seconds=900,
        )
        self.assertEqual(backend.postprocessor.ffmpeg_timeout_seconds, 900)

    def test_configured_timeout_reaches_the_ffmpeg_subprocess(self):
        import subprocess

        from unittest.mock import patch

        from feverslop.adapters.video_postprocessor import VideoPostProcessor

        processor = VideoPostProcessor(ffmpeg_path="ffmpeg", ffmpeg_timeout_seconds=900)
        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run, patch.object(
            VideoPostProcessor, "_validate_video_output"
        ):
            run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            processor.concat_clips(
                concat_list=Path("list.txt"),
                output_file=Path("out.mp4"),
            )
        self.assertEqual(run.call_args.kwargs["timeout"], 900)


if __name__ == "__main__":
    unittest.main()
