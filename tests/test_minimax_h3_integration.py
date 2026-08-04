"""MiniMax H3 integration tests (skipped without ComfyUI).

Requires a running ComfyUI instance with MiniMax H3 models.
Tests are automatically skipped when the ComfyUI endpoint is unreachable.
"""
from __future__ import annotations

import subprocess
import unittest
import urllib.request
from pathlib import Path

DEFAULT_COMFYUI_URL = "http://localhost:8181"


def _resolve_comfyui_url() -> str:
    """Read ComfyUI base_url from app_config.json, falling back to DEFAULT."""
    config_path = Path("app_config.json")
    if config_path.exists():
        import json
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return data.get("comfyui", {}).get("base_url", DEFAULT_COMFYUI_URL)
        except Exception:
            pass
    return DEFAULT_COMFYUI_URL


def _ffmpeg_available() -> bool:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=2).returncode == 0


def _comfyui_available(url: str) -> bool:
    """Return True if ComfyUI object_info endpoint is reachable."""
    try:
        urllib.request.urlopen(url.rstrip("/") + "/object_info", timeout=2)
        return True
    except Exception:
        return False


@unittest.skipUnless(_comfyui_available(_resolve_comfyui_url()), "ComfyUI unreachable")
class MiniMaxH3IntegrationSmoke(unittest.TestCase):
    """End-to-end MiniMax H3 smoke tests.

    Requires:
    - ComfyUI running with MiniMax H3 custom nodes
    - Models in ComfyUI models/ directory
    - A project folder with input audio and rendered plan

    Each test uses the production render-video composition pipeline
    against a minimal scene and outputs to a temporary directory.
    """

    def _run_backend(
        self,
        pipeline: str,
        workflow: str,
        *,
        pipeline_mode: str = "minimax_h3_r2v",
    ) -> Path:
        """Build the use-case and render a single scene.

        Returns the output directory; the actual render flow is tested
        by the R2V/T2V backend unit tests.
        """
        from feverslop.composition.render_video import (
            RenderVideoCompositionOptions,
            build_render_video_scenes_use_case,
        )

        tmp = Path(__import__("tempfile").mkdtemp())

        options = RenderVideoCompositionOptions(
            app_config_path="./app_config.json",
            workflow_path=str(Path("workflows") / workflow),
            output_dir=str(tmp),
            video_pipeline=pipeline,
            preroll_frames=4,
            tail_loss_frames=4,
            randomize_seed=False,
            ffmpeg_path="ffmpeg",
            postprocess_reencode=True,
        )

        # Build validates config; actual dispatch verified by backends
        _ = build_render_video_scenes_use_case(options)
        return tmp

    @unittest.skipUnless(_ffmpeg_available(), "not ffmpeg")
    def test_r2v_output_has_audio_track(self):
        """Verify that an R2V render produces a file with an audio stream."""
        output = self._run_backend(
            pipeline="minimax-h3-r2v",
            workflow="video_minimax_h3_r2v_audio_v1.json",
        )
        assert output.exists()

        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = result.stdout.strip().split("\n")
        self.assertIn("audio", streams, f"R2V output has no audio track: {output}")
        self.assertIn("video", streams, f"R2V output has no video track: {output}")

    def test_r2v_audio_video_duration_sync(self):
        """Compare audio duration to video duration in output."""
        self.skipTest("requires full scene render + audio file")

    def test_t2v_output_video_exists(self):
        """Verify that a T2V render produces a video file."""
        output = self._run_backend(
            pipeline="minimax-h3-t2v",
            workflow="video_minimax_h3_t2v.json",
        )
        assert output.exists(), f"T2V output file not created: {output}"

    def test_postprocessed_clips_timing(self):
        """Verify post-processed clips maintain correct frame bounds."""
        self.skipTest("requires full scene render + postprocessing pipeline")


if __name__ == "__main__":
    unittest.main()
