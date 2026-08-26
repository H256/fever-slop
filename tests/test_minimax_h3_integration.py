"""MiniMax H3 render-contract tests without a live ComfyUI instance."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_video import (
    RenderVideoScenesRequest,
    RenderVideoScenesUseCase,
)
from feverslop.composition.render_video import (
    RenderVideoCompositionOptions,
    build_render_video_scenes_use_case,
)


class FakeMiniMaxBackend:
    def __init__(self) -> None:
        self.requests = []

    def render_video(self, request):
        self.requests.append(request)
        output = request.output_dir / "final" / f"scene_{request.scene_number:04}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake rendered clip")
        return output


def _render_plan() -> list[dict]:
    return [
        {
            "scene": 1,
            "duration_seconds": 2.0,
            "fps": 24,
            "width": 672,
            "height": 1216,
            "frame_count": 48,
            "h3": {"prompt": "subject_definitions: Leo\nsummary: Leo runs."},
        },
        {
            "scene": 2,
            "duration_seconds": 3.0,
            "fps": 24,
            "width": 672,
            "height": 1216,
            "frame_count": 72,
            "h3": {"prompt": "subject_definitions: Leo\nsummary: Leo stops."},
        },
    ]


class MiniMaxH3RenderContractTests(unittest.TestCase):
    def test_r2v_composition_builds_the_production_use_case(self):
        options = RenderVideoCompositionOptions(
            app_config_path="app_config.json",
            workflow_path="workflows/video/minimax_h3/r2v_audio_v1.json",
            output_dir="test-output",
            video_pipeline="minimax-h3-r2v",
        )

        use_case = build_render_video_scenes_use_case(options)

        self.assertIsInstance(use_case, RenderVideoScenesUseCase)
        self.assertEqual("ComfyUIMiniMaxH3R2VBackend", type(use_case.backend).__name__)

    def test_t2v_composition_builds_the_production_use_case(self):
        options = RenderVideoCompositionOptions(
            app_config_path="app_config.json",
            workflow_path="workflows/video/minimax_h3/t2v.json",
            output_dir="test-output",
            video_pipeline="minimax-h3-t2v",
        )

        use_case = build_render_video_scenes_use_case(options)

        self.assertIsInstance(use_case, RenderVideoScenesUseCase)
        self.assertEqual("ComfyUIMiniMaxH3T2VBackend", type(use_case.backend).__name__)

    def test_r2v_render_delegates_each_scene_with_prompt_and_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(_render_plan()), encoding="utf-8")
            audio = temp / "song.wav"
            audio.write_bytes(b"audio fixture")
            backend = FakeMiniMaxBackend()

            rendered = RenderVideoScenesUseCase(
                backend=backend,
                artifact_store=JsonArtifactStore(),
            ).execute(
                RenderVideoScenesRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    audio_file=audio,
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "rendered",
                    upload_audio=False,
                ),
            )

            self.assertEqual(2, len(rendered))
            self.assertEqual(2, len(backend.requests))
            self.assertEqual(audio, backend.requests[0].audio_file)
            self.assertIn("Leo runs", backend.requests[0].prompt)
            self.assertEqual(2, backend.requests[1].scene_number)

    def test_render_reports_progress_after_each_completed_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(_render_plan()), encoding="utf-8")
            progress = []

            RenderVideoScenesUseCase(
                backend=FakeMiniMaxBackend(),
                artifact_store=JsonArtifactStore(),
            ).execute(
                RenderVideoScenesRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.wav",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "rendered",
                    upload_audio=False,
                    on_scene_complete=lambda output, completed, total: progress.append(
                        (output.name, completed, total),
                    ),
                ),
            )

            self.assertEqual(
                [("scene_0001.mp4", 1, 2), ("scene_0002.mp4", 2, 2)],
                progress,
            )

    def test_render_reuses_existing_valid_clip_without_backend_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(_render_plan()[:1]), encoding="utf-8")
            existing = temp / "rendered" / "final" / "scene_0001.mp4"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"existing rendered clip")
            backend = FakeMiniMaxBackend()

            rendered = RenderVideoScenesUseCase(
                backend=backend,
                artifact_store=JsonArtifactStore(),
            ).execute(
                RenderVideoScenesRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.wav",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "rendered",
                    upload_audio=False,
                ),
            )

            self.assertEqual([existing], rendered)
            self.assertEqual([], backend.requests)


if __name__ == "__main__":
    unittest.main()
