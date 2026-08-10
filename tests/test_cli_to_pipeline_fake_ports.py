"""Integration test: render-video use case → FakeVideoBackend through the real pipeline.

Exercises the real RenderVideoScenesUseCase and JsonArtifactStore with a
FakeVideoBackend so the JSON render plan is parsed, RenderRequest objects are
constructed, and the backend receives correct prompts — without needing a real
ComfyUI instance.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_video import RenderVideoScenesRequest, RenderVideoScenesUseCase
from feverslop.ports.rendering import VideoRenderRequest


# ---------------------------------------------------------------------------
# FakeVideoBackend (reused pattern from test_architecture_ports)
# ---------------------------------------------------------------------------


class FakeVideoBackend:
    """Minimal video backend that records render requests."""

    def __init__(self):
        self.requests: list[VideoRenderRequest] = []

    def render_video(self, request: VideoRenderRequest) -> Path:
        self.requests.append(request)
        output = request.output_dir / "final" / f"scene_{request.scene_number:04}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake mp4")
        return output


class ManifestBackfillBackend(FakeVideoBackend):
    def __init__(self):
        super().__init__()
        self.backfilled: list[int] = []

    def ensure_scene_manifest(self, request: VideoRenderRequest) -> None:
        self.backfilled.append(request.scene_number)
        manifest = request.output_dir / f"scene_{request.scene_number:04}" / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _render_plan_scenes() -> list[dict]:
    return [
        {
            "scene": 1,
            "abs_start_seconds": 0.0,
            "abs_end_seconds": 3.0,
            "duration_seconds": 3.0,
            "fps": 24,
            "width": 1280,
            "height": 704,
            "frame_count": 72,
            "z_image": {"prompt": "start frame prompt for scene 1"},
            "ltx": {
                "original_style_i2v_prompt": "video prompt scene 1",
                "render_mode_hint": "single_prompt",
            },
        },
        {
            "scene": 2,
            "abs_start_seconds": 3.0,
            "abs_end_seconds": 6.0,
            "duration_seconds": 3.0,
            "fps": 24,
            "width": 1280,
            "height": 704,
            "frame_count": 72,
            "z_image": {"prompt": "start frame prompt for scene 2"},
            "ltx": {
                "original_style_i2v_prompt": "video prompt scene 2",
                "render_mode_hint": "single_prompt",
            },
        },
    ]


def _project_config(data: dict | None = None) -> dict:
    base = {
        "project_name": "test-project",
        "input_audio": "input/song.mp3",
        "lyrics": "[Verse]\ntest",
        "video": {"fps": 24, "width": 1280, "height": 704},
        "story_idea": "test story",
        "style": "test style",
    }
    if data:
        base.update(data)
    return base


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class CLIToPipelineFakePortsTests(unittest.TestCase):
    """Verify RenderVideoScenesUseCase wires JsonArtifactStore and VideoRenderBackend."""

    def test_backend_receives_video_prompt_from_render_plan(self):
        """FakeVideoBackend receives the correct prompt derived from the render plan."""
        backend = FakeVideoBackend()
        store = JsonArtifactStore()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=store)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = _render_plan_scenes()
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(scenes), encoding="utf-8")

            rendered = use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "render",
                    render_mode="single_prompt",
                )
            )

            self.assertEqual(2, len(rendered))
            self.assertEqual(2, len(backend.requests))
            self.assertEqual("video prompt scene 1", backend.requests[0].prompt)
            self.assertEqual("video prompt scene 2", backend.requests[1].prompt)
            self.assertEqual(1, backend.requests[0].scene_number)
            self.assertEqual(2, backend.requests[1].scene_number)

    def test_backend_receives_scene_number_and_output_dir(self):
        """Each VideoRenderRequest contains the correct scene number and output dir."""
        backend = FakeVideoBackend()
        store = JsonArtifactStore()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=store)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = _render_plan_scenes()
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(scenes), encoding="utf-8")

            use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "output/ltx",
                    render_mode="single_prompt",
                )
            )

            output_dir = temp / "output/ltx"
            for req in backend.requests:
                self.assertEqual(output_dir, req.output_dir)
                self.assertTrue(req.output_dir.exists())

    def test_skip_existing_avoids_backend_call(self):
        """Pre-existing output files skip the backend render call."""
        backend = FakeVideoBackend()
        store = JsonArtifactStore()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=store)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = _render_plan_scenes()
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(scenes), encoding="utf-8")

            # Pre-create the first scene's output
            first_output = temp / "render/final/scene_0001.mp4"
            first_output.parent.mkdir(parents=True, exist_ok=True)
            first_output.write_bytes(b"existing")
            # Pre-create the second scene's output
            second_output = temp / "render/final/scene_0002.mp4"
            second_output.parent.mkdir(parents=True, exist_ok=True)
            second_output.write_bytes(b"existing")

            rendered = use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "render",
                    render_mode="single_prompt",
                    skip_existing=True,
                )
            )

            self.assertEqual(2, len(rendered))
            self.assertEqual(0, len(backend.requests))

    def test_skip_existing_recognises_per_scene_dir_output(self):
        """Pre-existing per-scene dir output (Minimax R2V/T2V) skips backend call."""
        backend = FakeVideoBackend()
        store = JsonArtifactStore()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=store)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = _render_plan_scenes()
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(scenes), encoding="utf-8")

            # Pre-create scene outputs in per-scene directory structure
            # (pattern used by ComfyUIMiniMaxH3R2VBackend and ComfyUIMiniMaxH3T2VBackend)
            for scene_num in (1, 2):
                scene_output = temp / "render" / f"scene_{scene_num:04}" / "final.mp4"
                scene_output.parent.mkdir(parents=True, exist_ok=True)
                scene_output.write_bytes(b"existing")

            rendered = use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "render",
                    render_mode="single_prompt",
                    skip_existing=True,
                )
            )

            self.assertEqual(2, len(rendered))
            self.assertEqual(0, len(backend.requests))
            self.assertEqual(
                temp / "render/scene_0001/final.mp4", rendered[0]
            )
            self.assertEqual(
                temp / "render/scene_0002/final.mp4", rendered[1]
            )

    def test_skip_existing_backfills_minimax_scene_manifests(self):
        backend = ManifestBackfillBackend()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=JsonArtifactStore())
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(_render_plan_scenes()[:1]), encoding="utf-8")
            scene_dir = temp / "render" / "scene_0001"
            scene_dir.mkdir(parents=True)
            (scene_dir / "final.mp4").write_bytes(b"existing")
            (scene_dir / "workflow.json").write_text("{}", encoding="utf-8")

            use_case.execute(RenderVideoScenesRequest(
                render_plan_path=plan_path,
                workflow_path=temp / "workflow.json",
                audio_file=temp / "song.mp3",
                storyboard_dir=temp / "storyboard",
                output_dir=temp / "render",
            ))

            self.assertEqual([1], backend.backfilled)
            self.assertTrue((scene_dir / "manifest.json").is_file())
            self.assertEqual([], backend.requests)

    def test_scene_selection_limits_backend_calls(self):
        """scene_numbers filters which scenes reach the backend."""
        backend = FakeVideoBackend()
        store = JsonArtifactStore()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=store)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = _render_plan_scenes()
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(scenes), encoding="utf-8")

            rendered = use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "render",
                    render_mode="single_prompt",
                    scene_numbers={2},
                )
            )

            self.assertEqual(1, len(rendered))
            self.assertEqual(1, len(backend.requests))
            self.assertEqual(2, backend.requests[0].scene_number)

    def test_render_mode_propagates_to_backend_request(self):
        """The render_mode from RenderVideoScenesRequest reaches the backend."""
        backend = FakeVideoBackend()
        store = JsonArtifactStore()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=store)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = _render_plan_scenes()
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(scenes), encoding="utf-8")

            use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "render",
                    render_mode="relay",
                )
            )

            self.assertEqual("relay", backend.requests[0].render_mode)

    def test_output_files_created_with_correct_names(self):
        """Backend and use case produce properly named output files."""
        backend = FakeVideoBackend()
        store = JsonArtifactStore()
        use_case = RenderVideoScenesUseCase(backend=backend, artifact_store=store)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = _render_plan_scenes()
            plan_path = temp / "render_plan.json"
            plan_path.write_text(json.dumps(scenes), encoding="utf-8")

            rendered = use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "render",
                    render_mode="single_prompt",
                )
            )

            expected = [
                temp / "render/final/scene_0001.mp4",
                temp / "render/final/scene_0002.mp4",
            ]
            self.assertEqual(expected, rendered)
            for path in rendered:
                self.assertTrue(path.exists())
                self.assertEqual(b"fake mp4", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
