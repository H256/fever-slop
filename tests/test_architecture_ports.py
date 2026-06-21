import json
import inspect
import io
import tempfile
import unittest
from pathlib import Path

import main
from rich.console import Console
from feverslop.application.render_storyboard import RenderStoryboardRequest, RenderStoryboardUseCase
from feverslop.application.render_video import RenderVideoScenesRequest, RenderVideoScenesUseCase
from feverslop.application.generate_render_plan import GenerateRenderPlanRequest, GenerateRenderPlanUseCase
from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend, ComfyUIVideoRenderBackend
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.domain.render_plan import PromptSet, RenderPlan, RenderResult, RenderScene
from feverslop.ports.audio import AudioAnalyzerPort
from feverslop.ports.rendering import ImageRenderRequest, VideoRenderRequest
from feverslop.ports.workflow import WorkflowBackendPort
from feverslop.config.project_config import ProjectConfig, ProjectPaths


class FakeImageBackend:
    def __init__(self):
        self.requests = []

    def render_image(self, request: ImageRenderRequest) -> Path:
        self.requests.append(request)
        output = request.output_dir / f"scene_{request.scene_number:04}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake png")
        return output


class FakeVideoBackend:
    def __init__(self):
        self.requests = []

    def render_video(self, request: VideoRenderRequest) -> Path:
        self.requests.append(request)
        output = request.output_dir / "final" / f"scene_{request.scene_number:04}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake mp4")
        return output


class FakeAudioAnalyzer:
    def analyze(self, audio_file: Path) -> dict:
        return {"audio_file": str(audio_file)}


class FakeWorkflowBackend:
    def validate_workflow(self, workflow_path: Path, required_titles: list[str]) -> None:
        self.validated = (workflow_path, required_titles)


class ArchitecturePortsTests(unittest.TestCase):
    def _render_plan(self) -> list[dict]:
        return [
            {
                "scene": 1,
                "abs_start_seconds": 0.0,
                "abs_end_seconds": 2.0,
                "duration_seconds": 2.0,
                "fps": 24,
                "width": 1280,
                "height": 704,
                "frame_count": 48,
                "z_image": {"prompt": "start frame prompt"},
                "ltx": {
                    "original_style_i2v_prompt": "video prompt",
                    "render_mode_hint": "single_prompt",
                },
            }
        ]

    def test_project_config_load_does_not_create_output_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "demo", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual(temp / "song.mp3", config.input_audio)
            self.assertFalse((temp / "output").exists())

    def test_project_paths_match_existing_layout_without_creating_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "demo", "input_audio": "input/song.mp3"}),
                encoding="utf-8",
            )
            config = ProjectConfig.load(config_path)

            paths = ProjectPaths.from_config(config)

            self.assertEqual(temp / "output", paths.output_dir)
            self.assertEqual(temp / "output" / "stems", paths.stems_dir)
            self.assertEqual(temp / "output" / "timeline", paths.timeline_dir)
            self.assertEqual(temp / "output" / "prompts", paths.prompts_dir)
            self.assertEqual(temp / "output" / "render", paths.render_dir)
            self.assertFalse(paths.output_dir.exists())

    def test_json_artifact_store_round_trips_render_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "render_plan.json"
            store = JsonArtifactStore()
            plan = self._render_plan()

            store.write_render_plan(path, plan)

            self.assertEqual(plan, store.read_render_plan(path))

    def test_render_plan_dataclass_preserves_dict_shape(self):
        plan = RenderPlan.from_dicts(self._render_plan())

        self.assertIsInstance(plan.scenes[0], RenderScene)
        self.assertEqual(self._render_plan(), plan.to_dicts())

    def test_render_plan_selects_scenes_and_limit_without_mutating_shape(self):
        scenes = [
            {**self._render_plan()[0], "scene": 1},
            {**self._render_plan()[0], "scene": 2},
            {**self._render_plan()[0], "scene": 3},
        ]
        plan = RenderPlan.from_dicts(scenes)

        selected = plan.select(scene_numbers={2, 3}, limit=1)

        self.assertEqual([2], [scene.scene_number for scene in selected.scenes])
        self.assertEqual([2], [scene["scene"] for scene in selected.to_dicts()])

    def test_public_application_types_cover_prompt_and_render_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            request = GenerateRenderPlanRequest(
                project_config_path=temp / "config.json",
                app_config_path=temp / "app_config.json",
                concept_batch_size=10,
            )
            prompt_set = PromptSet(
                z_image_prompt="still prompt",
                i2v_prompt="video prompt",
            )
            result = RenderResult(scene_number=1, output_path=temp / "scene_0001.mp4")

            self.assertEqual(10, request.concept_batch_size)
            self.assertEqual("still prompt", prompt_set.z_image_prompt)
            self.assertEqual(temp / "scene_0001.mp4", result.output_path)
            self.assertEqual(
                {"scene": 1, "output_path": str(temp / "scene_0001.mp4")},
                result.as_manifest_entry(),
            )

    def test_audio_and_workflow_ports_are_structural(self):
        audio: AudioAnalyzerPort = FakeAudioAnalyzer()
        workflow: WorkflowBackendPort = FakeWorkflowBackend()
        path = Path("workflow.json")

        self.assertEqual({"audio_file": "song.mp3"}, audio.analyze(Path("song.mp3")))
        workflow.validate_workflow(path, ["#PROMPT"])
        self.assertEqual((path, ["#PROMPT"]), workflow.validated)

    def test_main_delegates_pipeline_to_generate_render_plan_use_case(self):
        self.assertTrue(hasattr(GenerateRenderPlanUseCase, "execute"))
        source = inspect.getsource(main.main)

        self.assertIn("build_generate_render_plan_use_case", source)
        self.assertIn(".execute(", source)

    def test_render_storyboard_cli_uses_composition_root(self):
        import render_storyboard

        source = inspect.getsource(render_storyboard.main)

        self.assertIn("build_render_storyboard_use_case", source)
        self.assertNotIn("ComfyUIImageBackend(", source)
        self.assertNotIn("JsonArtifactStore(", source)

    def test_storyboard_use_case_delegates_each_scene_to_image_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(self._render_plan()), encoding="utf-8")
            backend = FakeImageBackend()

            rendered = RenderStoryboardUseCase(
                backend=backend,
                artifact_store=JsonArtifactStore(),
            ).execute(
                RenderStoryboardRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    output_dir=temp / "storyboard",
                )
            )

            self.assertEqual([temp / "storyboard" / "scene_0001.png"], rendered)
            self.assertEqual(1, len(backend.requests))
            self.assertEqual("start frame prompt", backend.requests[0].prompt)

    def test_storyboard_use_case_reports_progress_after_each_available_frame(self):
        scenes = [
            {**self._render_plan()[0], "scene": 1},
            {**self._render_plan()[0], "scene": 2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(scenes), encoding="utf-8")
            progress = []

            rendered = RenderStoryboardUseCase(
                backend=FakeImageBackend(),
                artifact_store=JsonArtifactStore(),
            ).execute(
                RenderStoryboardRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    output_dir=temp / "storyboard",
                    on_frame_complete=lambda output, completed, total: progress.append(
                        (output.name, completed, total)
                    ),
                )
            )

            self.assertEqual(2, len(rendered))
            self.assertEqual(
                [("scene_0001.png", 1, 2), ("scene_0002.png", 2, 2)],
                progress,
            )

    def test_video_use_case_delegates_each_scene_to_video_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(self._render_plan()), encoding="utf-8")
            backend = FakeVideoBackend()

            rendered = RenderVideoScenesUseCase(
                backend=backend,
                artifact_store=JsonArtifactStore(),
            ).execute(
                RenderVideoScenesRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "ltx",
                    render_mode="single_prompt",
                )
            )

            self.assertEqual([temp / "ltx" / "final" / "scene_0001.mp4"], rendered)
            self.assertEqual(1, len(backend.requests))
            self.assertEqual("single_prompt", backend.requests[0].render_mode)
            self.assertEqual("video prompt", backend.requests[0].prompt)

    def test_video_use_case_reports_progress_after_each_available_scene(self):
        scenes = [
            {**self._render_plan()[0], "scene": 1},
            {**self._render_plan()[0], "scene": 2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(scenes), encoding="utf-8")
            progress = []

            rendered = RenderVideoScenesUseCase(
                backend=FakeVideoBackend(),
                artifact_store=JsonArtifactStore(),
            ).execute(
                RenderVideoScenesRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "ltx",
                    render_mode="single_prompt",
                    on_scene_complete=lambda output, completed, total: progress.append(
                        (output.name, completed, total)
                    ),
                )
            )

            self.assertEqual(2, len(rendered))
            self.assertEqual(
                [("scene_0001.mp4", 1, 2), ("scene_0002.mp4", 2, 2)],
                progress,
            )

    def test_video_use_case_prints_rich_status_when_console_is_provided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps(self._render_plan()), encoding="utf-8")
            console = Console(record=True, force_terminal=False, color_system=None, file=io.StringIO())

            RenderVideoScenesUseCase(
                backend=FakeVideoBackend(),
                artifact_store=JsonArtifactStore(),
                console=console,
            ).execute(
                RenderVideoScenesRequest(
                    render_plan_path=render_plan,
                    workflow_path=temp / "workflow.json",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp / "storyboard",
                    output_dir=temp / "ltx",
                    render_mode="single_prompt",
                )
            )

            output = console.export_text()
            self.assertIn("OK Rendered scene 1/1", output)
            self.assertIn("scene_0001.mp4", output)

    def test_comfy_image_backend_is_constructed_from_client_not_legacy_renderer(self):
        class FakeClient:
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = ComfyUIImageBackend(
                client=FakeClient(),
                workflow_path=Path(temp_dir) / "workflow.json",
                output_dir=Path(temp_dir) / "storyboard",
            )

            self.assertFalse(hasattr(backend, "renderer"))
            self.assertEqual(Path(temp_dir) / "workflow.json", backend.workflow_path)

    def test_comfy_video_backend_is_constructed_from_client_not_legacy_renderer(self):
        class FakeClient:
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = ComfyUIVideoRenderBackend(
                client=FakeClient(),
                ltx_workflow_path=Path(temp_dir) / "workflow.json",
                output_dir=Path(temp_dir) / "ltx",
            )

            self.assertFalse(hasattr(backend, "renderer"))
            self.assertEqual(Path(temp_dir) / "workflow.json", backend.ltx_workflow_path)

    def test_comfy_video_render_backend_is_primary_adapter_not_rendering_subclass(self):
        from feverslop.adapters.comfyui_video_backend import (
            ComfyUIVideoBackend,
            ComfyUIVideoRenderBackend,
        )

        self.assertTrue(issubclass(ComfyUIVideoBackend, ComfyUIVideoRenderBackend))
        self.assertFalse(issubclass(ComfyUIVideoRenderBackend, ComfyUIVideoBackend))
        self.assertEqual(
            "feverslop.adapters.comfyui_video_backend",
            ComfyUIVideoRenderBackend.__module__,
        )

    def test_ports_do_not_import_adapter_implementations(self):
        ports_dir = Path("src/feverslop/ports")
        forbidden = [
            "feverslop.adapters.",
            "video_postprocessor",
            "comfyui_client",
            "workflow_patcher",
        ]
        offenders = []
        for path in ports_dir.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
