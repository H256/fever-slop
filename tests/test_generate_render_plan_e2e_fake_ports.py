import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from feverslop.application.generate_render_plan import (
    GenerateRenderPlanRequest,
    GenerateRenderPlanUseCase,
)
from feverslop.composition.generate_render_plan import build_generate_render_plan_execution_request


class FakeArtifactStore:
    pass


class FakeService:
    def __init__(self, name, calls, apply):
        self.name = name
        self.calls = calls
        self.apply = apply

    def execute(self, context):
        self.calls.append(self.name)
        self.apply(context)
        return context


class ConsolePrintingFakeService(FakeService):
    def execute(self, context):
        context.console.print("scene pipeline status")
        return super().execute(context)


class GenerateRenderPlanE2EFakePortsTests(unittest.TestCase):
    def test_clamped_scene_duration_is_reported_in_english_before_services(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.wav"
            audio.write_bytes(b"dummy audio")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "demo",
                        "input_audio": "song.wav",
                        "video": {"fps": 24},
                        "scene_generation": {"min_duration": 2.0, "max_duration": 30.0},
                    }
                ),
                encoding="utf-8",
            )
            app_config_path = temp / "app_config.json"
            app_config_path.write_text(
                json.dumps(
                    {"comfyui": {"default_max_render_duration_seconds": 18.0}}
                ),
                encoding="utf-8",
            )
            events = []

            class Reporter:
                def panel(self, text, *, title=None):
                    events.append(("panel", title, text))

                def step(self, _title):
                    pass

                def file(self, _label, _path):
                    pass

                def message(self, _text):
                    pass

                def table(self, _title, _columns, _rows):
                    pass

                def run_progress(self, _description, func):
                    return func()

            service = FakeService(
                "service",
                events,
                lambda context: setattr(
                    context,
                    "render_plan",
                    [{"scene": 1, "frame_count": 24, "duration_seconds": 1.0}],
                ),
            )
            request = build_generate_render_plan_execution_request(
                GenerateRenderPlanRequest(
                    project_config_path=config_path,
                    app_config_path=app_config_path,
                    video_workflow_paths=(Path("workflows/video.json"),),
                )
            )

            GenerateRenderPlanUseCase(
                reporter=Reporter(),
                pipeline_services=[service],
                artifact_store=FakeArtifactStore(),
            ).execute(request)

            clamp_event = next(event for event in events if event[:2] == ("panel", "Scene duration limit"))
            self.assertLess(events.index(clamp_event), events.index("service"))
            self.assertIn("Requested scene duration: 2.000s..30.000s", clamp_event[2])
            self.assertIn("Effective scene duration: 2.000s..14.917s", clamp_event[2])
            self.assertIn(
                "Render limit: 18.000s including 50 pre-roll and 25 tail frames",
                clamp_event[2],
            )
            self.assertIn("Limiting workflow: video.json", clamp_event[2])

    def test_unclamped_scene_duration_does_not_report_limit_panel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.wav"
            audio.write_bytes(b"dummy audio")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "demo", "input_audio": "song.wav"}),
                encoding="utf-8",
            )
            app_config_path = temp / "app_config.json"
            app_config_path.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            service = FakeService(
                "service",
                [],
                lambda context: setattr(context, "render_plan", []),
            )

            GenerateRenderPlanUseCase(
                console=Console(file=output),
                pipeline_services=[service],
                artifact_store=FakeArtifactStore(),
            ).execute(
                build_generate_render_plan_execution_request(
                    GenerateRenderPlanRequest(config_path, app_config_path)
                )
            )

            self.assertNotIn("Scene duration limit", output.getvalue())

    def test_generate_render_plan_execute_can_run_with_fake_pipeline_services(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.wav"
            audio.write_bytes(b"dummy audio")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "demo",
                        "input_audio": "song.wav",
                        "video": {"fps": 24, "width": 1280, "height": 704},
                    }
                ),
                encoding="utf-8",
            )
            app_config_path = temp / "app_config.json"
            app_config_path.write_text("{}", encoding="utf-8")

            calls = []
            services = [
                FakeService(
                    "audio",
                    calls,
                    lambda context: setattr(context, "stem_files", {"vocals": temp / "vocals.wav"}),
                ),
                FakeService(
                    "scene",
                    calls,
                    lambda context: setattr(
                        context,
                        "stage1_segments",
                        [{"scene": 1, "duration_seconds": 1.0}],
                    ),
                ),
                FakeService(
                    "prompt",
                    calls,
                    lambda context: setattr(context, "scene_prompts_json", temp / "scene_prompts.json"),
                ),
                FakeService(
                    "render_plan",
                    calls,
                    lambda context: setattr(
                        context,
                        "render_plan",
                        [{"scene": 1, "frame_count": 24, "duration_seconds": 1.0}],
                    ),
                ),
            ]

            use_case = GenerateRenderPlanUseCase(
                console=Console(file=io.StringIO()),
                pipeline_services=services,
                artifact_store=FakeArtifactStore(),
            )

            result = use_case.execute(
                build_generate_render_plan_execution_request(
                    GenerateRenderPlanRequest(
                        project_config_path=config_path,
                        app_config_path=app_config_path,
                    )
                )
            )

            self.assertEqual(["audio", "scene", "prompt", "render_plan"], calls)
            self.assertEqual(1, result.scene_count)
            self.assertEqual(24, result.total_frames)
            self.assertEqual(1.0, result.total_duration_seconds)
            self.assertEqual(temp / "output" / "render" / "plans" / "base.json", result.render_plan_path)

    def test_generate_render_plan_context_provides_safe_console_without_rich_console(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.wav"
            audio.write_bytes(b"dummy audio")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "demo",
                        "input_audio": "song.wav",
                        "video": {"fps": 24, "width": 1280, "height": 704},
                    }
                ),
                encoding="utf-8",
            )
            app_config_path = temp / "app_config.json"
            app_config_path.write_text("{}", encoding="utf-8")

            services = [
                ConsolePrintingFakeService("scene", [], lambda context: None),
                FakeService(
                    "render_plan",
                    [],
                    lambda context: setattr(
                        context,
                        "render_plan",
                        [{"scene": 1, "frame_count": 24, "duration_seconds": 1.0}],
                    ),
                ),
            ]

            result = GenerateRenderPlanUseCase(
                pipeline_services=services,
                artifact_store=FakeArtifactStore(),
            ).execute(
                build_generate_render_plan_execution_request(
                    GenerateRenderPlanRequest(
                        project_config_path=config_path,
                        app_config_path=app_config_path,
                    )
                )
            )

            self.assertEqual(1, result.scene_count)


if __name__ == "__main__":
    unittest.main()
