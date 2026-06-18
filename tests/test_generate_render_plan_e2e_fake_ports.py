import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from autoprompter.application.generate_render_plan import (
    GenerateRenderPlanRequest,
    GenerateRenderPlanUseCase,
)


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


class GenerateRenderPlanE2EFakePortsTests(unittest.TestCase):
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
                GenerateRenderPlanRequest(
                    project_config_path=config_path,
                    app_config_path=app_config_path,
                )
            )

            self.assertEqual(["audio", "scene", "prompt", "render_plan"], calls)
            self.assertEqual(1, result.scene_count)
            self.assertEqual(24, result.total_frames)
            self.assertEqual(1.0, result.total_duration_seconds)
            self.assertEqual(temp / "output" / "render" / "render_plan_song.json", result.render_plan_path)


if __name__ == "__main__":
    unittest.main()
