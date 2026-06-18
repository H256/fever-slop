import unittest
from pathlib import Path

from application.audio_timeline_pipeline import AudioTimelinePipeline
from application.generate_render_plan import GenerateRenderPlanUseCase
from application.pipeline_context import GenerateRenderPlanContext
from application.prompt_generation_pipeline import PromptGenerationPipeline
from application.render_plan_pipeline import RenderPlanPipeline
from application.scene_timeline_pipeline import SceneTimelinePipeline


class RecordingService:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = []

    def execute(self, context):
        self.calls.append(dict(context))
        context.setdefault("order", []).append(self.name)
        context.update(self.result)
        return context


class GenerateRenderPlanServiceTests(unittest.TestCase):
    def test_audio_timeline_pipeline_declares_required_context_keys(self):
        service = AudioTimelinePipeline()

        self.assertEqual(
            {"config", "paths", "song_id", "video_settings"},
            service.required_keys,
        )
        self.assertIn("timeline_json", service.produced_keys)
        self.assertIn("beat_json", service.produced_keys)
        self.assertIn("stem_files", service.produced_keys)

    def test_scene_prompt_and_render_plan_services_declare_context_contracts(self):
        self.assertIn("timeline_json", SceneTimelinePipeline.required_keys)
        self.assertIn("stage1_segments", SceneTimelinePipeline.produced_keys)
        self.assertIn("stage1_segments", PromptGenerationPipeline.required_keys)
        self.assertIn("scene_prompts_json", PromptGenerationPipeline.produced_keys)
        self.assertIn("scene_prompts_json", RenderPlanPipeline.required_keys)
        self.assertIn("render_plan", RenderPlanPipeline.produced_keys)

    def test_use_case_accepts_pipeline_services_and_runs_them_in_order(self):
        services = [
            RecordingService("audio", {"timeline_json": "timeline.json"}),
            RecordingService("scene", {"stage1_segments": [{"segment_id": "segment_001"}]}),
            RecordingService("prompts", {"scene_prompts_json": "scene_prompts.json"}),
            RecordingService("render_plan", {"render_plan": [{"scene": 1, "frame_count": 24, "duration_seconds": 1.0}]}),
        ]
        use_case = GenerateRenderPlanUseCase(pipeline_services=services)

        result_context = use_case.execute_services({"request": "fake"})

        self.assertEqual(["audio", "scene", "prompts", "render_plan"], result_context["order"])
        self.assertEqual("timeline.json", services[1].calls[0]["timeline_json"])
        self.assertEqual([{"scene": 1, "frame_count": 24, "duration_seconds": 1.0}], result_context["render_plan"])

    def test_pipeline_context_exposes_planned_artifact_paths(self):
        context = GenerateRenderPlanContext(
            song_id="demo",
            timeline_json=Path("timeline.json"),
            beat_json=Path("beat.json"),
            scene_srt=Path("scene.srt"),
            render_plan_json=Path("render_plan.json"),
        )

        self.assertEqual("demo", context.song_id)
        self.assertEqual(Path("render_plan.json"), context.render_plan_json)

    def test_default_use_case_exposes_four_pipeline_services(self):
        use_case = GenerateRenderPlanUseCase()

        service_names = [service.__class__.__name__ for service in use_case.pipeline_services]

        self.assertEqual(
            [
                "AudioTimelinePipeline",
                "SceneTimelinePipeline",
                "PromptGenerationPipeline",
                "RenderPlanPipeline",
            ],
            service_names,
        )


if __name__ == "__main__":
    unittest.main()
