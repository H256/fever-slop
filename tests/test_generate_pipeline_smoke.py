import unittest

from feverslop.application.generate_render_plan import GenerateRenderPlanUseCase
from feverslop.application.pipeline_context import GenerateRenderPlanContext


class RecordingPipelineService:
    def __init__(self, name, updates):
        self.name = name
        self.updates = updates
        self.calls = []

    def execute(self, context):
        self.calls.append(context)
        context.setdefault("order", []).append(self.name)
        context.update(self.updates)
        return context


class GeneratePipelineSmokeTests(unittest.TestCase):
    def test_generate_pipeline_orchestration_runs_with_fake_services_only(self):
        services = [
            RecordingPipelineService("audio", {"stem_files": {"vocals": "vocals.wav"}}),
            RecordingPipelineService(
                "scene",
                {"stage1_segments": [{"segment_id": "segment_001", "scene": 1}]},
            ),
            RecordingPipelineService("prompts", {"scene_prompts_json": "scene_prompts.json"}),
            RecordingPipelineService(
                "render_plan",
                {
                    "render_plan": [
                        {
                            "scene": 1,
                            "frame_count": 24,
                            "duration_seconds": 1.0,
                        },
                    ],
                },
            ),
        ]
        use_case = GenerateRenderPlanUseCase(pipeline_services=services)
        context = GenerateRenderPlanContext()

        result = use_case.execute_services(context)

        self.assertEqual(["audio", "scene", "prompts", "render_plan"], result.order)
        self.assertEqual("vocals.wav", result.stem_files["vocals"])
        self.assertEqual("scene_prompts.json", result.scene_prompts_json)
        self.assertEqual(1, result.render_plan[0]["scene"])
        self.assertEqual(24, result.render_plan[0]["frame_count"])
        self.assertEqual(1.0, result.render_plan[0]["duration_seconds"])


if __name__ == "__main__":
    unittest.main()
