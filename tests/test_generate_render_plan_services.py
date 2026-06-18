import unittest

from application.generate_render_plan import GenerateRenderPlanUseCase


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
