import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
from feverslop.application.generate_render_plan import GenerateRenderPlanExecutionRequest, GenerateRenderPlanUseCase
from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.application.render_plan_pipeline import RenderPlanPipeline
from feverslop.application.scene_timeline_pipeline import SceneTimelinePipeline
from feverslop.config.project_config import SceneGenerationConfig
from feverslop.domain.scene_duration_limits import ResolvedSceneDurationPolicy
from feverslop.scene_artifacts import SceneArtifactLayout


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


class RecordingReporter:
    def __init__(self):
        self.steps = []
        self.files = []
        self.messages = []
        self.progress = []

    def step(self, title):
        self.steps.append(title)

    def file(self, label, path):
        self.files.append((label, path))

    def message(self, text):
        self.messages.append(text)

    def table(self, title, columns, rows):
        self.messages.append((title, columns, rows))

    def panel(self, text, *, title=None):
        self.messages.append((title, text))

    def run_progress(self, description, func):
        self.progress.append(description)
        return func()


class GenerateRenderPlanServiceTests(unittest.TestCase):
    def test_clamp_report_uses_default_limit_label_without_selected_workflow(self):
        reporter = RecordingReporter()
        policy = ResolvedSceneDurationPolicy(
            requested_min_seconds=2.0,
            requested_max_seconds=30.0,
            effective_min_seconds=2.0,
            effective_max_seconds=14.916,
            max_render_duration_seconds=18.0,
            max_render_frames=433,
            max_scene_frames=358,
            fps=24,
            preroll_frames=50,
            tail_frames=25,
            limiting_workflow=None,
        )

        GenerateRenderPlanUseCase(reporter=reporter).report_scene_duration_clamp(policy)

        self.assertEqual("Scene duration limit", reporter.messages[0][0])
        self.assertIn("Limiting workflow: Default ComfyUI video limit", reporter.messages[0][1])

    def test_scene_timeline_pipeline_uses_effective_duration_policy_everywhere(self):
        received = {}

        class Generator:
            def generate_from_json_file(self, **_kwargs):
                return None

        def scene_generator_factory(scene_cfg):
            received["generator"] = (scene_cfg.min_duration, scene_cfg.max_duration)
            return Generator()

        def enforce_scene_srt_file(**kwargs):
            received["enforce"] = (kwargs["min_duration"], kwargs["max_duration"])

        def validate_scene_durations(_scenes, *, min_duration, max_duration):
            received["validate"] = (min_duration, max_duration)
            return []

        class Store:
            def read_json(self, _path):
                return []

        requested_scene_cfg = SceneGenerationConfig(min_duration=2.0, max_duration=30.0)
        policy = ResolvedSceneDurationPolicy(
            requested_min_seconds=2.0,
            requested_max_seconds=30.0,
            effective_min_seconds=2.0,
            effective_max_seconds=14.916,
            max_render_duration_seconds=18.0,
            max_render_frames=433,
            max_scene_frames=358,
            fps=24,
            preroll_frames=50,
            tail_frames=25,
            limiting_workflow="video.json",
        )
        context = GenerateRenderPlanContext(
            config=SimpleNamespace(scene_generation=requested_scene_cfg),
            video_settings=SimpleNamespace(fps=24),
            timeline_json=Path("timeline.json"),
            beat_json=Path("beat.json"),
            scene_srt_raw=Path("raw.srt"),
            scene_srt=Path("scenes.srt"),
            stage1_segments_json=Path("stage1.json"),
            ltx_prompt_relay_json=Path("relay.json"),
            scene_duration_policy=policy,
            artifact_store=Store(),
            log_step=lambda _title: None,
            log_file=lambda _label, _path: None,
            console=SimpleNamespace(print=lambda *_args, **_kwargs: None),
        )
        pipeline = SceneTimelinePipeline(
            scene_generator_factory=scene_generator_factory,
            enforce_scene_srt_file=enforce_scene_srt_file,
            parse_scene_srt=lambda _path: [],
            validate_scene_durations=validate_scene_durations,
            build_stage1_segment_json=lambda **_kwargs: None,
            build_scene_prompt_relay=lambda **_kwargs: None,
        )

        pipeline.execute(context)

        self.assertEqual((2.0, 14.916), received["generator"])
        self.assertEqual((2.0, 14.916), received["enforce"])
        self.assertEqual((2.0, 14.916), received["validate"])
        self.assertEqual(30.0, requested_scene_cfg.max_duration)

    def test_audio_timeline_pipeline_declares_required_context_keys(self):
        self.assertEqual(
            {"config", "paths", "song_id", "video_settings"},
            AudioTimelinePipeline.required_keys,
        )
        self.assertIn("timeline_json", AudioTimelinePipeline.produced_keys)
        self.assertIn("beat_json", AudioTimelinePipeline.produced_keys)
        self.assertIn("stem_files", AudioTimelinePipeline.produced_keys)

    def test_scene_prompt_and_render_plan_services_declare_context_contracts(self):
        self.assertIn("timeline_json", SceneTimelinePipeline.required_keys)
        self.assertIn("stage1_segments", SceneTimelinePipeline.produced_keys)
        self.assertIn("stage1_segments", PromptGenerationPipeline.required_keys)
        self.assertIn("scene_prompts_json", PromptGenerationPipeline.produced_keys)
        self.assertIn("scene_prompts_json", RenderPlanPipeline.required_keys)
        self.assertIn("render_plan", RenderPlanPipeline.produced_keys)

    def test_render_plan_pipeline_accepts_missing_optional_h3_prompts(self):
        calls = []

        class Store:
            def read_json(self, path):
                self.read_path = path
                return [{"scene": 1}]

        def build_render_plan(**kwargs):
            calls.append(kwargs)

        context = GenerateRenderPlanContext(
            scene_prompts_json=Path("scene_prompts.json"),
            ltx_prompt_relay_json=Path("relay.json"),
            render_plan_json=Path("render_plan.json"),
            video_settings=SimpleNamespace(),
            artifact_store=Store(),
            log_step=lambda _title: None,
            log_file=lambda _label, _path: None,
        )

        RenderPlanPipeline(build_render_plan=build_render_plan).execute(context)

        self.assertEqual(1, len(calls))
        self.assertIsNone(calls[0]["h3_prompts_json"])
        self.assertEqual([{"scene": 1}], context.render_plan)

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

    def test_default_use_case_starts_without_concrete_pipeline_services(self):
        use_case = GenerateRenderPlanUseCase()

        self.assertEqual([], use_case.pipeline_services)

    def test_use_case_default_helper_does_not_build_concrete_adapters(self):
        use_case = GenerateRenderPlanUseCase()

        self.assertEqual([], use_case.build_default_pipeline_services())

    def test_injected_services_are_used_without_building_default_pipeline(self):
        services = [RecordingService("only", {"render_plan": []})]
        use_case = GenerateRenderPlanUseCase(pipeline_services=services)

        self.assertIs(services, use_case.pipeline_services)

    def test_execute_uses_loaded_request_without_loading_config_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.wav"
            audio.write_bytes(b"dummy")
            reporter = RecordingReporter()
            services = [
                RecordingService(
                    "render_plan",
                    {"render_plan": [{"scene": 1, "frame_count": 24, "duration_seconds": 1.0}]},
                )
            ]
            config = SimpleNamespace(
                project_name="demo",
                input_audio=audio,
                output_dir=temp / "output",
            )
            paths = SimpleNamespace(
                ensure_output_dirs=lambda: None,
                timeline_dir=temp / "timeline",
                prompts_dir=temp / "prompts",
                render_dir=temp / "render",
                artifact_layout=SceneArtifactLayout(temp),
            )
            request = GenerateRenderPlanExecutionRequest(
                source_request=SimpleNamespace(render_storyboard=False, zimage_workflow_path=None),
                config=config,
                paths=paths,
                app_config=SimpleNamespace(llm=SimpleNamespace(model="fake", base_url="http://fake")),
                video_settings=SimpleNamespace(fps=24, width=1280, height=704),
                song_id="song",
            )
            use_case = GenerateRenderPlanUseCase(
                reporter=reporter,
                pipeline_services=services,
                artifact_store=object(),
            )

            result = use_case.execute(request)

            self.assertEqual(1, result.scene_count)
            self.assertEqual("song", services[0].calls[0]["song_id"])
            self.assertEqual(temp / "output" / "render" / "plans" / "base.json", result.render_plan_path)


if __name__ == "__main__":
    unittest.main()
