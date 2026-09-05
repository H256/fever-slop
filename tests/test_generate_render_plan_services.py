import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, sentinel

from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
from feverslop.application.generate_render_plan import (
    GenerateRenderPlanExecutionRequest,
    GenerateRenderPlanUseCase,
)
from feverslop.application.h3_prompt_pipeline import H3PromptPipeline
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


class DeferredRecordingService(RecordingService):
    defer_until_references = True


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
    def test_render_plan_builders_share_common_services_without_reordering(self):
        from feverslop.composition.generate_render_plan import (
            build_generate_render_plan_use_case,
            build_rebuild_render_plan_use_case,
        )

        common_services = [sentinel.scene, sentinel.prompt, sentinel.h3, sentinel.render]
        with patch(
            "feverslop.composition.generate_render_plan._common_pipeline_services",
            create=True,
            return_value=common_services,
        ) as common_pipeline_services:
            full_use_case = build_generate_render_plan_use_case()
            rebuild_use_case = build_rebuild_render_plan_use_case()

        self.assertEqual(2, common_pipeline_services.call_count)
        self.assertIsInstance(full_use_case.pipeline_services[0], AudioTimelinePipeline)
        self.assertEqual(common_services, full_use_case.pipeline_services[1:])
        self.assertEqual(common_services, rebuild_use_case.pipeline_services)

    def test_production_h3_service_has_project_checkpoint_store_factory(self):
        from feverslop.composition.generate_render_plan import build_generate_render_plan_use_case

        use_case = build_generate_render_plan_use_case()
        h3_service = next(
            service for service in use_case.pipeline_services
            if isinstance(service, H3PromptPipeline)
        )

        self.assertIsNotNone(h3_service.checkpoint_store_factory)

    def test_production_h3_service_enables_deterministic_fallback(self):
        from feverslop.composition.generate_render_plan import build_generate_render_plan_use_case

        use_case = build_generate_render_plan_use_case()
        h3_service = next(
            service for service in use_case.pipeline_services
            if isinstance(service, H3PromptPipeline)
        )
        with patch("feverslop.composition.generate_render_plan.build_dspy_generator", return_value=object()):
            builder = h3_service.dspy_prompt_builder_factory(object())

        self.assertTrue(builder.allow_fallback)

    def test_deferred_reference_services_are_skipped_when_requested(self):
        immediate = RecordingService("audio", {})
        deferred = DeferredRecordingService("h3", {})
        use_case = GenerateRenderPlanUseCase(pipeline_services=[immediate, deferred])

        result = use_case.execute_services({"request": SimpleNamespace(defer_h3_until_references=True)})

        self.assertEqual(["audio"], result["order"])
        self.assertEqual([], deferred.calls)
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
            config=SimpleNamespace(project_dir=Path("/portable/project")),
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
        self.assertEqual(Path("/portable/project"), calls[0]["project_dir"])
        self.assertEqual([{"scene": 1}], context.render_plan)

    def test_deferred_h3_prompts_are_skipped_until_reference_stages_finish(self):
        calls = []

        class Store:
            def read_json(self, path):
                return [{"scene": 1}]

        def build_render_plan(**kwargs):
            calls.append(kwargs)

        context = GenerateRenderPlanContext(
            request=SimpleNamespace(defer_h3_until_references=True),
            h3_prompts_json=Path("missing-h3-prompts.json"),
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

    def test_render_plan_pipeline_captures_regeneration_before_builder_and_injects_writer(self):
        events = []

        class Store:
            def read_json(self, _path):
                return [{"scene": 1}]

        class Regenerator:
            def write(self, path, scenes):
                events.append(("write", path, scenes))
                return Path(path)

        def factory(context):
            events.append(("capture", context["render_plan_json"]))
            return Regenerator()

        def build_render_plan(**kwargs):
            events.append(("build", kwargs["plan_writer"]))
            kwargs["plan_writer"](kwargs["output_json_file"], [{"scene": 1}])

        context = GenerateRenderPlanContext(
            config=SimpleNamespace(project_dir=Path("/portable/project")),
            scene_prompts_json=Path("scene_prompts.json"),
            ltx_prompt_relay_json=Path("relay.json"),
            render_plan_json=Path("base.json"),
            video_settings=SimpleNamespace(),
            artifact_store=Store(),
            log_step=lambda _title: None,
            log_file=lambda _label, _path: None,
        )

        RenderPlanPipeline(
            build_render_plan=build_render_plan,
            regenerator_factory=factory,
        ).execute(context)

        self.assertEqual("capture", events[0][0])
        self.assertEqual("build", events[1][0])
        self.assertEqual("write", events[2][0])

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
                ),
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
