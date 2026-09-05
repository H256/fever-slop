import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rich.progress import TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

import run_pipeline
from feverslop.composition.arg_parser import PipelineStage
from feverslop.composition.stage_runners import (
    _discover_stem_files,
    _initial_render_plan,
    _preserve_enriched_reference_paths,
    _read_h3_input,
    _run_concat_video_only_stage,
    _run_main_pipeline_stage,
    _run_msr_references_stage,
    _run_msr_reference_sheets_stage,
    _run_mux_original_audio_stage,
    _run_render_plan_stage,
    _seed_reference_bindings,
    _selected_video_workflows,
)
from feverslop.config.project_config import (
    ActorConfig,
    ProjectConfig,
    StructuredLocationConfig,
)
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.domain.duration_capability import DurationCapability
from feverslop.scene_artifacts import SceneArtifactLayout


class RunPipelinePathTests(unittest.TestCase):
    def test_render_plan_stage_uses_one_regenerator_for_selection_and_reference_handoff(self):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            prompts = project / "output/prompts"
            prompts.mkdir(parents=True)
            paths = Namespace(
                prompts_dir=prompts,
                stems_dir=project / "output/stems",
            )
            config = Namespace(
                paths=paths,
                song_id="song",
                minimax_h3_audio_refs=Namespace(stems=()),
                input_audio=project / "song.wav",
                project_dir=project,
                max_scene_actors=4,
                video_pipeline="minimax-h3-r2v",
                to_video_settings=lambda: object(),
            )
            context = Namespace(
                project_config_path=project / "config.json",
                render_plan=project / "output/render/plans/base.json",
                reference_plan=project / "output/render/plans/references.json",
            )
            state = Namespace(
                args=Namespace(scenes="1", video_pipeline="minimax-h3-r2v"),
                context=context,
                app_config_path=project / "app-config.json",
                plan_for_next_step=None,
            )
            regenerator = Mock()
            regenerator.write = Mock()

            with patch("feverslop.composition.stage_runners.ProjectConfig.load", return_value=config), \
                patch("feverslop.composition.stage_runners.AppConfig.load", return_value=Mock(
                    resolve_video_workflow_profile=Mock(return_value=None),
                )), \
                patch("feverslop.composition.stage_runners.CanonicalPlanRegenerator", return_value=regenerator) as factory, \
                patch("feverslop.composition.stage_runners.build_render_plan") as builder:
                _run_render_plan_stage(state)

        factory.assert_called_once()
        self.assertEqual({1}, factory.call_args.kwargs["selected_scene_numbers"])
        self.assertEqual(context.reference_plan, factory.call_args.kwargs["reference_plan_path"])
        self.assertIs(regenerator.write, builder.call_args.kwargs["plan_writer"])
        self.assertEqual(context.render_plan, state.plan_for_next_step)

    def test_render_plan_stage_passes_selected_profile_duration_capability(self):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            prompts = project / "output/prompts"
            prompts.mkdir(parents=True)
            paths = Namespace(prompts_dir=prompts, stems_dir=project / "output/stems")
            config = Namespace(
                paths=paths,
                song_id="song",
                minimax_h3_audio_refs=Namespace(stems=()),
                input_audio=project / "song.wav",
                project_dir=project,
                max_scene_actors=4,
                video_pipeline="minimax-h3-r2v",
                to_video_settings=lambda: object(),
            )
            context = Namespace(
                project_config_path=project / "config.json",
                render_plan=project / "output/render/plans/base.json",
                reference_plan=project / "output/render/plans/references.json",
            )
            state = Namespace(
                args=Namespace(scenes=None, video_pipeline="minimax-h3-r2v", video_workflow_profile=None),
                context=context,
                app_config_path=project / "app-config.json",
                plan_for_next_step=None,
            )
            capability = DurationCapability.create(
                fps=24, min_seconds=2.0, max_seconds=3.0,
                preferred_seconds=3.0, frame_alignment=8, frame_offset=0,
            )
            profile = SimpleNamespace(duration_capability=capability)
            regenerator = Mock(write=Mock())
            with patch("feverslop.composition.stage_runners.ProjectConfig.load", return_value=config), \
                patch("feverslop.composition.stage_runners.AppConfig.load", return_value=Mock(
                    resolve_video_workflow_profile=Mock(return_value=profile),
                )), \
                patch("feverslop.composition.stage_runners.CanonicalPlanRegenerator", return_value=regenerator), \
                patch("feverslop.composition.stage_runners.build_render_plan") as builder:
                _run_render_plan_stage(state)

        self.assertIs(capability, builder.call_args.kwargs["duration_capability"])

    def test_deferred_h3_partial_resume_preserves_override_and_reference_bindings(self):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            prompts = project / "output/prompts"
            plans = project / "output/render/plans"
            prompts.mkdir(parents=True)
            plans.mkdir(parents=True)
            scene_prompt = {
                "scene": 1,
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "zimage_prompt": "new still",
                "t2i_prompt": "new base",
                "i2v_prompt_from_t2i": "new i2v",
                "segment_id": "segment-a",
                "type": "vocals",
            }
            relay = {
                "scene": 1,
                "prompt_relay": [{"frame_start": 0, "frame_end": 48, "state": "singing"}],
            }
            (prompts / "scene_prompts_song.json").write_text(json.dumps([scene_prompt]), encoding="utf-8")
            (prompts / "ltx_prompt_relay_song.json").write_text(json.dumps([relay]), encoding="utf-8")
            (prompts / "h3_prompts_song.json").write_text(
                json.dumps([{"segment_id": "segment-a", "prompt": "new judged h3"}]),
                encoding="utf-8",
            )
            canonical = build_canonical_scene(
                segment_id="segment-a",
                generated_roles={PromptRole.H3_VIDEO: "old h3"},
            )
            canonical["roles"][PromptRole.H3_VIDEO]["override"] = {
                "value": "human approved h3",
                "provenance": {"source": "human"},
            }
            existing = {"scene": 1, "canonical": canonical}
            base = plans / "base.json"
            base.write_text(json.dumps([existing]), encoding="utf-8")
            enriched = json.loads(json.dumps(existing))
            enriched["references"] = {"actor_msr_paths": ["actors/a.png"]}
            references = plans / "references.json"
            references.write_text(json.dumps([enriched]), encoding="utf-8")
            paths = Namespace(prompts_dir=prompts, stems_dir=project / "output/stems")
            config = Namespace(
                paths=paths,
                song_id="song",
                minimax_h3_audio_refs=Namespace(stems=()),
                input_audio=None,
                project_dir=project,
                max_scene_actors=4,
                video_pipeline="minimax-h3-r2v",
                to_video_settings=lambda: VideoSettings(width=1280, height=720, fps=24),
            )
            state = Namespace(
                args=Namespace(scenes=None, video_pipeline="minimax-h3-r2v"),
                context=Namespace(
                    project_config_path=project / "config.json",
                    render_plan=base,
                    reference_plan=references,
                ),
                app_config_path=project / "app-config.json",
                plan_for_next_step=None,
            )

            with patch("feverslop.composition.stage_runners.ProjectConfig.load", return_value=config):
                _run_render_plan_stage(state)

            saved = json.loads(base.read_text(encoding="utf-8"))[0]

        role = saved["canonical"]["roles"][PromptRole.H3_VIDEO]
        self.assertEqual("new judged h3", role["generated"]["value"])
        self.assertEqual("human approved h3", role["override"]["value"])
        self.assertEqual(["actors/a.png"], saved["references"]["actor_msr_paths"])

    def test_stem_discovery_requires_exact_input_audio_basename(self):
        with TemporaryDirectory() as temp_dir:
            stems_dir = Path(temp_dir) / "stems"
            stems_dir.mkdir()
            wrong = stems_dir / "vocals_midnight_stars.mp3"
            correct = stems_dir / "vocals_midnight_stars (4).wav"
            wrong.write_bytes(b"wrong")
            correct.write_bytes(b"correct")

            result = _discover_stem_files(
                stems_dir,
                Path(temp_dir) / "midnight_stars (4).mp3",
            )

        self.assertEqual(correct, result["vocals"])

    def test_seed_reference_bindings_assigns_actor_and_prompt_location(self):
        with TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "base.json"
            plan_path.write_text(
                json.dumps([{
                    "scene": 1,
                    "z_image": {"prompt": "A singer in the Necromantic Cathedral."},
                }]),
                encoding="utf-8",
            )
            config = ProjectConfig(
                project_dir=Path(temp_dir),
                project_name="test",
                input_audio=Path(temp_dir) / "song.wav",
                actors=(ActorConfig(id="singer", name="Singer"),),
                structured_locations=(
                    StructuredLocationConfig(id="mountain", name="Storm Mountain"),
                    StructuredLocationConfig(id="cathedral", name="Necromantic Cathedral"),
                ),
            )

            _seed_reference_bindings(plan_path, config)
            scene = json.loads(plan_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(["singer"], scene["references"]["actor_ids"])
        self.assertEqual("cathedral", scene["references"]["location_id"])

    def test_seed_reference_bindings_fills_unstructured_scenes_from_existing_bible(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan_path = temp / "base.json"
            plan_path.write_text(
                json.dumps([
                    {"scene": 1, "references": {"actor_ids": ["actor_cat_01"], "location_id": "loc_kitchen_dawn"}},
                    {"scene": 2, "z_image": {"prompt": "A sunlit kitchen in the morning."}, "references": {}},
                ]),
                encoding="utf-8",
            )
            locations_dir = temp / "output" / "references" / "locations" / "loc_kitchen_morning"
            locations_dir.mkdir(parents=True)
            (locations_dir / "manifest.json").write_text(
                json.dumps({"name": "Sunlit Kitchen", "visual_description": "warm morning kitchen"}),
                encoding="utf-8",
            )
            config = ProjectConfig(
                project_dir=temp,
                project_name="test",
                input_audio=temp / "song.wav",
            )

            warnings = _seed_reference_bindings(plan_path, config)
            scene = json.loads(plan_path.read_text(encoding="utf-8"))[1]

        self.assertEqual(["actor_cat_01"], scene["references"]["actor_ids"])
        self.assertEqual("loc_kitchen_morning", scene["references"]["location_id"])
        self.assertEqual(1, len(warnings))
        self.assertIn("structured reference serialization missing", warnings[0])

    def test_seed_reference_bindings_preserves_explicit_empty_actor_selection(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan_path = temp / "base.json"
            plan_path.write_text(
                json.dumps([{
                    "scene": 1,
                    "references": {"actor_ids": [], "location_id": "crowd"},
                }]),
                encoding="utf-8",
            )
            config = ProjectConfig(
                project_dir=temp,
                project_name="test",
                input_audio=temp / "song.wav",
                actors=(ActorConfig(id="singer", name="Singer"),),
                structured_locations=(
                    StructuredLocationConfig(id="crowd", name="Crowd"),
                ),
            )

            _seed_reference_bindings(plan_path, config)
            scene = json.loads(plan_path.read_text(encoding="utf-8"))[0]

        self.assertEqual([], scene["references"]["actor_ids"])

    def test_selected_pipeline_scenes_filters_render_plan(self):
        from feverslop.composition.stage_runners import _select_pipeline_scenes

        scenes = [{"scene": 1}, {"scene": 3}, {"scene": 5}]

        self.assertEqual([{"scene": 3}], _select_pipeline_scenes(scenes, "3"))
        self.assertEqual(scenes, _select_pipeline_scenes(scenes, None))

    @patch("feverslop.composition.stage_runners.VideoPostProcessor")
    def test_original_audio_mux_still_uses_video_only_concat(self, postprocessor_class):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            video_only = temp / "video_only.mp4"
            video_audio = temp / "video_audio.mp4"
            final = temp / "movie.mp4"
            input_audio = temp / "song.mp3"
            video_only.touch()
            video_audio.touch()

            state = Namespace(
                video_only_path=video_only,
                context=Namespace(
                    final_concat_video=video_only,
                    final_concat_video_audio=video_audio,
                    final_concat=final,
                    input_audio=input_audio,
                ),
            )

            _run_mux_original_audio_stage(state)

        postprocessor_class.return_value.mux_original_audio.assert_called_once_with(
            video_file=video_only,
            audio_file=input_audio,
            output_file=final,
        )

    @patch("feverslop.composition.stage_runners.VideoPostProcessor")
    def test_concat_and_mux_build_each_complete_scene_variant(self, postprocessor_class):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.base_plan.write_text(
                json.dumps([{"scene": 1}, {"scene": 2}]),
                encoding="utf-8",
            )
            for scene_number in (1, 2):
                layout.scene_dir(scene_number).mkdir(parents=True)
                layout.scene_final_video(scene_number).touch()
                layout.scene_final_facefix_video(scene_number).touch()
                layout.scene_upscaled_video(scene_number).touch()

            input_audio = project / "song.mp3"
            input_audio.touch()
            context = Namespace(
                artifact_layout=layout,
                ltx_dir=layout.scenes_dir,
                concat_list=layout.final_dir / "concat_list.txt",
                concat_raw=layout.concat_raw,
                final_concat_video=layout.video_only,
                final_concat=layout.movie,
                input_audio=input_audio,
            )
            state = Namespace(
                plan_for_next_step=layout.base_plan,
                context=context,
                video_only_path=None,
                final_video_path=None,
            )
            processor = postprocessor_class.return_value

            def create_output(*, output_file, **_kwargs):
                Path(output_file).touch()
                return Path(output_file)

            processor.concat_clips.side_effect = create_output
            processor.mux_original_audio.side_effect = create_output

            _run_concat_video_only_stage(state)
            _run_mux_original_audio_stage(state)

            video_only_outputs = {
                call.kwargs["output_file"]
                for call in processor.concat_clips.call_args_list
                if call.kwargs["video_only"]
            }
            self.assertEqual(
                {layout.video_only, layout.video_only_facefix, layout.video_only_upscaled},
                video_only_outputs,
            )
            mux_outputs = {
                call.kwargs["output_file"]
                for call in processor.mux_original_audio.call_args_list
            }
            self.assertEqual(
                {layout.movie, layout.movie_facefix, layout.movie_upscaled},
                mux_outputs,
            )
            self.assertEqual(layout.movie, state.final_video_path)

    @patch("feverslop.composition.stage_runners.VideoPostProcessor")
    def test_concat_skips_partial_upscaled_variant_instead_of_mixing_clips(self, postprocessor_class):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.base_plan.write_text(
                json.dumps([{"scene": 1}, {"scene": 2}]),
                encoding="utf-8",
            )
            for scene_number in (1, 2):
                layout.scene_dir(scene_number).mkdir(parents=True)
                layout.scene_final_video(scene_number).touch()
            layout.scene_upscaled_video(1).touch()
            layout.scene_final_facefix_video(1).touch()
            context = Namespace(
                artifact_layout=layout,
                ltx_dir=layout.scenes_dir,
                concat_list=layout.final_dir / "concat_list.txt",
                concat_raw=layout.concat_raw,
            )
            state = Namespace(
                plan_for_next_step=layout.base_plan,
                context=context,
                video_only_path=None,
            )
            processor = postprocessor_class.return_value
            processor.concat_clips.side_effect = lambda *, output_file, **_kwargs: Path(output_file)

            _run_concat_video_only_stage(state)

            video_only_outputs = [
                call.kwargs["output_file"]
                for call in processor.concat_clips.call_args_list
                if call.kwargs["video_only"]
            ]
            self.assertEqual([layout.video_only], video_only_outputs)

    @patch("feverslop.composition.stage_runners.VideoPostProcessor")
    def test_concat_uses_only_selected_scene(self, postprocessor_class):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.base_plan.write_text(
                json.dumps([{"scene": 1}, {"scene": 2}, {"scene": 3}]),
                encoding="utf-8",
            )
            layout.scene_dir(2).mkdir(parents=True)
            layout.scene_final_video(2).touch()
            context = Namespace(
                artifact_layout=layout,
                ltx_dir=layout.scenes_dir,
                concat_list=layout.final_dir / "concat_list.txt",
                concat_raw=layout.concat_raw,
            )
            state = Namespace(
                args=Namespace(scenes="2"),
                plan_for_next_step=layout.base_plan,
                context=context,
                video_only_path=None,
            )
            processor = postprocessor_class.return_value
            processor.concat_clips.side_effect = lambda *, output_file, **_kwargs: Path(output_file)

            _run_concat_video_only_stage(state)

            video_only_call = next(
                call for call in processor.concat_clips.call_args_list
                if call.kwargs["video_only"]
            )
            self.assertEqual(
                [layout.scene_final_video(2)],
                [Path(line.removeprefix("file '").removesuffix("'"))
                 for line in video_only_call.kwargs["concat_list"].read_text().splitlines()],
            )

    @patch("feverslop.composition.stage_runners.VideoPostProcessor")
    def test_concat_semantic_selection_uses_technical_scene_artifacts(self, postprocessor_class):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.base_plan.write_text(
                json.dumps([{"scene": 1}, {"scene": 2001001, "semantic_scene": 2, "technical_segment_id": "s2-1"}, {"scene": 3}]),
                encoding="utf-8",
            )
            layout.scene_dir(2001001).mkdir(parents=True)
            layout.scene_final_video(2001001).touch()
            context = Namespace(
                artifact_layout=layout,
                ltx_dir=layout.scenes_dir,
                concat_list=layout.final_dir / "concat_list.txt",
                concat_raw=layout.concat_raw,
            )
            state = Namespace(
                args=Namespace(scenes="2"),
                plan_for_next_step=layout.base_plan,
                context=context,
                video_only_path=None,
            )
            processor = postprocessor_class.return_value
            processor.concat_clips.side_effect = lambda *, output_file, **_kwargs: Path(output_file)

            _run_concat_video_only_stage(state)

            video_only_call = next(
                call for call in processor.concat_clips.call_args_list
                if call.kwargs["video_only"]
            )
            self.assertEqual(
                [layout.scene_final_video(2001001)],
                [Path(line.removeprefix("file '").removesuffix("'"))
                 for line in video_only_call.kwargs["concat_list"].read_text().splitlines()],
            )

    @patch("feverslop.composition.stage_runners.VideoPostProcessor")
    def test_concat_does_not_mix_canonical_and_legacy_base_clips(self, postprocessor_class):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.base_plan.write_text(
                json.dumps([{"scene": 1}, {"scene": 2}]),
                encoding="utf-8",
            )
            layout.scene_dir(1).mkdir(parents=True)
            layout.scene_final_video(1).touch()
            layout.scenes_dir.mkdir(parents=True, exist_ok=True)
            (layout.scenes_dir / "scene_0002.mp4").touch()
            state = Namespace(
                plan_for_next_step=layout.base_plan,
                context=Namespace(
                    artifact_layout=layout,
                    ltx_dir=layout.scenes_dir,
                    concat_list=layout.final_dir / "concat_list.txt",
                    concat_raw=layout.concat_raw,
                ),
                video_only_path=None,
            )

            with self.assertRaisesRegex(FileNotFoundError, "base variant"):
                _run_concat_video_only_stage(state)

        postprocessor_class.assert_not_called()

    def test_default_facefix_flow_still_runs_shared_base_concat(self):
        from feverslop.composition.stage_runners import resolve_pipeline_stages

        args = run_pipeline.build_arg_parser().parse_args(["--skip-tests", "--skip-ltx"])
        args.skip_facefix = False

        stages = resolve_pipeline_stages(args)

        self.assertNotIn(PipelineStage.FACEFIX_CONCAT, stages)
        self.assertLess(stages.index(PipelineStage.FACEFIX), stages.index(PipelineStage.CONCAT_VIDEO_ONLY))
        self.assertLess(
            stages.index(PipelineStage.CONCAT_VIDEO_ONLY),
            stages.index(PipelineStage.MUX_ORIGINAL_AUDIO),
        )

    @patch("feverslop.composition.stage_runners.VideoPostProcessor")
    def test_standalone_mux_ignores_stale_incomplete_optional_aggregate(self, postprocessor_class):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.final_dir.mkdir(parents=True)
            layout.base_plan.write_text(
                json.dumps([{"scene": 1}, {"scene": 2}]),
                encoding="utf-8",
            )
            layout.video_only.touch()
            layout.video_only_upscaled.touch()
            layout.scene_dir(1).mkdir(parents=True)
            layout.scene_upscaled_video(1).touch()
            input_audio = project / "song.mp3"
            input_audio.touch()
            state = Namespace(
                plan_for_next_step=layout.base_plan,
                context=Namespace(
                    artifact_layout=layout,
                    final_concat_video=layout.video_only,
                    final_concat=layout.movie,
                    input_audio=input_audio,
                ),
                video_only_path=None,
                final_video_path=None,
            )
            processor = postprocessor_class.return_value
            processor.mux_original_audio.side_effect = (
                lambda *, output_file, **_kwargs: Path(output_file)
            )

            _run_mux_original_audio_stage(state)

        processor.mux_original_audio.assert_called_once_with(
            video_file=layout.video_only,
            audio_file=input_audio,
            output_file=layout.movie,
        )

    def test_minimax_r2v_prefers_base_plan_with_h3_prompts(self):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.references_plan.write_text(
                json.dumps([{"scene": 1, "references": {"actor_msr_paths": ["actor.png"]}}]),
                encoding="utf-8",
            )
            layout.base_plan.write_text(
                json.dumps([{"scene": 1, "h3": {"prompt": "H3 prompt"}}]),
                encoding="utf-8",
            )
            context = Namespace(
                artifact_layout=layout,
                render_dir=layout.render_dir,
                song_id="song",
                reference_plan=layout.references_plan,
                ingredients_plan=layout.ingredients_plan,
                render_plan=layout.base_plan,
            )
            args = Namespace(video_pipeline="minimax-h3-r2v")

            selected = _initial_render_plan(context, args, [])

        self.assertEqual(layout.base_plan, selected)

    def test_ingredients_llm_phase_starts_from_reference_plan(self):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.base_plan.write_text(json.dumps([{"scene": 1, "source": "base"}]), encoding="utf-8")
            layout.references_plan.write_text(
                json.dumps([{"scene": 1, "source": "references"}]),
                encoding="utf-8",
            )
            context = Namespace(
                artifact_layout=layout,
                render_dir=layout.render_dir,
                song_id="song",
                reference_plan=layout.references_plan,
                ingredients_plan=layout.ingredients_plan,
                render_plan=layout.base_plan,
            )
            args = Namespace(video_pipeline="ltx_ingredients")

            selected = _initial_render_plan(
                context,
                args,
                [PipelineStage.MSR_PROMPT_ENRICH, PipelineStage.INGREDIENTS_SHEETS],
            )

        self.assertEqual(layout.references_plan, selected)

    def test_openshot_export_stage_reuses_existing_plan_without_upstream_stages(self):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            layout.references_plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            context = Namespace(
                artifact_layout=layout,
                render_dir=layout.render_dir,
                song_id="song",
                reference_plan=layout.references_plan,
                ingredients_plan=layout.ingredients_plan,
                render_plan=layout.base_plan,
                anchored_plan=layout.anchored_plan,
                compact_plan=layout.compact_plan,
            )
            args = Namespace(video_pipeline="ltx_i2v")

            selected = _initial_render_plan(context, args, [PipelineStage.OPENSHOT_EXPORT])

        self.assertEqual(layout.references_plan, selected)

    def test_timeline_export_stage_accepts_mlt_and_openshot_formats(self):
        from feverslop.composition.arg_parser import build_arg_parser
        from feverslop.composition.stage_runners import resolve_pipeline_stages

        parser = build_arg_parser()
        mlt_args = parser.parse_args(["--stage", "export_timeline", "--format", "mlt"])
        openshot_args = parser.parse_args(["--stage", "export_timeline", "--format", "openshot"])
        default_args = parser.parse_args(["--stage", "export_timeline"])

        self.assertEqual(resolve_pipeline_stages(mlt_args), [PipelineStage.EXPORT_TIMELINE])
        self.assertEqual(mlt_args.timeline_format, "mlt")
        self.assertEqual(openshot_args.timeline_format, "openshot")
        self.assertEqual(default_args.timeline_format, "both")

    @patch("feverslop.composition.stage_runners._run_render_plan_stage")
    @patch("feverslop.composition.stage_runners.enrich_render_plan_with_reference_sheets")
    def test_reference_sheets_create_missing_intermediate_render_plan(
        self,
        enrich_render_plan,
        run_render_plan,
    ):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan = temp / "render" / "plans" / "base.json"
            plan.parent.mkdir(parents=True)
            references_dir = temp / "references"
            references_dir.mkdir()
            reference_plan = temp / "render" / "plans" / "references.json"
            reference_plan.write_text("[]", encoding="utf-8")
            plan.write_text("[]", encoding="utf-8")

            state = Namespace(
                args=Namespace(video_pipeline="minimax-h3-r2v"),
                plan_for_next_step=plan,
                context=Namespace(
                    artifact_layout=Namespace(plans_dir=plan.parent),
                    references_dir=references_dir,
                    reference_plan=reference_plan,
                ),
            )

            plan.unlink()
            run_render_plan.side_effect = lambda current_state: plan.write_text("[]", encoding="utf-8")
            _run_msr_reference_sheets_stage(state)

        run_render_plan.assert_called_once_with(state)
        enrich_render_plan.assert_called_once()

    @patch("feverslop.composition.stage_runners.render_reference_bible")
    def test_msr_references_reuses_complete_existing_manifests(self, render_reference_bible):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            config_path = project / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.mp3",
                "video_pipeline": "minimax-h3-r2v",
                "actors": [{"id": "actor_01", "name": "Actor"}],
                "locations": [{"id": "loc_01", "name": "Location"}],
            }), encoding="utf-8")
            references_dir = project / "output" / "references"
            for kind, identifier in (("actors", "actor_01"), ("locations", "loc_01")):
                reference_dir = references_dir / kind / identifier
                reference_dir.mkdir(parents=True)
                (reference_dir / "sheet.png").write_bytes(b"sheet")
                (reference_dir / "manifest.json").write_text(json.dumps({
                    "sheet_path": "sheet.png",
                    "msr_input_path": "sheet.png",
                }), encoding="utf-8")
            plan = project / "plan.json"
            plan.write_text("[]", encoding="utf-8")
            state = Namespace(
                args=Namespace(
                    video_pipeline="minimax-h3-r2v",
                    reference_generation="sequence_sheet",
                    sequence_to_sheet_workflow="workflow.json",
                ),
                plan_for_next_step=plan,
                app_config_path=project / "app.json",
                reference_hero_workflow=project / "hero.json",
                reference_edit_workflow=project / "edit.json",
                context=Namespace(
                    project_config_path=config_path,
                    references_dir=references_dir,
                ),
            )

            _run_msr_references_stage(state)

        render_reference_bible.assert_not_called()

    def test_rebuilt_minimax_plan_preserves_existing_reference_paths(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            rebuilt = temp / "base.json"
            enriched = temp / "render_plan_refs.json"
            rebuilt.write_text(
                json.dumps([{
                    "scene": 3,
                    "references": {
                        "actor_ids": ["medieval_bard"],
                        "location_id": "tavern",
                        "reference_audio_paths": ["vocals.wav"],
                    },
                }]),
                encoding="utf-8",
            )
            enriched.write_text(
                json.dumps([{
                    "scene": 3,
                    "references": {
                        "actor_ids": ["medieval_bard"],
                        "actor_msr_paths": ["output/references/actors/medieval_bard/views/msr_sheet.png"],
                        "location_msr_path": "output/references/locations/tavern/views/hero.png",
                    },
                }]),
                encoding="utf-8",
            )

            _preserve_enriched_reference_paths(
                output_path=rebuilt,
                reference_plan_path=enriched,
            )

            result = json.loads(rebuilt.read_text(encoding="utf-8"))[0]

        self.assertEqual(
            ["output/references/actors/medieval_bard/views/msr_sheet.png"],
            result["references"]["actor_msr_paths"],
        )
        self.assertEqual(
            "output/references/locations/tavern/views/hero.png",
            result["references"]["location_msr_path"],
        )
        self.assertEqual(["vocals.wav"], result["references"]["reference_audio_paths"])

    def test_h3_prompts_is_an_atomic_cli_stage(self):
        args = run_pipeline.build_arg_parser().parse_args(["--stage", "h3_prompts"])

        self.assertEqual(["h3_prompts"], args.stages)

    def test_h3_input_reports_the_upstream_artifact_required(self):
        with TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "stage1_segments.json"

            with self.assertRaisesRegex(FileNotFoundError, "main_pipeline"):
                _read_h3_input(missing, "stage 1 segments")

    def test_run_pipeline_defaults_do_not_embed_windows_only_relative_prefixes(self):
        args = run_pipeline.build_arg_parser().parse_args([])

        self.assertEqual("app_config.json", args.app_config)
        self.assertNotIn(".\\", args.storyboard_workflow)
        self.assertNotIn(".\\", args.single_prompt_workflow)

    def test_runner_path_accepts_windows_relative_cli_paths(self):
        self.assertEqual(
            run_pipeline.runner_root() / "app_config.json",
            run_pipeline.resolve_runner_path(".\\app_config.json"),
        )

    def test_build_run_context_resolves_project_paths_like_test_ps1(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            input_dir = project_dir / "input"
            input_dir.mkdir(parents=True)
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "My Song: Final!",
                        "input_audio": "input/song demo.mp3",
                    },
                ),
                encoding="utf-8",
            )

            args = run_pipeline.build_arg_parser().parse_args([str(project_dir), "--smoke-only"])
            context = run_pipeline.build_run_context(args)

        self.assertEqual(config_path.resolve(), context.project_config_path)
        self.assertEqual((input_dir / "song demo.mp3").resolve(), context.input_audio)
        self.assertEqual("song demo", context.song_id)
        self.assertEqual(project_dir / "output" / "render" / "plans" / "base.json", context.render_plan)
        self.assertEqual(project_dir / "output" / "render" / "plans" / "compact.json", context.compact_plan)
        self.assertEqual(project_dir / "output" / "render" / "plans" / "anchored.json", context.anchored_plan)
        self.assertEqual(project_dir / "output" / "render" / "storyboard" / "index.html", context.storyboard_page)
        self.assertEqual(project_dir / "output" / "render" / "ltx_single_prompt_smoke", context.ltx_dir)
        self.assertEqual(project_dir / "output" / "render" / "final" / "video_only.mp4", context.final_concat_video)
        self.assertEqual(project_dir / "output" / "render" / "final" / "movie.mp4", context.final_concat)
        self.assertEqual(project_dir / "output" / "render" / "final" / "scene_audio_debug.mp4", context.final_concat_scene_audio_debug)


class RunPipelineOrchestrationTests(unittest.TestCase):
    def test_runner_reports_each_completed_stage_through_callback(self):
        state = Namespace(
            comfyui_client=None,
            plan_for_next_step=Path("plan.json"),
            final_video_path=None,
            video_only_path=None,
            openshot_project_path=None,
            timeline_project_path=None,
        )
        completed = []
        runner = Mock()
        with patch(
            "feverslop.composition.pipeline_runner.resolve_pipeline_stages",
            return_value=[PipelineStage.ANCHOR_FIX],
        ), patch(
            "feverslop.composition.pipeline_runner.build_run_state",
            return_value=state,
        ), patch.dict(
            "feverslop.composition.pipeline_runner.STAGE_RUNNERS",
            {PipelineStage.ANCHOR_FIX: runner},
        ):
            run_pipeline.run(Namespace(), on_stage_complete=completed.append)

        runner.assert_called_once_with(state)
        self.assertEqual(["anchor_fix"], completed)

    def test_selected_video_workflows_match_specialized_pipelines_and_render_modes(self):
        paths = {
            "msr_workflow": Path("msr.json"),
            "ingredients_workflow": Path("ingredients.json"),
            "relay_workflow": Path("relay.json"),
            "single_prompt_workflow": Path("single.json"),
        }
        cases = [
            ("ltx_msr", "auto", (paths["msr_workflow"],)),
            ("ltx_ingredients", "auto", (paths["ingredients_workflow"],)),
            ("ltx_i2v", "single_prompt", (paths["single_prompt_workflow"],)),
            ("ltx_i2v", "relay", (paths["relay_workflow"],)),
            (
                "ltx_i2v",
                "auto",
                (paths["relay_workflow"], paths["single_prompt_workflow"]),
            ),
        ]

        for video_pipeline, render_mode, expected in cases:
            with self.subTest(video_pipeline=video_pipeline, render_mode=render_mode):
                state = Namespace(
                    args=Namespace(video_pipeline=video_pipeline, render_mode=render_mode),
                    **paths,
                )
                self.assertEqual(expected, _selected_video_workflows(state))

    def test_selected_video_workflows_omits_empty_relay_path_in_auto_mode(self):
        state = Namespace(
            args=Namespace(video_pipeline="ltx_i2v", render_mode="auto"),
            msr_workflow=Path("msr.json"),
            ingredients_workflow=Path("ingredients.json"),
            relay_workflow=Path(),
            single_prompt_workflow=Path("single.json"),
        )

        self.assertEqual((Path("single.json"),), _selected_video_workflows(state))

    def test_main_pipeline_stage_forwards_selected_workflows_and_rolling_profile(self):
        state = Namespace(
            args=Namespace(
                video_pipeline="ltx_i2v",
                render_mode="auto",
                concept_batch_size=3,
                rolling_frame_profile="safe",
            ),
            context=Namespace(
                project_config_path=Path("project.json"),
                render_plan=Path("render-plan.json"),
            ),
            app_config_path=Path("app.json"),
            msr_workflow=Path("msr.json"),
            ingredients_workflow=Path("ingredients.json"),
            relay_workflow=Path("relay.json"),
            single_prompt_workflow=Path("single.json"),
        )

        with patch("feverslop.composition.stage_runners.execute_generate_render_plan") as execute:
            _run_main_pipeline_stage(state)

        request = execute.call_args.args[0]
        self.assertEqual((Path("relay.json"), Path("single.json")), request.video_workflow_paths)
        self.assertEqual("safe", request.rolling_frame_profile)

    def test_pipeline_stage_arg_executes_only_anchor_fix(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            prompts_dir = project_dir / "output" / "prompts"
            render_dir = project_dir / "output" / "render"
            prompts_dir.mkdir(parents=True)
            render_dir.mkdir(parents=True)
            (prompts_dir / "resolved_context_song.json").write_text(json.dumps({"subject": "Singer"}), encoding="utf-8")
            (render_dir / "render_plan_song.json").write_text(json.dumps([{"scene": 1, "prompt": "Singer sings"}]), encoding="utf-8")
            args = run_pipeline.build_arg_parser().parse_args([str(config_path), "--stage", "anchor_fix"])

            fixer = Mock()
            anchored_plan = render_dir / "plans" / "anchored.json"

            def fix_file(*, input_render_plan, output_render_plan):
                output_render_plan.write_text(Path(input_render_plan).read_text(encoding="utf-8"), encoding="utf-8")
                return output_render_plan

            fixer.fix_file.side_effect = fix_file
            with patch("feverslop.composition.stage_runners.run_unittest_suite") as tests, \
                patch("feverslop.composition.stage_runners.LTXPromptAnchorFixer", return_value=fixer) as fixer_class, \
                patch("feverslop.composition.stage_runners.build_generate_render_plan_use_case") as main_builder, \
                patch("feverslop.composition.stage_runners.build_render_storyboard_use_case") as storyboard_builder, \
                patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case") as video_builder, \
                patch("feverslop.composition.stage_runners.VideoPostProcessor") as postprocessor:
                result = run_pipeline.run(args)

        tests.assert_not_called()
        main_builder.assert_not_called()
        storyboard_builder.assert_not_called()
        video_builder.assert_not_called()
        postprocessor.assert_not_called()
        fixer_class.assert_called_once_with(subject_anchor="Singer")
        fixer.fix_file.assert_called_once()
        self.assertEqual(anchored_plan, result.render_plan_path)

    def test_pipeline_stage_error_names_failed_stage(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args([str(config_path), "--stage", "mux_original_audio"])

            with self.assertRaisesRegex(RuntimeError, "Mux original audio failed"):
                run_pipeline.run(args)

    def test_skip_flags_suppress_pipeline_steps(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-ltx",
                    "--skip-final-concat",
                ],
            )

            with patch("feverslop.composition.stage_runners.run_unittest_suite") as tests, \
                patch("feverslop.composition.stage_runners.build_generate_render_plan_use_case") as main_builder, \
                patch("feverslop.composition.stage_runners.OpenAICompatibleLLMClient") as llm, \
                patch("feverslop.composition.stage_runners.LTXPromptAnchorFixer") as fixer, \
                patch("feverslop.composition.stage_runners.build_render_storyboard_use_case") as storyboard_builder, \
                patch("feverslop.composition.stage_runners.generate_storyboard_page") as storyboard_page, \
                patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case") as video_builder, \
                patch("feverslop.composition.stage_runners.VideoPostProcessor") as postprocessor:
                result = run_pipeline.run(args)

        self.assertEqual(config_path.parent / "output" / "render" / "plans" / "base.json", result.render_plan_path)
        tests.assert_not_called()
        main_builder.assert_not_called()
        llm.assert_not_called()
        fixer.assert_not_called()
        storyboard_builder.assert_not_called()
        storyboard_page.assert_not_called()
        video_builder.assert_not_called()
        postprocessor.assert_not_called()

    def test_runner_render_progress_has_percent_elapsed_and_eta_columns(self):
        progress = run_pipeline.RenderProgressReporter("Rendering", total=3)

        column_types = [type(column) for column in progress.progress.columns]

        self.assertIn(TaskProgressColumn, column_types)
        self.assertIn(TimeElapsedColumn, column_types)
        self.assertIn(TimeRemainingColumn, column_types)

    def test_smoke_ltx_uses_selected_scene_and_forces_rerender(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 7}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-final-concat",
                    "--smoke-only",
                    "--smoke-scene",
                    "7",
                ],
            )

            use_case = Mock()
            use_case.execute.return_value = [project_dir / "output" / "render" / "ltx_single_prompt_smoke" / "final" / "scene_0007.mp4"]
            with patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case):
                run_pipeline.run(args)

        request = use_case.execute.call_args.args[0]
        self.assertEqual({7}, request.scene_numbers)
        self.assertFalse(request.skip_existing)

    def test_storyboard_runner_wires_progress_callback(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 1}, {"scene": 2}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard-page",
                    "--skip-ltx",
                    "--skip-final-concat",
                ],
            )

            use_case = Mock()
            use_case.execute.return_value = []
            with patch("feverslop.composition.stage_runners.build_render_storyboard_use_case", return_value=use_case):
                run_pipeline.run(args)

        request = use_case.execute.call_args.args[0]
        self.assertIsNotNone(request.on_frame_complete)
        self.assertEqual(2, request.on_frame_complete.__self__.total)

    def test_ltx_runner_wires_progress_callback(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([
                    {"scene": 1, "duration_seconds": 1.0},
                    {"scene": 2, "duration_seconds": 1.0},
                    {"scene": 3, "duration_seconds": 1.0},
                ]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-final-concat",
                ],
            )

            use_case = Mock()
            use_case.execute.return_value = []
            with patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case):
                run_pipeline.run(args)

        request = use_case.execute.call_args.args[0]
        self.assertIsNotNone(request.on_scene_complete)
        self.assertEqual(3, request.on_scene_complete.__self__.total)

    def test_ltx_msr_runner_does_not_require_relay_workflow_in_auto_mode(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 1}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--video-pipeline",
                    "ltx_msr",
                    "--stage",
                    "ltx_prepare_workflows",
                    "--render-mode",
                    "auto",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-msr-reference-render",
                    "--skip-final-concat",
                ],
            )

            use_case = Mock()
            use_case.execute.return_value = []
            with patch("feverslop.composition.stage_runners._missing_prepare_inputs", return_value=[]), \
                patch("feverslop.composition.stage_runners.WorkflowMaterializer") as materializer, \
                patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case) as builder:
                run_pipeline.run(args)

        options = builder.call_args.args[0]
        self.assertEqual("ltx_msr", options.video_pipeline)
        self.assertTrue(str(options.output_dir).endswith("ltx_msr"))
        materializer.return_value.prepare.assert_called_once()

    def test_ltx_msr_runner_builds_references_enriches_msr_prompts_and_skips_storyboard(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            base_plan = render_dir / "render_plan_song.json"
            base_plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--video-pipeline",
                    "ltx_msr",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-ltx",
                    "--skip-final-concat",
                ],
            )

            use_case = Mock()
            use_case.execute.return_value = []

            def enrich(
                input_plan,
                references_dir,
                output_plan,
                on_scene_complete=None,
                canonical_plan_path=None,
            ):
                self.assertIsNotNone(on_scene_complete)
                Path(output_plan).write_text(json.dumps([{"scene": 1, "references": {}}]), encoding="utf-8")
                on_scene_complete(1, 1, 1)
                return Path(output_plan)

            def enrich_msr(
                input_plan,
                output_plan,
                *,
                llm,
                canonical_plan_path=None,
                on_analysis_status=None,
                on_scene_complete=None,
            ):
                self.assertIsNotNone(llm)
                self.assertIsNotNone(on_analysis_status)
                self.assertIsNotNone(on_scene_complete)
                Path(output_plan).write_text(json.dumps([{"scene": 1, "ltx": {"msr_prompt_relay": []}}]), encoding="utf-8")
                on_scene_complete(1, 1, 1)
                return Path(output_plan)

            with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}), patch("feverslop.composition.stage_runners.build_render_storyboard_use_case") as storyboard_builder, \
                patch("feverslop.composition.stage_runners.generate_storyboard_page") as storyboard_page, \
                patch("feverslop.composition.stage_runners.render_reference_bible") as reference_bible, \
                patch("feverslop.composition.stage_runners.enrich_render_plan_with_reference_sheets", side_effect=enrich) as enrich_refs, \
                patch("feverslop.composition.stage_runners.enrich_render_plan_with_msr_prompts", side_effect=enrich_msr) as enrich_msr_prompts, \
                patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case) as video_builder:
                result = run_pipeline.run(args)

        storyboard_builder.assert_not_called()
        storyboard_page.assert_not_called()
        reference_bible.assert_called_once()
        self.assertEqual(base_plan, enrich_refs.call_args.args[0])
        self.assertEqual(project_dir / "output" / "references", enrich_refs.call_args.args[1])
        canonical_refs = project_dir / "output" / "render" / "plans" / "references.json"
        self.assertEqual(canonical_refs, enrich_refs.call_args.args[2])
        self.assertIn("on_scene_complete", enrich_refs.call_args.kwargs)
        enrich_msr_prompts.assert_called_once()
        self.assertEqual(canonical_refs, enrich_msr_prompts.call_args.args[0])
        self.assertEqual(canonical_refs, enrich_msr_prompts.call_args.args[1])
        self.assertIn("on_scene_complete", enrich_msr_prompts.call_args.kwargs)
        self.assertEqual(canonical_refs, result.render_plan_path)
        video_builder.assert_not_called()

    def test_ltx_msr_runner_can_resume_selected_scenes_without_prompt_enrichment(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            refs_plan = render_dir / "render_plan_song_refs.json"
            refs_plan.write_text(json.dumps([{"scene": 14}, {"scene": 15}, {"scene": 16}]), encoding="utf-8")
            (render_dir / "render_plan_song.json").write_text(
                json.dumps([{"scene": 14}, {"scene": 15}, {"scene": 16}]),
                encoding="utf-8",
            )
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--video-pipeline",
                    "ltx_msr",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-anchor-fix",
                    "--skip-msr-reference-render",
                    "--skip-msr-prompt-enrichment",
                    "--skip-ltx",
                    "--skip-final-concat",
                    "--scenes",
                    "15-16",
                ],
            )

            use_case = Mock()
            use_case.execute.return_value = []

            def enrich_refs(
                input_plan,
                references_dir,
                output_plan,
                on_scene_complete=None,
                canonical_plan_path=None,
            ):
                Path(output_plan).write_text(Path(input_plan).read_text(encoding="utf-8"), encoding="utf-8")
                return Path(output_plan)

            with patch("feverslop.composition.stage_runners.enrich_render_plan_with_reference_sheets", side_effect=enrich_refs), \
                patch("feverslop.composition.stage_runners.enrich_render_plan_with_msr_prompts") as enrich_msr_prompts, \
                patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case):
                run_pipeline.run(args)

            enrich_msr_prompts.assert_not_called()
            use_case.execute.assert_not_called()

    def test_ltx_resume_rewrites_concat_list_from_full_render_plan_before_final_concat(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            (project_dir / "song.mp3").write_bytes(b"audio")
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            plan_path = render_dir / "render_plan_song.json"
            plan_path.write_text(
                json.dumps([
                    {"scene": 1, "duration_seconds": 1.0},
                    {"scene": 2, "duration_seconds": 1.0},
                    {"scene": 3, "duration_seconds": 1.0},
                ]),
                encoding="utf-8",
            )
            clip_1 = render_dir / "ltx_single_prompt" / "final" / "scene_0001.mp4"
            clip_2 = render_dir / "ltx_single_prompt" / "final" / "scene_0002.mp4"
            clip_3 = render_dir / "ltx_single_prompt" / "final" / "scene_0003.mp4"
            clip_1.parent.mkdir(parents=True)
            clip_1.write_bytes(b"clip 1")
            clip_2.write_bytes(b"clip 2")
            clip_3.write_bytes(b"clip 3")
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                ],
            )

            use_case = Mock()
            use_case.execute.return_value = [clip_2]
            postprocessor = Mock()
            postprocessor.concat_clips.return_value = render_dir / "ltx_single_prompt" / "Song_video_only.mp4"
            postprocessor.mux_original_audio.return_value = render_dir / "ltx_single_prompt" / "Song.mp4"
            with patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case), \
                patch("feverslop.composition.stage_runners.VideoPostProcessor", return_value=postprocessor):
                run_pipeline.run(args)

            concat_list = render_dir / "final" / "concat_list.txt"
            self.assertEqual(
                [
                    f"file '{clip_1.resolve().as_posix()}'",
                    f"file '{clip_2.resolve().as_posix()}'",
                    f"file '{clip_3.resolve().as_posix()}'",
                ],
                concat_list.read_text(encoding="utf-8").splitlines(),
            )

    def test_ltx_msr_skip_ltx_rewrites_concat_list_from_existing_scene_clips(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            config_path = project_dir / "config.json"
            config_path.write_text(
                json.dumps({"project_name": "Song", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            (project_dir / "song.mp3").write_bytes(b"audio")
            render_dir = project_dir / "output" / "render"
            render_dir.mkdir(parents=True)
            plan_path = render_dir / "render_plan_song_refs.json"
            plan_path.write_text(
                json.dumps([
                    {"scene": 1, "duration_seconds": 1.0},
                    {"scene": 2, "duration_seconds": 1.0},
                    {"scene": 3, "duration_seconds": 1.0},
                ]),
                encoding="utf-8",
            )
            base_plan = render_dir / "render_plan_song.json"
            base_plan.write_text(
                json.dumps([
                    {"scene": 1, "duration_seconds": 1.0},
                    {"scene": 2, "duration_seconds": 1.0},
                    {"scene": 3, "duration_seconds": 1.0},
                ]),
                encoding="utf-8",
            )
            clip_1 = render_dir / "ltx_msr" / "scene_0001.mp4"
            clip_2 = render_dir / "ltx_msr" / "scene_0002.mp4"
            clip_3 = render_dir / "ltx_msr" / "scene_0003.mp4"
            clip_1.parent.mkdir(parents=True)
            clip_1.write_bytes(b"clip 1")
            clip_2.write_bytes(b"clip 2")
            clip_3.write_bytes(b"clip 3")
            args = run_pipeline.build_arg_parser().parse_args(
                [
                    str(config_path),
                    "--app-config",
                    "app_config.example.json",
                    "--video-pipeline",
                    "ltx_msr",
                    "--skip-tests",
                    "--skip-main-pipeline",
                    "--skip-relay-compact",
                    "--skip-anchor-fix",
                    "--skip-storyboard",
                    "--skip-storyboard-page",
                    "--skip-msr-reference-render",
                    "--skip-ltx",
                ],
            )

            def enrich(
                _input_plan,
                _references_dir,
                output_plan,
                on_scene_complete=None,
                canonical_plan_path=None,
            ):
                Path(output_plan).parent.mkdir(parents=True, exist_ok=True)
                Path(output_plan).write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
                if on_scene_complete is not None:
                    on_scene_complete(1, 1, 3)
                    on_scene_complete(2, 2, 3)
                    on_scene_complete(3, 3, 3)
                return Path(output_plan)

            postprocessor = Mock()
            postprocessor.concat_clips.return_value = render_dir / "ltx_msr" / "Song_video_only.mp4"
            postprocessor.mux_original_audio.return_value = render_dir / "ltx_msr" / "Song.mp4"
            with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}), patch("feverslop.composition.stage_runners.enrich_render_plan_with_reference_sheets", side_effect=enrich), \
                patch("feverslop.composition.stage_runners.VideoPostProcessor", return_value=postprocessor):
                run_pipeline.run(args)

            concat_list = render_dir / "final" / "concat_list.txt"
            self.assertEqual(
                [
                    f"file '{clip_1.resolve().as_posix()}'",
                    f"file '{clip_2.resolve().as_posix()}'",
                    f"file '{clip_3.resolve().as_posix()}'",
                ],
                concat_list.read_text(encoding="utf-8").splitlines(),
            )


def context_path(config_path: Path) -> Path:
    return config_path.parent / "output" / "render" / "render_plan_song.json"


if __name__ == "__main__":
    unittest.main()
