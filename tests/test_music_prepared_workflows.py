import argparse
import json
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from feverslop.composition.arg_parser import PipelineStage
from feverslop.composition.config_loader import PipelineRunState, build_run_context
from feverslop.composition.stage_runners import (
    STAGE_RUNNERS,
    _load_continuity_dirty,
    _merge_reference_paths_into_h3_segments,
    _run_ltx_prepare_workflows_stage,
    _run_ltx_render_scenes_stage,
    _run_visual_consistency_preflight,
    resolve_pipeline_stages,
)
from feverslop.domain.visual_consistency import (
    PreflightMode,
    ReferenceAnchor,
    SceneConsistencyContract,
)
from feverslop.ports.visual_consistency import ReferenceManifestSnapshot
from feverslop.application.visual_consistency_preflight import (
    VisualConsistencyPreflightResult,
)


class MusicPreparedWorkflowStageTests(unittest.TestCase):
    def test_minimax_uses_canonical_scene_artifacts_without_debug_directory(self):
        with TemporaryDirectory() as tmp:
            for pipeline in ("minimax-h3-r2v", "minimax-h3-t2v"):
                project = Path(tmp) / pipeline
                project.mkdir()
                state = self._state(project, pipeline=pipeline)

                self.assertEqual(state.context.artifact_layout.scenes_dir, state.context.ltx_dir)
                self.assertIsNone(state.context.ltx_debug_dir)

    def test_h3_segments_receive_reference_paths_from_enriched_plan(self):
        with TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "reference_plan.json"
            plan_path.write_text(json.dumps([{
                "scene": 1,
                "references": {
                    "actor_msr_paths": ["output/references/actors/bard.png"],
                    "location_msr_path": "output/references/locations/tavern.png",
                },
            }]), encoding="utf-8")

            enriched = _merge_reference_paths_into_h3_segments(
                [{"segment_id": "segment_001", "scene": 1, "references": {"actor_ids": ["bard"]}}],
                plan_path,
            )

        self.assertEqual(
            "output/references/actors/bard.png",
            enriched[0]["references"]["actor_msr_paths"][0],
        )
        self.assertEqual(
            "output/references/locations/tavern.png",
            enriched[0]["references"]["location_msr_path"],
        )
    def test_stage_resolves_selected_profile_continuous_capability(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr")
            state.args.video_workflow_profile = "custom-msr"
            state.args.visual_consistency_preflight = PreflightMode.STRICT
            scene = Mock()
            scene.to_dict.return_value = {
                "scene": 1,
                "visual_consistency": {"workflow_profile": "custom-msr"},
            }
            with patch(
                "feverslop.composition.stage_runners.ProjectReferenceManifestAdapter.load",
                return_value=Mock(spec=ReferenceManifestSnapshot),
            ), patch(
                "feverslop.composition.stage_runners._resolved_startframe_profile",
                return_value=SimpleNamespace(
                    name="custom-msr",
                    supports_start_frame=True,
                ),
            ), patch(
                "feverslop.composition.stage_runners.preflight_visual_consistency",
                return_value=VisualConsistencyPreflightResult((), ()),
            ) as preflight, patch(
                "feverslop.composition.stage_runners.validate_project_scene_artifacts",
                return_value=(),
            ):
                _run_visual_consistency_preflight(state, [scene])

        self.assertEqual("custom-msr", preflight.call_args.kwargs["workflow_profile"])
        self.assertTrue(
            preflight.call_args.kwargs["supports_continuous_transitions"]
        )

    def test_malformed_continuity_marker_fails_closed(self):
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "continuity_dirty.json"
            marker.write_text("{", encoding="utf-8")

            self.assertEqual(
                {1, 2, 3},
                _load_continuity_dirty(marker, {1, 2, 3}),
            )

    def _state(self, project: Path, *, pipeline: str, scenes: str = "") -> PipelineRunState:
        config = project / "config.json"
        config.write_text(json.dumps({"project_name": "Song", "input_audio": "song.mp3"}), encoding="utf-8")
        args = argparse.Namespace(
            project_config=str(config), project_root=None, video_pipeline=pipeline,
            render_mode="single_prompt", smoke_only=False, scenes=scenes, smoke_scene=1,
            no_skip_existing=False, randomize_seed=False, rolling_frame_profile="off",
            video_character_lora_strength=None, video_lora_1_strength_model=None,
            video_lora_1_strength_clip=None, lora_split_enabled=False,
            single_prompt_title="#PROMPT", single_prompt_input="text", relay_workflow="",
            skip_facefix=False,
        )
        context = build_run_context(args)
        return PipelineRunState(
            args=args, context=context, app_config_path=project / "app.json",
            storyboard_workflow=project / "storyboard.json",
            reference_hero_workflow=project / "hero.json", reference_edit_workflow=project / "edit.json",
            msr_workflow=project / "msr.json", ingredients_workflow=project / "ingredients.json",
            relay_workflow=Path(""), single_prompt_workflow=project / "i2v.json",
            facefix_workflow=project / "facefix.json",
            plan_for_next_step=context.ingredients_plan if pipeline == "ltx_ingredients" else context.reference_plan,
        )

    def test_default_specialized_pipeline_prepares_before_render(self):
        args = argparse.Namespace(
            stages=None, skip_tests=True, skip_main_pipeline=True, skip_relay_compact=True,
            skip_anchor_fix=True, video_pipeline="ltx_msr", skip_msr_reference_render=True,
            skip_msr_prompt_enrichment=True, skip_ingredients_sheets=True, skip_ltx=False,
            skip_final_concat=True, render_mode="single_prompt", skip_storyboard=True,
            skip_storyboard_page=True, diagnostic_original_audio_mux=False,
            no_original_audio_mux=False, skip_facefix=True,
        )

        stages = resolve_pipeline_stages(args)

        self.assertLess(stages.index(PipelineStage.LTX_PREPARE_WORKFLOWS), stages.index(PipelineStage.LTX_RENDER_SCENES))
        self.assertIn(PipelineStage.LTX_PREPARE_WORKFLOWS, STAGE_RUNNERS)

    def test_minimax_r2v_uses_msr_reference_stages_before_render(self):
        args = argparse.Namespace(
            stages=None, skip_tests=True, skip_main_pipeline=False, skip_relay_compact=True,
            skip_anchor_fix=True, video_pipeline="minimax-h3-r2v", skip_msr_reference_render=True,
            skip_msr_prompt_enrichment=True, skip_ingredients_sheets=True, skip_ltx=False,
            skip_final_concat=True, render_mode="single_prompt", skip_storyboard=True,
            skip_storyboard_page=True, diagnostic_original_audio_mux=False,
            no_original_audio_mux=False, skip_facefix=True,
        )

        stages = resolve_pipeline_stages(args)

        self.assertNotIn(PipelineStage.STORYBOARD_FRAMES, stages)
        self.assertNotIn(PipelineStage.STORYBOARD_PAGE, stages)
        self.assertLess(
            stages.index(PipelineStage.MSR_REFERENCE_SHEETS),
            stages.index(PipelineStage.LTX_RENDER_SCENES),
        )
        self.assertLess(
            stages.index(PipelineStage.MSR_REFERENCE_SHEETS),
            stages.index(PipelineStage.H3_PROMPTS),
        )
        self.assertLess(
            stages.index(PipelineStage.H3_PROMPTS),
            stages.index(PipelineStage.RENDER_PLAN),
        )

    def test_prepare_uses_same_scene_selection_and_never_queues(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1,3")
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([
                {"scene": number, "ingredients": {
                    "sheet_path": f"sheet{number}.png", "anchors": [], "global_prompt": f"scene {number}",
                }, "ltx": {"base_prompt": f"scene {number}", "static_prompt": f"scene {number}", "prompt_relay": [
                    {"frame_start": 0, "frame_end": 48, "state": "instrumental", "prompt": "mouth closed"},
                ]}}
                for number in (1, 2, 3)
            ]), encoding="utf-8")
            for number in (1, 2, 3):
                (project / f"sheet{number}.png").write_bytes(b"sheet")
            use_case = Mock()
            backend = use_case.backend
            materializer = Mock()
            with patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case), \
                 patch("feverslop.composition.stage_runners.WorkflowMaterializer", return_value=materializer), \
                 patch("feverslop.composition.stage_runners._run_visual_consistency_preflight") as preflight:
                _run_ltx_prepare_workflows_stage(state)

        self.assertEqual([1, 3], [call.args[0].scene["scene"] for call in materializer.prepare.call_args_list])
        self.assertEqual(
            [1, 2, 3],
            [scene.scene_number for scene in preflight.call_args.args[1]],
        )
        backend.render_queue.queue_workflow_and_download_first_video.assert_not_called()

    def test_selected_continuous_scene_preflights_its_predecessor(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="2")
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([
                {
                    "scene": 1,
                    "ingredients": {"sheet_path": "sheet1.png", "anchors": []},
                    "ltx": {"base_prompt": "one", "static_prompt": "one"},
                },
                {
                    "scene": 2,
                    "transition_from_previous": "continuous",
                    "ingredients": {"sheet_path": "sheet2.png", "anchors": []},
                    "ltx": {"base_prompt": "two", "static_prompt": "two"},
                },
            ]), encoding="utf-8")
            for number in (1, 2):
                (project / f"sheet{number}.png").write_bytes(b"sheet")
            materializer = Mock()

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(),
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ), patch(
                "feverslop.composition.stage_runners._run_visual_consistency_preflight"
            ) as preflight:
                _run_ltx_prepare_workflows_stage(state)

        self.assertEqual(
            [1, 2],
            [scene.scene_number for scene in preflight.call_args.args[1]],
        )
        self.assertEqual(
            [2],
            [
                call.args[0].scene["scene"]
                for call in materializer.prepare.call_args_list
            ],
        )

    def test_prepare_defers_startframe_handoff_until_sequential_render(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr", scenes="2")
            state.args.video_workflow_profile = "msr-startframe"
            state.context.input_audio.write_bytes(b"audio")
            state.msr_workflow = Path(
                "workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json"
            ).resolve()
            state.app_config_path.write_text(
                json.dumps(
                    {
                        "video_workflow_profiles": [
                            {
                                "name": "msr-startframe",
                                "pipeline": "ltx_msr",
                                "workflow": "workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json",
                                "purpose": "final",
                                "stages": 1,
                                "output_scale": 1,
                                "supports_per_pass_loras": False,
                                "supports_start_frame": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            actor = project / "actor.png"
            location = project / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            scenes = [
                _consistent_music_scene(1, actor, location),
                _consistent_music_scene(
                    2,
                    actor,
                    location,
                    transition="continuous",
                ),
            ]
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps(scenes), encoding="utf-8")
            previous_clip = state.context.artifact_layout.scene_final_video(1)
            previous_clip.parent.mkdir(parents=True)
            previous_clip.write_bytes(b"clip")
            extractor = Mock()
            materializer = Mock()

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(),
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ), patch(
                "feverslop.composition.stage_runners.PostprocessorFrameExtractor",
                return_value=extractor,
            ), patch(
                "feverslop.composition.stage_runners._run_visual_consistency_preflight"
            ):
                _run_ltx_prepare_workflows_stage(state)

        materializer.prepare.assert_not_called()
        self.assertFalse(
            state.context.artifact_layout.scene_manifest(2).exists()
        )
        extractor.extract_last_frame.assert_not_called()

    def test_prepare_projects_preflight_contracts_without_mutating_source_plan(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr")
            _configure_startframe_music_state(state, project)
            source = json.loads(
                state.plan_for_next_step.read_text(encoding="utf-8")
            )
            contracts = tuple(
                SceneConsistencyContract.from_dict(
                    scene.pop("visual_consistency")
                )
                for scene in source
            )
            state.plan_for_next_step.write_text(
                json.dumps(source),
                encoding="utf-8",
            )
            materializer = Mock()

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(),
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ), patch(
                "feverslop.composition.stage_runners._run_visual_consistency_preflight",
                return_value=VisualConsistencyPreflightResult(contracts, ()),
            ):
                _run_ltx_prepare_workflows_stage(state)

            [request] = [
                call.args[0]
                for call in materializer.prepare.call_args_list
            ]
            self.assertEqual(
                contracts[0].to_dict(),
                request.scene["visual_consistency"],
            )
            self.assertEqual(
                source,
                json.loads(
                    state.plan_for_next_step.read_text(encoding="utf-8")
                ),
            )
            self.assertFalse(
                state.context.artifact_layout.scene_manifest(2).exists()
            )

    def test_full_prepare_defers_continuous_manifest_until_sequential_render(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr")
            _configure_startframe_music_state(state, project)
            contracts = tuple(
                SceneConsistencyContract.from_dict(
                    scene["visual_consistency"]
                )
                for scene in json.loads(
                    state.plan_for_next_step.read_text(encoding="utf-8")
                )
            )
            backend = _prepared_backend(_RecordingFramePostprocessor())
            materializer = Mock()

            def persist_prepared(request):
                number = request.scene["scene"]
                workflow = (
                    state.context.artifact_layout.scene_workflow(number)
                )
                workflow.parent.mkdir(parents=True, exist_ok=True)
                workflow.write_text("{}", encoding="utf-8")
                workflow.with_name("manifest.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

            materializer.prepare.side_effect = persist_prepared
            preflight = VisualConsistencyPreflightResult(contracts, ())
            common_patches = (
                patch(
                    "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                    return_value=Mock(backend=backend),
                ),
                patch(
                    "feverslop.composition.stage_runners.WorkflowMaterializer",
                    return_value=materializer,
                ),
                patch(
                    "feverslop.composition.stage_runners._run_visual_consistency_preflight",
                    return_value=preflight,
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2]:
                _run_ltx_prepare_workflows_stage(state)

            self.assertEqual(
                [1],
                [
                    call.args[0].scene["scene"]
                    for call in materializer.prepare.call_args_list
                ],
            )
            self.assertFalse(
                state.context.artifact_layout.scene_manifest(2).exists()
            )
            materializer.reset_mock()
            renderer = _SequentialPreparedRenderer(
                state.context.artifact_layout
            )

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(backend=backend),
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ), patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=renderer,
            ), patch(
                "feverslop.composition.stage_runners._run_visual_consistency_preflight",
                return_value=preflight,
            ):
                _run_ltx_render_scenes_stage(state)

            self.assertEqual([1, 2], renderer.rendered)
            self.assertEqual(
                [1, 2],
                [
                    call.args[0].scene["scene"]
                    for call in materializer.prepare.call_args_list
                ],
            )
            self.assertTrue(
                state.context.artifact_layout.scene_manifest(2).exists()
            )

    def test_render_handoff_uses_fresh_or_required_immediate_predecessor(self):
        cases = (
            ("", None, b"new-1"),
            ("1", None, b"new-1"),
            ("1,2", b"stale-1", b"new-1"),
            ("2", b"existing-1", b"existing-1"),
        )
        for selection, predecessor_content, expected_source in cases:
            with self.subTest(selection=selection), TemporaryDirectory() as tmp:
                project = Path(tmp)
                state = self._state(
                    project,
                    pipeline="ltx_msr",
                    scenes=selection,
                )
                _configure_startframe_music_state(state, project)
                if predecessor_content is not None:
                    predecessor = (
                        state.context.artifact_layout.scene_final_video(1)
                    )
                    predecessor.parent.mkdir(parents=True, exist_ok=True)
                    predecessor.write_bytes(predecessor_content)
                if selection in {"", "1"}:
                    stale_downstream = (
                        state.context.artifact_layout.scene_final_video(2)
                    )
                    stale_downstream.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    stale_downstream.write_bytes(b"stale-2")
                for number in ((1, 2) if selection != "2" else (2,)):
                    workflow = state.context.artifact_layout.scene_workflow(number)
                    workflow.parent.mkdir(parents=True, exist_ok=True)
                    workflow.write_text("{}", encoding="utf-8")
                    workflow.with_name("manifest.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )
                postprocessor = _RecordingFramePostprocessor()
                backend = _prepared_backend(postprocessor)
                renderer = _SequentialPreparedRenderer(
                    state.context.artifact_layout
                )
                materializer = Mock()

                with patch(
                    "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                    return_value=Mock(backend=backend),
                ), patch(
                    "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                    return_value=renderer,
                ), patch(
                    "feverslop.composition.stage_runners.WorkflowMaterializer",
                    return_value=materializer,
                ):
                    _run_ltx_render_scenes_stage(state)

                self.assertEqual([expected_source], postprocessor.sources)
                self.assertEqual(
                    [1, 2] if selection != "2" else [2],
                    renderer.rendered,
                )
                handoff_scene = materializer.prepare.call_args.args[0].scene
                self.assertEqual(2, handoff_scene["scene"])
                self.assertEqual(
                    "last_frame_from_previous",
                    handoff_scene["keyframes"]["startframe_mode"],
                )

    def test_selected_handoff_requires_existing_unselected_predecessor(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr", scenes="2")
            _configure_startframe_music_state(state, project)
            workflow = state.context.artifact_layout.scene_workflow(2)
            workflow.parent.mkdir(parents=True, exist_ok=True)
            workflow.write_text("{}", encoding="utf-8")
            workflow.with_name("manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
            backend = _prepared_backend(_RecordingFramePostprocessor())

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(backend=backend),
            ), patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=_SequentialPreparedRenderer(
                    state.context.artifact_layout
                ),
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
            ), self.assertRaisesRegex(
                ValueError,
                "missing previous movie scene clip",
            ):
                _run_ltx_render_scenes_stage(state)

    def test_rendered_predecessor_dirties_entire_continuous_chain(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr")
            _configure_startframe_music_state(state, project)
            plan = json.loads(
                state.plan_for_next_step.read_text(encoding="utf-8")
            )
            plan.append(
                _consistent_music_scene(
                    3,
                    project / "actor.png",
                    project / "location.png",
                    transition="continuous",
                )
            )
            state.plan_for_next_step.write_text(
                json.dumps(plan),
                encoding="utf-8",
            )
            for number in (1, 2, 3):
                workflow = state.context.artifact_layout.scene_workflow(number)
                workflow.parent.mkdir(parents=True, exist_ok=True)
                workflow.write_text("{}", encoding="utf-8")
                workflow.with_name("manifest.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
            for number in (2, 3):
                stale = state.context.artifact_layout.scene_final_video(number)
                stale.write_bytes(f"stale-{number}".encode())
            postprocessor = _RecordingFramePostprocessor()
            backend = _prepared_backend(postprocessor)
            renderer = _SequentialPreparedRenderer(
                state.context.artifact_layout
            )
            materializer = Mock()

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(backend=backend),
            ), patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=renderer,
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ):
                _run_ltx_render_scenes_stage(state)

            self.assertEqual([1, 2, 3], renderer.rendered)
            self.assertEqual([b"new-1", b"new-2"], postprocessor.sources)
            self.assertEqual(
                [1, 2, 3],
                [
                    call.args[0].scene["scene"]
                    for call in materializer.prepare.call_args_list
                ],
            )

    def test_failed_dependent_persists_dirty_state_for_next_invocation(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr")
            _configure_startframe_music_state(state, project)
            _write_prepared_placeholders(state, (1, 2))
            stale = state.context.artifact_layout.scene_final_video(2)
            stale.write_bytes(b"stale-2")
            backend = _prepared_backend(_RecordingFramePostprocessor())
            failing = _SequentialPreparedRenderer(
                state.context.artifact_layout,
                fail_on=2,
            )

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(backend=backend),
            ), patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=failing,
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=Mock(),
            ), self.assertRaisesRegex(RuntimeError, "scene 2 failed"):
                _run_ltx_render_scenes_stage(state)

            marker = state.context.render_dir / "continuity_dirty.json"
            self.assertEqual(
                [2],
                json.loads(marker.read_text(encoding="utf-8"))[
                    "dirty_scenes"
                ],
            )
            resumed = _SequentialPreparedRenderer(
                state.context.artifact_layout
            )
            manifest = Mock(pipeline="ltx_msr")
            manifest.verify.return_value = []
            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(backend=backend),
            ), patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=resumed,
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=Mock(),
            ), patch(
                "feverslop.composition.stage_runners.SceneWorkflowManifest.read",
                return_value=manifest,
            ):
                _run_ltx_render_scenes_stage(state)

            self.assertEqual([2], resumed.rendered)
            self.assertFalse(marker.exists())

    def test_chain_failure_resume_clears_marker_after_last_dependent(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr")
            _configure_startframe_music_state(state, project)
            plan = json.loads(
                state.plan_for_next_step.read_text(encoding="utf-8")
            )
            plan.append(
                _consistent_music_scene(
                    3,
                    project / "actor.png",
                    project / "location.png",
                    transition="continuous",
                )
            )
            state.plan_for_next_step.write_text(
                json.dumps(plan),
                encoding="utf-8",
            )
            _write_prepared_placeholders(state, (1, 2, 3))
            for number in (2, 3):
                state.context.artifact_layout.scene_final_video(
                    number
                ).write_bytes(b"stale")
            backend = _prepared_backend(_RecordingFramePostprocessor())
            failing = _SequentialPreparedRenderer(
                state.context.artifact_layout,
                fail_on=3,
            )
            common = (
                patch(
                    "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                    return_value=Mock(backend=backend),
                ),
                patch(
                    "feverslop.composition.stage_runners.WorkflowMaterializer",
                    return_value=Mock(),
                ),
            )
            with common[0], common[1], patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=failing,
            ), self.assertRaisesRegex(RuntimeError, "scene 3 failed"):
                _run_ltx_render_scenes_stage(state)

            marker = state.context.render_dir / "continuity_dirty.json"
            self.assertEqual(
                [3],
                json.loads(marker.read_text(encoding="utf-8"))[
                    "dirty_scenes"
                ],
            )
            resumed = _SequentialPreparedRenderer(
                state.context.artifact_layout
            )
            manifest = Mock(pipeline="ltx_msr")
            manifest.verify.return_value = []
            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(backend=backend),
            ), patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=resumed,
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=Mock(),
            ), patch(
                "feverslop.composition.stage_runners.SceneWorkflowManifest.read",
                return_value=manifest,
            ):
                _run_ltx_render_scenes_stage(state)

            self.assertEqual([3], resumed.rendered)
            self.assertFalse(marker.exists())

    def test_startframe_profile_rejects_mismatched_materialized_workflow(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr", scenes="1")
            _configure_startframe_music_state(state, project)
            mismatched = project / "other-msr.json"
            mismatched.write_text("{}", encoding="utf-8")
            state.msr_workflow = mismatched

            with patch(
                "feverslop.composition.stage_runners._run_visual_consistency_preflight"
            ), patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case"
            ) as builder, patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer"
            ) as materializer, self.assertRaisesRegex(
                ValueError,
                "start-frame profile workflow",
            ):
                _run_ltx_prepare_workflows_stage(state)

            builder.assert_not_called()
            materializer.assert_not_called()

    def test_strict_visual_consistency_preflight_blocks_before_materialization(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1")
            (project / "song.mp3").write_bytes(b"audio")
            state.args.visual_consistency_preflight = "strict"
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([{
                "scene": 1,
                "references": {"actor_ids": ["missing"]},
                "ingredients": {
                    "sheet_path": "sheet.png",
                    "anchors": [{"id": "missing"}],
                    "global_prompt": "Reference `missing`",
                },
                "ltx": {"base_prompt": "scene", "static_prompt": "scene"},
            }]), encoding="utf-8")
            (project / "sheet.png").write_bytes(b"sheet")

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case"
            ) as use_case, patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer"
            ) as materializer, self.assertRaisesRegex(
                ValueError, "missing_actor_reference"
            ):
                _run_ltx_prepare_workflows_stage(state)

            use_case.assert_not_called()
            materializer.assert_not_called()

    def test_warn_preflight_reports_missing_actor_and_still_prepares(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1")
            state.args.visual_consistency_preflight = PreflightMode.WARN
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([{
                "scene": 1,
                "references": {"actor_ids": ["missing"]},
                "ingredients": {
                    "sheet_path": "sheet.png",
                    "anchors": [{"id": "missing"}],
                    "global_prompt": "Reference `missing`",
                },
                "ltx": {"base_prompt": "scene", "static_prompt": "scene"},
            }]), encoding="utf-8")
            (project / "sheet.png").write_bytes(b"sheet")
            use_case = Mock()
            materializer = Mock()

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=use_case,
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ), patch(
                "feverslop.composition.stage_runners.console.print"
            ) as output:
                _run_ltx_prepare_workflows_stage(state)

            materializer.prepare.assert_called_once()
            self.assertTrue(
                any(
                    "WARNING" in str(call.args[0])
                    and "missing_actor_reference" in str(call.args[0])
                    for call in output.call_args_list
                )
            )

    def test_off_preflight_bypasses_manifest_config_and_contract_loading(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1")
            state.args.visual_consistency_preflight = PreflightMode.OFF
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([{
                "scene": 1,
                "ingredients": {"sheet_path": "sheet.png", "anchors": []},
                "ltx": {"base_prompt": "scene", "static_prompt": "scene"},
            }]), encoding="utf-8")
            (project / "sheet.png").write_bytes(b"sheet")
            materializer = Mock()

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(),
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ), patch(
                "feverslop.composition.stage_runners.ProjectConfig.load"
            ) as config_load, patch(
                "feverslop.composition.stage_runners.ProjectReferenceManifestAdapter"
            ) as manifest_adapter, patch(
                "feverslop.composition.stage_runners.preflight_visual_consistency"
            ) as contract_preflight:
                _run_ltx_prepare_workflows_stage(state)

            materializer.prepare.assert_called_once()
            config_load.assert_not_called()
            manifest_adapter.assert_not_called()
            contract_preflight.assert_not_called()

    def test_strict_preflight_rejects_external_ingredients_artifact(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside_tmp:
            project = Path(tmp)
            outside = Path(outside_tmp) / "outside.png"
            outside.write_bytes(b"outside")
            state = self._state(project, pipeline="ltx_ingredients", scenes="1")
            state.args.visual_consistency_preflight = PreflightMode.STRICT
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            reference_dir = (
                project / "output" / "references" / "actors" / "hero"
            )
            reference_dir.mkdir(parents=True)
            reference_asset = reference_dir / "sheet.png"
            reference_asset.write_bytes(b"hero")
            (reference_dir / "manifest.json").write_text(
                json.dumps({
                    "id": "hero",
                    "sheet_path": reference_asset.relative_to(project).as_posix(),
                    "visual_description": "hero reference",
                }),
                encoding="utf-8",
            )
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([{
                "scene": 1,
                "references": {"actor_ids": ["hero"]},
                "ingredients": {
                    "sheet_path": str(outside),
                    "anchors": [{"id": "hero"}],
                    "global_prompt": "Reference `hero`",
                },
                "ltx": {"base_prompt": "scene", "static_prompt": "scene"},
            }]), encoding="utf-8")

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case"
            ) as use_case, patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer"
            ) as materializer, self.assertRaisesRegex(
                ValueError, "invalid_ingredients_sheet_path"
            ):
                _run_ltx_prepare_workflows_stage(state)

            use_case.assert_not_called()
            materializer.assert_not_called()

    def test_default_preflight_warns_for_legacy_plan_and_loads_manifest_once(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1,2")
            self.assertFalse(hasattr(state.args, "visual_consistency_preflight"))
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([
                {
                    "scene": number,
                    "ingredients": {
                        "sheet_path": f"sheet{number}.png",
                        "anchors": [],
                    },
                    "ltx": {"base_prompt": f"scene {number}", "static_prompt": f"scene {number}"},
                }
                for number in (1, 2)
            ]), encoding="utf-8")
            for number in (1, 2):
                (project / f"sheet{number}.png").write_bytes(b"sheet")
            adapter = Mock()
            adapter.load.return_value = ReferenceManifestSnapshot(
                actors={},
                locations={},
                revision="legacy",
            )
            materializer = Mock()

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=Mock(),
            ), patch(
                "feverslop.composition.stage_runners.WorkflowMaterializer",
                return_value=materializer,
            ), patch(
                "feverslop.composition.stage_runners.ProjectReferenceManifestAdapter",
                return_value=adapter,
            ), patch(
                "feverslop.composition.stage_runners.console.print"
            ) as output:
                _run_ltx_prepare_workflows_stage(state)

            self.assertEqual(2, materializer.prepare.call_count)
            adapter.load.assert_called_once()
            warnings = [
                str(call.args[0])
                for call in output.call_args_list
                if "legacy_contract_unknown" in str(call.args[0])
            ]
            self.assertEqual(2, len(warnings))

    def test_render_requires_prepared_scene_and_names_prepare_stage(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_msr", scenes="5")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([{"scene": 5}]), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "--stage ltx_prepare_workflows"):
                _run_ltx_render_scenes_stage(state)

    def test_render_wires_current_server_adapters_into_prepared_renderer(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            layout = state.context.artifact_layout
            layout.scene_workflow(1).parent.mkdir(parents=True)
            layout.scene_workflow(1).write_text("{}", encoding="utf-8")
            layout.scene_manifest(1).write_text("{}", encoding="utf-8")
            (project / "song.mp3").write_bytes(b"audio")
            Path(state.ingredients_workflow).write_text("{}", encoding="utf-8")
            backend = Mock()
            backend.max_render_frames = None
            backend.max_render_duration_seconds = None
            backend.render_budget_workflow_path = state.ingredients_workflow
            backend.round_render_frames_to_8n1 = False
            backend.workflow_label = state.ingredients_workflow
            backend.asset_uploader.names = {}
            backend.asset_uploader.resolve_audio_name.return_value = "song.mp3"
            backend._rolling_spec.return_value = {}
            backend._seed_for_scene.return_value = 1
            backend.build_workflow.return_value = {}
            backend.model_resolver.resolve_workflow_models.return_value = {}
            renderer = Mock()
            renderer.render.return_value = layout.scene_final_video(1)
            use_case = Mock(backend=backend)

            with patch(
                "feverslop.composition.stage_runners.build_render_video_scenes_use_case",
                return_value=use_case,
            ), patch(
                "feverslop.composition.stage_runners.PreparedWorkflowRenderer",
                return_value=renderer,
            ) as renderer_class:
                _run_ltx_render_scenes_stage(state)

            renderer_class.assert_called_once_with(
                project_dir=state.context.project_config_dir,
                render_queue=backend.render_queue,
                postprocessor=backend.postprocessor,
                expected_pipeline="ltx_ingredients",
                expected_workflow_profile=state.ingredients_workflow.stem,
                max_render_frames=None,
                max_render_duration_seconds=None,
                render_budget_workflow_path=state.ingredients_workflow,
                round_render_frames_to_8n1=False,
                asset_uploader=backend.asset_uploader,
                model_resolver=backend.model_resolver,
                model_workflow_path=backend.workflow_label,
            )

    def test_prepare_aggregates_missing_plan_audio_and_template(self):
        with TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), pipeline="ltx_ingredients")

            with self.assertRaises(FileNotFoundError) as raised:
                _run_ltx_prepare_workflows_stage(state)

            message = str(raised.exception)
            self.assertIn("render plan", message)
            self.assertIn("audio", message)
            self.assertIn("workflow template", message)

    def test_prepare_failure_rolls_back_manifests_created_in_same_invocation(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = self._state(project, pipeline="ltx_ingredients", scenes="1,2")
            state.context.input_audio.write_bytes(b"audio")
            state.ingredients_workflow.write_text("{}", encoding="utf-8")
            state.plan_for_next_step.parent.mkdir(parents=True)
            state.plan_for_next_step.write_text(json.dumps([
                {"scene": 1, "ingredients_scene_sheet": "sheet1.png"},
                {"scene": 2, "ingredients_scene_sheet": "sheet2.png"},
            ]), encoding="utf-8")
            for number in (1, 2):
                (project / f"sheet{number}.png").write_bytes(b"sheet")
            use_case = Mock()
            materializer = Mock()

            def prepare(request):
                layout = state.context.artifact_layout
                layout.scene_workflow(request.scene["scene"]).parent.mkdir(parents=True, exist_ok=True)
                layout.scene_workflow(request.scene["scene"]).write_text("{}")
                layout.scene_manifest(request.scene["scene"]).write_text("{}")
                if request.scene["scene"] == 2:
                    raise RuntimeError("failed")

            materializer.prepare.side_effect = prepare
            with patch("feverslop.composition.stage_runners.build_render_video_scenes_use_case", return_value=use_case), \
                 patch("feverslop.composition.stage_runners.WorkflowMaterializer", return_value=materializer), \
                 self.assertRaisesRegex(RuntimeError, "failed"):
                _run_ltx_prepare_workflows_stage(state)

            self.assertFalse(state.context.artifact_layout.scene_manifest(1).exists())
            self.assertFalse(state.context.artifact_layout.scene_manifest(2).exists())


def _consistent_music_scene(
    number: int,
    actor_path: Path,
    location_path: Path,
    *,
    transition: str = "cut",
) -> dict:
    actor = ReferenceAnchor(
        id="hero",
        kind="actor",
        look_id="default",
        asset_role="identity-reference",
        asset_sha256="a" * 64,
        prompt_anchor="hero",
    )
    location = ReferenceAnchor(
        id="archive",
        kind="location",
        look_id="default",
        asset_role="environment-reference",
        asset_sha256="b" * 64,
        prompt_anchor="archive",
    )
    contract = SceneConsistencyContract.create(
        scene=number,
        mode="msr",
        workflow_profile="msr-startframe",
        actors=(actor,),
        location=location,
        transition_from_previous=transition,
    )
    return {
        "scene": number,
        "transition_from_previous": transition,
        "references": {
            "actor_ids": ["hero"],
            "location_id": "archive",
            "actor_msr_paths": [actor_path.as_posix()],
            "location_msr_path": location_path.as_posix(),
        },
        "visual_consistency": contract.to_dict(),
        "ltx": {"base_prompt": f"scene {number}"},
    }


def _configure_startframe_music_state(
    state: PipelineRunState,
    project: Path,
) -> None:
    state.args.video_workflow_profile = "msr-startframe"
    state.context.input_audio.write_bytes(b"audio")
    state.msr_workflow = Path(
        "workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json"
    ).resolve()
    state.app_config_path.write_text(
        json.dumps(
            {
                "video_workflow_profiles": [
                    {
                        "name": "msr-startframe",
                        "pipeline": "ltx_msr",
                        "workflow": "workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json",
                        "purpose": "final",
                        "stages": 1,
                        "output_scale": 1,
                        "supports_per_pass_loras": False,
                        "supports_start_frame": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    actor = project / "actor.png"
    location = project / "location.png"
    actor.write_bytes(b"actor")
    location.write_bytes(b"location")
    state.plan_for_next_step.parent.mkdir(parents=True, exist_ok=True)
    state.plan_for_next_step.write_text(
        json.dumps(
            [
                _consistent_music_scene(1, actor, location),
                _consistent_music_scene(
                    2,
                    actor,
                    location,
                    transition="continuous",
                ),
            ]
        ),
        encoding="utf-8",
    )


def _prepared_backend(postprocessor):
    backend = Mock()
    backend.postprocessor = postprocessor
    backend.max_render_frames = None
    backend.max_render_duration_seconds = None
    backend.render_budget_workflow_path = Path("msr.json")
    backend.round_render_frames_to_8n1 = False
    backend.asset_uploader = Mock()
    backend.model_resolver = Mock()
    backend.workflow_label = Path("msr.json")
    return backend


def _write_prepared_placeholders(
    state: PipelineRunState,
    numbers,
) -> None:
    for number in numbers:
        workflow = state.context.artifact_layout.scene_workflow(number)
        workflow.parent.mkdir(parents=True, exist_ok=True)
        workflow.write_text("{}", encoding="utf-8")
        workflow.with_name("manifest.json").write_text(
            "{}",
            encoding="utf-8",
        )


class _RecordingFramePostprocessor:
    def __init__(self):
        self.sources = []

    def extract_last_frame(self, video_path, output_path):
        source = Path(video_path).read_bytes()
        self.sources.append(source)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"frame:" + source)
        return output_path


class _SequentialPreparedRenderer:
    def __init__(self, layout, *, fail_on=None):
        self.layout = layout
        self.rendered = []
        self.fail_on = fail_on

    def render(self, workflow_path):
        number = int(Path(workflow_path).parent.name.removeprefix("scene_"))
        if number == self.fail_on:
            raise RuntimeError(f"scene {number} failed")
        self.rendered.append(number)
        output = self.layout.scene_final_video(number)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"new-{number}".encode())
        return output


if __name__ == "__main__":
    unittest.main()
