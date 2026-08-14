import unittest
import json
import io
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from rich.console import Console

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts
from feverslop.prompting.msr_signatures import MSRPromptResult

from tools.repair_scene_srt import main as repair_scene_srt_main
import movie_pipeline
import run_pipeline


class RunnerScriptTests(unittest.TestCase):
    def test_materialized_movie_workflow_actions_prepare_normal_and_debug_runs(self):
        from feverslop.composition import movie_pipeline as composition

        self.assertEqual((True, True), composition._movie_workflow_actions(False))
        self.assertEqual((True, False), composition._movie_workflow_actions(True))

    def test_vision_enriched_msr_prompts_reach_relay_inputs_with_separate_formats(self):
        class VisionLLM:
            model = "test-model"
            client = object()

        class VisionModules:
            def vision(self, _payload, _images):
                return MSRPromptResult.model_validate({
                    "references": [
                        {"id": "mara", "type": "actor", "description": "Mara has a sharp black bob and silver coat."},
                        {"id": "archive", "type": "location", "description": "The archive has amber lamps and oak shelves."},
                    ],
                    "relays": [{"index": 0, "prompt": "Mara crosses toward the desk as the camera tracks beside her and dust lifts in the warm light."}],
                })

            def segments(self, _payload):
                return MSRPromptResult(relays=[])

        class Uploader:
            def resolve_reference_image_name(self, path, **_kwargs):
                return Path(path).name

        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            (temp / "mara.png").write_bytes(b"actor")
            (temp / "archive.png").write_bytes(b"location")
            plan = temp / "plan.json"
            plan.write_text(json.dumps([{
                "scene": 1, "fps": 24, "frame_count": 49,
                "references": {
                    "actor_msr_paths": ["mara.png"], "location_msr_path": "archive.png",
                    "actor_reference_descriptions": [{"id": "mara", "name": "Mara"}],
                    "location_reference_description": {"id": "archive", "name": "Archive"},
                },
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 48, "state": "instrumental"}]},
            }]))
            with patch("feverslop.application.msr_prompt_enrichment.MSRPromptModules", return_value=VisionModules()):
                enriched = json.loads(enrich_render_plan_with_msr_prompts(
                    plan, temp / "enriched.json", llm=VisionLLM(),
                ).read_text())[0]
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps({
                "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                "3": {"inputs": {"global_prompt": "", "local_prompts": "", "segment_lengths": ""}, "_meta": {"title": "#PROMPT_RELAY"}},
                "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
            }))
            patched = ComfyUIMSRVideoRenderBackend(
                client=object(), workflow_path=workflow_path, output_dir=temp / "out",
                project_dir=temp, asset_uploader=Uploader(), postprocess=False,
            ).build_workflow(enriched, prompt="fallback")
            relay = patched["3"]["inputs"]
            self.assertIn("sharp black bob", relay["global_prompt"])
            self.assertIn("amber lamps", relay["global_prompt"])
            self.assertIn("camera tracks beside her", relay["local_prompts"])
            self.assertNotIn("camera tracks beside her", relay["global_prompt"])
            self.assertNotIn("sharp black bob", relay["local_prompts"])
            for heading in ("Reference Sheet Description", "Target Description"):
                self.assertNotIn(heading, relay["global_prompt"])
                self.assertNotIn(heading, relay["local_prompts"])

    def test_os_specific_runner_scripts_are_removed(self):
        self.assertFalse(Path("test.ps1").exists())
        self.assertFalse(Path("test.bat").exists())

    def test_run_pipeline_parser_defaults_match_python_runner_contract(self):
        args = run_pipeline.build_arg_parser().parse_args([])

        self.assertIsNone(args.project_root)
        self.assertIsNone(args.project_config)
        self.assertEqual("app_config.json", args.app_config)
        self.assertEqual(10, args.concept_batch_size)
        self.assertEqual(os.fspath(Path("workflows") / "image_t2i_startframe_v1.json"), args.storyboard_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "image_t2i_startframe_krea_v1.json"), args.reference_hero_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "image_edit_flux2_klein_1ref_v1.json"), args.reference_edit_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "video_ltxv_msr_1actor_1background_v4.json"), args.msr_workflow)
        self.assertEqual("", args.relay_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "video_ltxv_i2v_v2.json"), args.single_prompt_workflow)
        self.assertEqual("ltx_i2v", args.video_pipeline)
        self.assertEqual("single_prompt", args.render_mode)
        self.assertEqual("#PROMPT", args.single_prompt_title)
        self.assertEqual("text", args.single_prompt_input)
        self.assertEqual("original", args.rolling_frame_profile)
        self.assertFalse(args.randomize_seed)
        self.assertEqual(16, args.smoke_scene)
        self.assertFalse(args.skip_main_pipeline)

    def test_run_pipeline_parser_accepts_powershell_parity_flags(self):
        args = run_pipeline.build_arg_parser().parse_args(
            [
                "projects/song",
                "--render-mode",
                "auto",
                "--relay-workflow",
                "relay.json",
                "--single-prompt-workflow",
                "single.json",
                "--video-pipeline",
                "ltx_msr",
                "--reference-hero-workflow",
                "hero.json",
                "--reference-edit-workflow",
                "edit.json",
                "--msr-workflow",
                "msr.json",
                "--skip-msr-reference-render",
                "--storyboard-lora-strength",
                "0.4",
                "--video-character-lora-strength",
                "0.8",
                "--video-lora-1-strength-model",
                "0.7",
                "--video-lora-1-strength-clip",
                "0.6",
                "--lora-split-enabled",
                "--randomize-seed",
                "--smoke-only",
                "--no-skip-existing",
                "--skip-tests",
                "--skip-main-pipeline",
                "--skip-relay-compact",
                "--skip-anchor-fix",
                "--skip-storyboard",
                "--skip-storyboard-page",
                "--skip-ltx",
                "--skip-final-concat",
                "--diagnostic-original-audio-mux",
                "--no-original-audio-mux",
            ]
        )

        self.assertEqual("projects/song", args.project_root)
        self.assertEqual("auto", args.render_mode)
        self.assertEqual("ltx_msr", args.video_pipeline)
        self.assertEqual("hero.json", args.reference_hero_workflow)
        self.assertEqual("edit.json", args.reference_edit_workflow)
        self.assertEqual("msr.json", args.msr_workflow)
        self.assertTrue(args.skip_msr_reference_render)
        self.assertEqual(0.4, args.storyboard_lora_strength)
        self.assertEqual(0.8, args.video_character_lora_strength)
        self.assertEqual(0.7, args.video_lora_1_strength_model)
        self.assertEqual(0.6, args.video_lora_1_strength_clip)
        self.assertTrue(args.lora_split_enabled)
        self.assertTrue(args.randomize_seed)
        self.assertTrue(args.smoke_only)
        self.assertTrue(args.no_skip_existing)
        self.assertTrue(args.skip_tests)
        self.assertTrue(args.skip_main_pipeline)
        self.assertTrue(args.skip_relay_compact)
        self.assertTrue(args.skip_anchor_fix)
        self.assertTrue(args.skip_storyboard)
        self.assertTrue(args.skip_storyboard_page)
        self.assertTrue(args.skip_ltx)
        self.assertTrue(args.skip_final_concat)
        self.assertTrue(args.diagnostic_original_audio_mux)
        self.assertTrue(args.no_original_audio_mux)

    def test_repair_scene_srt_cli_writes_repaired_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_srt = temp / "input.srt"
            output_srt = temp / "output.srt"
            input_srt.write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nScene 1\n\n"
                "2\n00:00:00,500 --> 00:00:02,000\nScene 2\n",
                encoding="utf-8",
            )

            argv = [
                "repair_scene_srt.py",
                "--input-srt",
                str(input_srt),
                "--output-srt",
                str(output_srt),
                "--min-duration",
                "1.0",
                "--max-duration",
                "3.0",
            ]
            with patch.object(sys, "argv", argv):
                repair_scene_srt_main()

            self.assertTrue(output_srt.exists())
            self.assertIn("00:00:00,000 --> 00:00:02,000", output_srt.read_text(encoding="utf-8"))

    def test_movie_pipeline_parser_accepts_skip_stage_flags(self):
        args = movie_pipeline.build_arg_parser().parse_args(
            [
                "projects/tm3",
                "--reference-backend",
                "local",
                "--render-backend",
                "local",
                "--skip-movie-bible",
                "--force-movie-bible",
                "--movie-planner-backend",
                "llm",
                "--skip-movie-story-design",
                "--force-movie-story-design",
                "--skip-movie-screenplay",
                "--force-movie-screenplay",
                "--skip-movie-narrative",
                "--skip-movie-scene-cards",
                "--skip-movie-shot-cards",
                "--skip-movie-continuity",
                "--skip-movie-plan",
                "--skip-movie-references",
                "--skip-movie-msr-enrich",
                "--skip-movie-render",
                "--force-movie-references",
                "--keyframe-mode",
                "start",
                "--movie-video-workflow",
                "msr-i2v-startframe",
                "--continuity-keyframes",
                "last-to-start",
                "--msr-i2v-workflow",
                "msr-i2v.json",
                "--write-debug-workflows",
                "--debug-workflows-dir",
                "debug/workflows",
            ]
        )

        self.assertEqual("projects/tm3", args.project_dir)
        self.assertEqual("local", args.reference_backend)
        self.assertEqual("local", args.render_backend)
        self.assertTrue(args.skip_movie_bible)
        self.assertTrue(args.force_movie_bible)
        self.assertEqual("llm", args.movie_planner_backend)
        self.assertTrue(args.skip_movie_story_design)
        self.assertTrue(args.force_movie_story_design)
        self.assertTrue(args.skip_movie_screenplay)
        self.assertTrue(args.force_movie_screenplay)
        self.assertTrue(args.skip_movie_narrative)
        self.assertTrue(args.skip_movie_scene_cards)
        self.assertTrue(args.skip_movie_shot_cards)
        self.assertTrue(args.skip_movie_continuity)
        self.assertTrue(args.skip_movie_plan)
        self.assertTrue(args.skip_movie_references)
        self.assertTrue(args.skip_movie_msr_enrich)
        self.assertTrue(args.skip_movie_render)
        self.assertTrue(args.force_movie_references)
        self.assertEqual("start", args.keyframe_mode)
        self.assertEqual("msr-i2v-startframe", args.movie_video_workflow)
        self.assertEqual("last-to-start", args.continuity_keyframes)
        self.assertEqual("msr-i2v.json", args.msr_i2v_workflow)
        self.assertTrue(args.write_debug_workflows)
        self.assertEqual("debug/workflows", args.debug_workflows_dir)

    def test_movie_pipeline_maps_i2v_msr_workflow_argument_to_i2v_workflow_for_compatibility(self):
        args = movie_pipeline.build_arg_parser().parse_args(
            [
                "projects/tm3",
                "--movie-video-workflow",
                "msr-i2v-startframe",
                "--continuity-keyframes",
                "last-to-start",
                "--msr-workflow",
                "workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json",
            ]
        )

        config = movie_pipeline.config_from_args(args)

        self.assertEqual("workflows/video_default_ltxv_msr_1actor_1background_v4.json", config["msr_workflow"])
        self.assertEqual("workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json", config["msr_i2v_workflow"])

    def test_movie_pipeline_accepts_i2v_edit_workflow_mode(self):
        args = movie_pipeline.build_arg_parser().parse_args(
            [
                "projects/demo",
                "--movie-video-workflow",
                "i2v-edit",
            ]
        )

        config = movie_pipeline.config_from_args(args)

        self.assertEqual("i2v-edit", args.movie_video_workflow)
        self.assertEqual("i2v-edit", config["movie_video_workflow"])
        self.assertEqual("workflows/image_t2i_startframe_krea_v1.json", config["hero_workflow"])
        self.assertEqual("workflows/image_edit_flux2_klein_2ref_v1.json", config["edit_workflow"])
        self.assertEqual("workflows/video_ltxv_i2v_v2.json", config["i2v_workflow"])

    def test_movie_pipeline_accepts_startframe_director_workflow_mode(self):
        args = movie_pipeline.build_arg_parser().parse_args(
            [
                "projects/demo",
                "--movie-video-workflow",
                "startframe-director",
            ]
        )

        config = movie_pipeline.config_from_args(args)

        self.assertEqual("startframe-director", args.movie_video_workflow)
        self.assertEqual("startframe-director", config["movie_video_workflow"])
        self.assertEqual("krea2", config["startframe_director_backend"])
        self.assertEqual("workflows/image_t2i_startframe_krea_v1.json", config["director_workflow"])
        self.assertEqual("workflows/image_mask_sam3_actor_regions_v1.json", config["mask_workflow"])
        self.assertEqual("workflows/image_repair_sdxl_ipadapter_identity_v1.json", config["identity_repair_workflow"])
        self.assertEqual("workflows/image_detail_easyuse_startframe_v1.json", config["detail_workflow"])
        self.assertEqual("http://your-llm-server.local/v1", config["startframe_validator_base_url"])
        self.assertEqual("gemma4-26b-a4b:vision", config["startframe_validator_model"])
        self.assertFalse(config["startframe_write_debug_workflows"])
        self.assertEqual("workflows/video_ltxv_i2v_native_audio_v2.json", config["i2v_workflow"])

    def test_movie_pipeline_accepts_startframe_validator_overrides(self):
        args = movie_pipeline.build_arg_parser().parse_args(
            [
                "projects/demo",
                "--movie-video-workflow",
                "startframe-director",
                "--startframe-comfyui-base-url",
                "http://localhost:8188/",
                "--startframe-validator-base-url",
                "http://your-llm-server.local/v1/",
                "--startframe-validator-model",
                "gemma4-26b-a4b",
            ]
        )

        config = movie_pipeline.config_from_args(args)

        self.assertEqual("http://localhost:8188", config["startframe_comfyui_base_url"])
        self.assertEqual("http://your-llm-server.local/v1", config["startframe_validator_base_url"])
        self.assertEqual("gemma4-26b-a4b", config["startframe_validator_model"])

    def test_movie_pipeline_accepts_ideogram_startframe_director_backend(self):
        args = movie_pipeline.build_arg_parser().parse_args(
            [
                "projects/demo",
                "--movie-video-workflow",
                "startframe-director",
                "--startframe-director-backend",
                "ideogram",
            ]
        )

        config = movie_pipeline.config_from_args(args)

        self.assertEqual("ideogram", config["startframe_director_backend"])
        self.assertEqual("workflows/image_t2i_startframe_ideogram_director_v1.json", config["director_workflow"])

    def test_movie_pipeline_cli_can_run_references_only_with_local_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir))

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args(
                    [
                        str(project),
                        "--reference-backend",
                        "local",
                        "--render-backend",
                        "local",
                        "--skip-movie-render",
                    ]
                )
            )

            manifest = json.loads((project / "movie" / "references" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(project.resolve(), result.project_dir)
            self.assertEqual(project / "movie" / "references" / "manifest.json", result.reference_manifest_path)
            self.assertIsNone(result.final_video_path)
            self.assertEqual("local", manifest["generator_backend"])
            self.assertEqual("movie/references/actors/mara/views/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertTrue((project / "movie" / "render_plan_msr.json").exists())

    def test_movie_pipeline_cli_can_skip_references_and_render_with_local_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)

            with patch("builtins.print") as print_mock:
                result = movie_pipeline.run(
                    movie_pipeline.build_arg_parser().parse_args(
                        [
                            str(project),
                            "--reference-backend",
                            "local",
                            "--render-backend",
                            "local",
                            "--skip-movie-references",
                        ]
                    )
                )

            self.assertEqual(project / "output" / "movie" / "test-movie.mp4", result.final_video_path)
            self.assertTrue(result.final_video_path.exists())
            self.assertTrue(
                any("Rendered movie clip 1/" in str(call.args[0]) for call in print_mock.call_args_list)
            )

    def test_movie_pipeline_uses_deterministic_enrichment_without_llm_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = _write_movie_project(root, ready=True)

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args([
                    str(project), "--app-config", str(root / "missing-app-config.json"),
                    "--reference-backend", "local", "--render-backend", "local",
                    "--skip-movie-references",
                ])
            )

            self.assertTrue(result.render_plan_msr_path.is_file())
            self.assertTrue((project / "movie" / "render_plan_ingredients.json").is_file())

    def test_movie_pipeline_i2v_edit_local_backend_writes_i2v_plan_and_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args(
                    [
                        str(project),
                        "--movie-video-workflow",
                        "i2v-edit",
                        "--reference-backend",
                        "local",
                        "--render-backend",
                        "local",
                        "--skip-movie-bible",
                        "--skip-movie-story-design",
                        "--skip-movie-screenplay",
                        "--skip-movie-narrative",
                        "--skip-movie-scene-cards",
                        "--skip-movie-shot-cards",
                        "--skip-movie-continuity",
                        "--skip-movie-plan",
                        "--skip-movie-references",
                    ]
                )
            )

            self.assertEqual(project / "movie" / "visual_plan.json", result.visual_plan_path)
            self.assertEqual(project / "movie" / "render_plan_i2v.json", result.render_plan_i2v_path)
            self.assertEqual(project / "output" / "movie" / "test-movie.mp4", result.final_video_path)
            self.assertTrue(result.final_video_path.exists())
            self.assertTrue((project / "output" / "movie" / "storyboard" / "index.html").exists())

    def test_movie_pipeline_startframe_director_local_backend_writes_contracts_and_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args(
                    [
                        str(project),
                        "--movie-video-workflow",
                        "startframe-director",
                        "--reference-backend",
                        "local",
                        "--render-backend",
                        "local",
                        "--skip-movie-bible",
                        "--skip-movie-story-design",
                        "--skip-movie-screenplay",
                        "--skip-movie-narrative",
                        "--skip-movie-scene-cards",
                        "--skip-movie-shot-cards",
                        "--skip-movie-continuity",
                        "--skip-movie-plan",
                        "--skip-movie-references",
                    ]
                )
            )

            self.assertEqual(project / "movie" / "identity_ledger.json", result.identity_ledger_path)
            self.assertEqual(project / "movie" / "startframe_plan.json", result.startframe_plan_path)
            self.assertEqual(project / "movie" / "startframe_director_prompts.json", result.startframe_director_prompts_path)
            prompts = json.loads(result.startframe_director_prompts_path.read_text(encoding="utf-8"))
            self.assertEqual("krea2", prompts["shots"][0]["director_backend"])
            self.assertTrue((project / "movie" / "startframe_validation.json").exists())
            self.assertTrue((project / "output" / "movie" / "storyboard" / "final" / "scene_0001.png").exists())
            self.assertEqual(project / "output" / "movie" / "test-movie.mp4", result.final_video_path)

    def test_movie_pipeline_i2v_edit_prints_rich_stage_logs(self):
        from feverslop.composition import movie_pipeline as movie_pipeline_module

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, color_system=None, width=120)

            with patch.object(movie_pipeline_module, "console", console, create=True):
                movie_pipeline.run(
                    movie_pipeline.build_arg_parser().parse_args(
                        [
                            str(project),
                            "--movie-video-workflow",
                            "i2v-edit",
                            "--reference-backend",
                            "local",
                            "--render-backend",
                            "local",
                            "--skip-movie-bible",
                            "--skip-movie-story-design",
                            "--skip-movie-screenplay",
                            "--skip-movie-narrative",
                            "--skip-movie-scene-cards",
                            "--skip-movie-shot-cards",
                            "--skip-movie-continuity",
                            "--skip-movie-plan",
                            "--skip-movie-references",
                            "--skip-openshot-export",
                        ]
                    )
                )

            output = buffer.getvalue()

        self.assertIn("Movie visual plan", output)
        self.assertIn("Movie I2V render plan", output)
        self.assertIn("Storyboard review page", output)
        self.assertIn("Movie complete", output)

    def test_movie_pipeline_i2v_edit_prints_stage_progress_counts(self):
        from feverslop.composition import movie_pipeline as movie_pipeline_module

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, color_system=None, width=160)

            with patch.object(movie_pipeline_module, "console", console, create=True):
                movie_pipeline.run(
                    movie_pipeline.build_arg_parser().parse_args(
                        [
                            str(project),
                            "--movie-video-workflow",
                            "i2v-edit",
                            "--reference-backend",
                            "local",
                            "--render-backend",
                            "local",
                            "--skip-movie-bible",
                            "--skip-movie-story-design",
                            "--skip-movie-screenplay",
                            "--skip-movie-narrative",
                            "--skip-movie-scene-cards",
                            "--skip-movie-shot-cards",
                            "--skip-movie-continuity",
                            "--skip-movie-plan",
                            "--skip-movie-references",
                            "--skip-openshot-export",
                        ]
                    )
                )

            output = buffer.getvalue()

        self.assertIn("Movie pipeline stages", output)
        self.assertIn("8/8", output)
        self.assertIn("100%", output)

    def test_movie_pipeline_i2v_edit_uses_comfy_adapter_for_comfy_render_backend(self):
        class FakeAdapter:
            def __init__(self):
                self.render_plan_path = None

            def render_movie(self, *, project_dir, render_plan_path, on_clip_rendered=None, on_startframe_step=None, **_kwargs):
                self.render_plan_path = render_plan_path
                if on_startframe_step is not None:
                    on_startframe_step({"kind": "edit", "completed": 2, "total": 15, "scene": 1, "actor_id": "morwenna"})
                final = project_dir / "output" / "movie" / "comfy.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_bytes(b"mp4")
                return final

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            adapter = FakeAdapter()
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, color_system=None, width=160)

            with (
                patch("feverslop.composition.movie_pipeline._build_i2v_edit_visual_adapter", return_value=adapter, create=True) as builder,
                patch("feverslop.composition.movie_pipeline.console", console, create=True),
            ):
                result = movie_pipeline.run(
                    movie_pipeline.build_arg_parser().parse_args(
                        [
                            str(project),
                            "--movie-video-workflow",
                            "i2v-edit",
                            "--reference-backend",
                            "local",
                            "--render-backend",
                            "comfyui",
                            "--skip-movie-bible",
                            "--skip-movie-story-design",
                            "--skip-movie-screenplay",
                            "--skip-movie-narrative",
                            "--skip-movie-scene-cards",
                            "--skip-movie-shot-cards",
                            "--skip-movie-continuity",
                            "--skip-movie-plan",
                            "--skip-movie-references",
                            "--skip-openshot-export",
                        ]
                    )
                )
            output = buffer.getvalue()

        builder.assert_called_once()
        self.assertEqual(project / "movie" / "render_plan_i2v.json", adapter.render_plan_path)
        self.assertEqual(project / "output" / "movie" / "comfy.mp4", result.final_video_path)
        self.assertIn("Movie startframe: rendered edit 2/15: scene 1 actor morwenna", output)

    def test_movie_pipeline_startframe_director_uses_comfy_adapter_for_comfy_render_backend(self):
        class FakeAdapter:
            def __init__(self):
                self.render_plan_path = None

            def render_movie(self, *, project_dir, render_plan_path, on_clip_rendered=None, on_startframe_step=None, **_kwargs):
                self.render_plan_path = render_plan_path
                if on_startframe_step is not None:
                    on_startframe_step({"kind": "repair", "completed": 3, "total": 7, "scene": 1, "actor_id": "mara"})
                final = project_dir / "output" / "movie" / "startframe-comfy.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_bytes(b"mp4")
                return final

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            adapter = FakeAdapter()
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, color_system=None, width=160)

            with (
                patch("feverslop.composition.movie_pipeline._build_startframe_director_visual_adapter", return_value=adapter, create=True) as builder,
                patch("feverslop.composition.movie_pipeline.console", console, create=True),
            ):
                result = movie_pipeline.run(
                    movie_pipeline.build_arg_parser().parse_args(
                        [
                            str(project),
                            "--movie-video-workflow",
                            "startframe-director",
                            "--reference-backend",
                            "local",
                            "--render-backend",
                            "comfyui",
                            "--skip-movie-bible",
                            "--skip-movie-story-design",
                            "--skip-movie-screenplay",
                            "--skip-movie-narrative",
                            "--skip-movie-scene-cards",
                            "--skip-movie-shot-cards",
                            "--skip-movie-continuity",
                            "--skip-movie-plan",
                            "--skip-movie-references",
                            "--skip-openshot-export",
                        ]
                    )
                )
            output = buffer.getvalue()

        builder.assert_called_once()
        self.assertEqual(project / "movie" / "render_plan_i2v.json", adapter.render_plan_path)
        self.assertEqual(project / "output" / "movie" / "startframe-comfy.mp4", result.final_video_path)
        self.assertIn("Movie startframe-director render: rendered repair 3/7: scene 1 actor mara", output)

    def test_movie_pipeline_startframe_director_passes_debug_workflows_dir_to_comfy_adapter(self):
        class FakeAdapter:
            def __init__(self, **_kwargs):
                pass

            def render_movie(self, *, project_dir, render_plan_path, **_kwargs):
                final = project_dir / "output" / "movie" / "startframe-comfy.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_bytes(b"mp4")
                return final

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            debug_dir = project / "debug" / "startframe-workflows"

            with patch("feverslop.composition.movie_pipeline._build_startframe_director_visual_adapter", return_value=FakeAdapter(), create=True) as builder:
                result = movie_pipeline.run(
                    movie_pipeline.build_arg_parser().parse_args(
                        [
                            str(project),
                            "--movie-video-workflow",
                            "startframe-director",
                            "--reference-backend",
                            "local",
                            "--render-backend",
                            "comfyui",
                            "--skip-movie-bible",
                            "--skip-movie-story-design",
                            "--skip-movie-screenplay",
                            "--skip-movie-narrative",
                            "--skip-movie-scene-cards",
                            "--skip-movie-shot-cards",
                            "--skip-movie-continuity",
                            "--skip-movie-plan",
                            "--skip-movie-references",
                            "--skip-openshot-export",
                            "--write-debug-workflows",
                            "--debug-workflows-dir",
                            str(debug_dir),
                        ]
                    )
                )

        config = builder.call_args.args[1]
        self.assertTrue(config["startframe_write_debug_workflows"])
        self.assertEqual(debug_dir, Path(config["startframe_debug_workflows_dir"]))
        self.assertEqual(debug_dir, result.debug_workflows_dir)

    def test_startframe_director_ltx_handoff_uses_empty_audio_i2v_workflow(self):
        captured = {}

        class FakeAdapter:
            def __init__(self, **_kwargs):
                pass

            def render_movie(self, *, project_dir, render_plan_path, **_kwargs):
                final = project_dir / "output" / "movie" / "startframe-comfy.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_bytes(b"mp4")
                return final

        class FakeComfyClient:
            def __init__(self, *args, **kwargs):
                pass

        class FakeValidator:
            def __init__(self, *args, **kwargs):
                pass

        def fake_use_case(options):
            captured["workflow_path"] = Path(options.workflow_path)
            captured["single_prompt_workflow_path"] = Path(options.single_prompt_workflow_path)
            captured["debug_workflows_dir"] = options.debug_workflows_dir
            return object()

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            debug_dir = project / "debug" / "startframe-workflows"

            with (
                patch("feverslop.adapters.startframe_director_comfyui.ComfyUIStartframeDirectorVisualAdapter", FakeAdapter),
                patch("feverslop.adapters.comfyui_client.ComfyUIClient", FakeComfyClient),
                patch("feverslop.adapters.gemma4_startframe_validator.Gemma4StartframeValidator", FakeValidator),
                patch("feverslop.composition.render_video.build_render_video_scenes_use_case", side_effect=fake_use_case),
            ):
                movie_pipeline.run(
                    movie_pipeline.build_arg_parser().parse_args(
                        [
                            str(project),
                            "--movie-video-workflow",
                            "startframe-director",
                            "--reference-backend",
                            "local",
                            "--render-backend",
                            "comfyui",
                            "--skip-movie-bible",
                            "--skip-movie-story-design",
                            "--skip-movie-screenplay",
                            "--skip-movie-narrative",
                            "--skip-movie-scene-cards",
                            "--skip-movie-shot-cards",
                            "--skip-movie-continuity",
                            "--skip-movie-plan",
                            "--skip-movie-references",
                            "--skip-openshot-export",
                            "--write-debug-workflows",
                            "--debug-workflows-dir",
                            str(debug_dir),
                        ]
                    )
                )

            self.assertEqual(captured["workflow_path"], captured["single_prompt_workflow_path"])
            self.assertEqual(debug_dir, Path(captured["debug_workflows_dir"]))
            self.assertEqual(project / "output" / "movie" / "startframes" / "workflows" / "ltx_i2v_empty_audio.json", captured["workflow_path"])
            workflow = json.loads(captured["workflow_path"].read_text(encoding="utf-8"))
            classes = {node.get("class_type") for node in workflow.values()}
            self.assertNotIn("LoadAudio", classes)
            self.assertIn("LTXVEmptyLatentAudio", classes)

    def test_movie_pipeline_cli_can_write_debug_workflows_without_rendering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            debug_dir = project / "debug_workflows"

            with self.assertRaisesRegex(ValueError, "requires the ComfyUI render backend"):
                movie_pipeline.run(movie_pipeline.build_arg_parser().parse_args(
                    [
                        str(project),
                        "--reference-backend",
                        "local",
                        "--render-backend",
                        "local",
                        "--skip-movie-references",
                        "--write-debug-workflows",
                        "--debug-workflows-dir",
                        str(debug_dir),
                        "--skip-movie-render",
                    ]
                ))

    def test_movie_pipeline_cli_accepts_strict_scene_selection(self):
        args = movie_pipeline.build_arg_parser().parse_args(["projects/example", "--scenes", "1,3,5"])

        self.assertEqual([1, 3, 5], args.scenes)

    def test_movie_pipeline_can_regenerate_openshot_project_as_standalone_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            (project / "config.json").write_text(
                json.dumps({"project_name": "Test Movie", "input_audio": "", "video": {"fps": 24, "width": 1280, "height": 704}}),
                encoding="utf-8",
            )
            clip = project / "output" / "movie" / "minimax-h3-r2v" / "final" / "scene_0001.mp4"
            clip.parent.mkdir(parents=True)
            clip.touch()

            result = movie_pipeline.run(movie_pipeline.build_arg_parser().parse_args([
                str(project), "--stage", "openshot_export", "--movie-video-workflow", "minimax-h3-r2v",
            ]))

            output = project / "output" / "movie" / "openshot" / "test-movie.osp"
            self.assertEqual(project / "movie" / "render_plan.json", result.render_plan_path)
            self.assertTrue(output.is_file())
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("../minimax-h3-r2v/final/scene_0001.mp4", exported["files"][0]["path"])

    def test_movie_pipeline_debug_workflows_relative_dir_uses_cwd(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            project = _write_movie_project(root, ready=True)
            debug_dir = root.relative_to(Path.cwd()) / "debug_workflows"

            with self.assertRaisesRegex(ValueError, "requires the ComfyUI render backend"):
                movie_pipeline.run(movie_pipeline.build_arg_parser().parse_args(
                    [
                        str(project),
                        "--reference-backend",
                        "local",
                        "--render-backend",
                        "local",
                        "--skip-movie-references",
                        "--write-debug-workflows",
                        "--debug-workflows-dir",
                        str(debug_dir),
                        "--skip-movie-render",
                    ]
                ))
            self.assertFalse((root / "debug_workflows").exists())


def _write_movie_project(root: Path, *, ready: bool = False) -> Path:
    project = root / "test-movie"
    references = project / "movie" / "references"
    references.mkdir(parents=True)
    (project / "movie" / "render_plan.json").write_text(
        json.dumps(
            {
                "title": "Test Movie",
                "resolution": {"width": 1280, "height": 704},
                "shots": [
                    {
                        "shot_id": "shot_0001",
                        "description": "Mara enters the archive",
                        "duration_seconds": 1,
                        "reference_ids": {"actors": ["mara"], "location": "archive"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (project / "movie" / "bible.json").write_text(
        json.dumps(
            {
                "title": "Test Movie",
                "actors": [
                    {"id": "mara", "name": "Mara", "visual_description": "gothic archivist"},
                ],
                "locations": [
                    {"id": "archive", "name": "Archive", "visual_description": "dusty archive room"},
                ],
                "runtime_constraints": {"fps": 24},
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "story_design.json",
        "screenplay.json",
        "narrative_plan.json",
        "scene_cards.json",
        "shot_cards.json",
        "continuity_plan.json",
    ):
        (project / "movie" / name).write_text("{}", encoding="utf-8")
    manifest = {
        "actors": [
            {
                "id": "mara",
                "name": "Mara",
                "visual_description": "gothic archivist",
                "image_prompt": "Full-body cinematic character reference sheet for Mara. gothic archivist. Four vertical panels in one image.",
                "prompt": "Full-body cinematic character reference sheet for Mara. gothic archivist. Four vertical panels in one image.",
                "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png" if ready else "",
            }
        ],
        "locations": [
            {
                "id": "archive",
                "name": "Archive",
                "prompt": "Archive",
                "msr_sheet_path": "movie/references/locations/archive/views/hero.png" if ready else "",
            }
        ],
        "generator_backend": "local" if ready else "",
    }
    (references / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project


if __name__ == "__main__":
    unittest.main()
