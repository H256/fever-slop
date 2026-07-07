import unittest
import json
import io
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from rich.console import Console

from tools.repair_scene_srt import main as repair_scene_srt_main
import movie_pipeline
import run_pipeline


class RunnerScriptTests(unittest.TestCase):
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
        self.assertEqual(os.fspath(Path("workflows") / "video_ltxv_msr_1actor_1background_v1.json"), args.msr_workflow)
        self.assertEqual("", args.relay_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "video_ltxv_i2v_v1.json"), args.single_prompt_workflow)
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
                "workflows/video_default_i2v_ltxv_msr_1actor_1background_v1.json",
            ]
        )

        config = movie_pipeline.config_from_args(args)

        self.assertEqual("workflows/video_default_ltxv_msr_1actor_1background_v1.json", config["msr_workflow"])
        self.assertEqual("workflows/video_default_i2v_ltxv_msr_1actor_1background_v1.json", config["msr_i2v_workflow"])

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
        self.assertEqual("workflows/video_ltxv_i2v_v1.json", config["i2v_workflow"])

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
            self.assertEqual("movie/references/actors/mara/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
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
                        ]
                    )
                )

            output = buffer.getvalue()

        self.assertIn("Movie visual plan", output)
        self.assertIn("Movie I2V render plan", output)
        self.assertIn("Storyboard review page", output)
        self.assertIn("Movie complete", output)

    def test_movie_pipeline_i2v_edit_uses_comfy_adapter_for_comfy_render_backend(self):
        class FakeAdapter:
            def __init__(self):
                self.render_plan_path = None

            def render_movie(self, *, project_dir, render_plan_path, on_clip_rendered=None, **_kwargs):
                self.render_plan_path = render_plan_path
                final = project_dir / "output" / "movie" / "comfy.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_bytes(b"mp4")
                return final

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            adapter = FakeAdapter()

            with patch("feverslop.composition.movie_pipeline._build_i2v_edit_visual_adapter", return_value=adapter, create=True) as builder:
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
                        ]
                    )
                )

        builder.assert_called_once()
        self.assertEqual(project / "movie" / "render_plan_i2v.json", adapter.render_plan_path)
        self.assertEqual(project / "output" / "movie" / "comfy.mp4", result.final_video_path)

    def test_movie_pipeline_cli_can_write_debug_workflows_without_rendering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)
            debug_dir = project / "debug_workflows"

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args(
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
                )
            )

            debug_workflow = json.loads((debug_dir / "scene_0001_workflow.json").read_text(encoding="utf-8"))
            actor_node = next(node for node in debug_workflow.values() if node.get("_meta", {}).get("title") == "#MSR_ACTOR_1")
            location_node = next(node for node in debug_workflow.values() if node.get("_meta", {}).get("title") == "#MSR_BACKGROUND")
            relay_node = next(node for node in debug_workflow.values() if node.get("_meta", {}).get("title") == "#PROMPT_RELAY")
            self.assertIsNone(result.final_video_path)
            self.assertEqual(debug_dir, result.debug_workflows_dir)
            self.assertEqual("msr_sheet.png", actor_node["inputs"]["image"])
            self.assertEqual("hero.png", location_node["inputs"]["image"])
            self.assertIn("Mara enters the archive", relay_node["inputs"]["local_prompts"])

    def test_movie_pipeline_debug_workflows_relative_dir_uses_cwd(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            project = _write_movie_project(root, ready=True)
            debug_dir = root.relative_to(Path.cwd()) / "debug_workflows"

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args(
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
                )
            )

            self.assertEqual(root / "debug_workflows", result.debug_workflows_dir)
            self.assertTrue((root / "debug_workflows" / "scene_0001_workflow.json").exists())
            self.assertFalse((project / "debug_workflows" / "scene_0001_workflow.json").exists())


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
